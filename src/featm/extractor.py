"""Public, leakage-safe fit/transform API for stateful out-of-core features.

``featm.extract()`` (see :mod:`featm.engine`/:mod:`featm.execution`) fits and
transforms *the same source* in one call -- correct and convenient for the
common "engineer features for this one dataset" case, but it cannot express
"fit on train, transform train and test using only the fitted training
state" without recomputing (and therefore leaking) a fresh fit on whatever
source is passed. That gap was Finding 4 in the third independent audit
(``STEP2_AUDIT3_FIX_HANDOFF.md``): the internal :mod:`featm.state`
``StatefulEncoder`` protocol was already leakage-safe, but nothing public
exposed the fitted state for reuse on a different source.

This module closes that gap without inventing a new execution model: it
reuses the exact same :func:`featm.execution._prepare_run` /
:func:`featm.execution._run_fit_phase` /
:func:`featm.execution._run_transform_phase` machinery ``run_extraction``
itself calls, just invoked twice against two independently-prepared sources
instead of once against one.

    fitted = featm.fit(train_source, features=[...], columns={...})
    train_F = fitted.transform(train_source)
    test_F = fitted.transform(test_source)   # frozen training state only

Or the class form, for reuse across several fits with the same configuration:

    extractor = featm.FeatureExtractor(features=[...], columns={...})
    fitted = extractor.fit(train_source)
    test_F = fitted.transform(test_source)

Scope, stated plainly (see STEP2_AUDIT3_FIX_HANDOFF.md for the full
rationale): unlike ``checkpoint=`` on :func:`featm.extract`, ``.fit()`` here
does not support incremental checkpoint/resume of the fit phase itself -- it
runs the fit scan to completion in one call (still out-of-core: bounded
batching, adaptive memory retry, and the same resource budget all apply).
:meth:`FittedFeatureExtractor.save`/:func:`load_fitted` persist the
*finished* fitted-state artifact for later reuse, which is a different
(simpler, and sufficient for the leakage-safety requirement) guarantee than
resuming a partially-completed fit.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution import (
    DEFAULT_APPROXIMATE_SAMPLE_SIZE,
    PreparedRun,
    UnsafeMultiBatchFeatureError,
    _prepare_run,
    _resolve_output_backend,
    _run_fit_phase,
    _run_transform_phase,
)
from .matrix import FeatureMatrix
from .report import ExecutionReport

_ARTIFACT_FORMAT_VERSION = 2


class FittedFeatureExtractor:
    """Frozen fitted state from one :meth:`FeatureExtractor.fit` call, reusable across sources.

    Never mutated by :meth:`transform` -- the same fitted artifact can
    transform any number of different sources (or the same source
    repeatedly) and always applies the identical frozen state, exactly the
    leakage-safety guarantee a real train/test split requires.
    """

    def __init__(
        self,
        *,
        features: Sequence[str],
        modality: str | None,
        columns: Mapping[str, Any] | None,
        params: Mapping[str, Mapping[str, Any]] | None,
        mode: str,
        errors: str,
        memory_budget: "str | int | None",
        disk_budget: "str | int | None",
        cache_budget: "str | int | None",
        approximate_sample_size: int,
        fitted_states: dict[str, Any],
        failed_stateful: set[str],
        fit_report: ExecutionReport,
    ) -> None:
        self._features = list(features)
        self._modality = modality
        self._columns = dict(columns) if columns else None
        self._params = {k: dict(v) for k, v in (params or {}).items()}
        self._mode = mode
        self._errors = errors
        self._memory_budget = memory_budget
        self._disk_budget = disk_budget
        self._cache_budget = cache_budget
        self._approximate_sample_size = approximate_sample_size
        #: Fitted state per stateful feature name, frozen at fit time. Never
        #: mutated by transform() -- each transform() call passes this same
        #: dict by reference into a purely read-only transform phase
        #: (StatefulEncoder.transform_batch is documented to never mutate
        #: state), so repeated/concurrent transform() calls are all safe.
        self.fitted_states = dict(fitted_states)
        self.failed_stateful = set(failed_stateful)
        #: The ExecutionReport from the .fit() call itself (skipped/failed
        #: features during fitting, batch counts, warnings, ...) -- distinct
        #: from the report on a transform() call's returned FeatureMatrix.
        self.fit_report = fit_report

    @property
    def fitted_feature_names(self) -> list[str]:
        return sorted(self.fitted_states)

    def transform(
        self,
        source: Any,
        *,
        output: str = "auto",
        batch_rows: int | None = None,
        disk_budget: "str | int | None" = None,
        allow_memory_budget_override: bool = False,
    ) -> FeatureMatrix:
        """Apply this fitted state (plus any direct/stateless selected features) to ``source``.

        ``source`` is prepared fresh against its *own* schema (Finding 4:
        "make source identity for transform data distinct from
        training-state identity") -- a different source with the same
        column names in a different order, or read from a different file
        format, is not treated as a checkpoint mismatch the way
        ``featm.extract(..., checkpoint=...)`` would; it is simply the
        normal, expected train/test-split use case. A stateful feature
        selected here that has no fitted state (was skipped during
        ``.fit()``) is isolated per this extractor's ``errors=`` policy
        rather than silently dropped or crashing the whole call.
        """
        prepared = _prepare_run(
            source,
            features=self._features,
            modality=self._modality,
            columns=self._columns,
            mode=self._mode,
            errors=self._errors,
            memory_budget=self._memory_budget,
            disk_budget=disk_budget if disk_budget is not None else self._disk_budget,
            batch_rows=batch_rows,
            cache_budget=self._cache_budget,
            params=self._params,
        )
        use_disk, io_backend_available, disk_budget_bytes = _resolve_output_backend(
            prepared, output=output, checkpoint_mgr=None, disk_budget=disk_budget if disk_budget is not None else self._disk_budget
        )
        return _run_transform_phase(
            prepared,
            self.fitted_states,
            set(self.failed_stateful),
            errors=self._errors,
            output=output,
            use_disk=use_disk,
            io_backend_available=io_backend_available,
            disk_budget_bytes=disk_budget_bytes,
            allow_memory_budget_override=allow_memory_budget_override,
            params=self._params,
            checkpoint_mgr=None,
            manifest=None,
        )

    # --- persistence: a serializable fitted-state artifact for later reuse ---
    def save(self, path: "str | Path") -> None:
        """Persist this fitted artifact so a later process can ``load_fitted()`` and transform.

        This is a simpler guarantee than ``featm.extract(..., checkpoint=)``'s
        crash-safe, resumable, per-batch checkpoint: it is a single pickle of
        the *finished* fitted state and configuration, written atomically
        (temp file + rename), meant for "fit once, reuse the fitted model
        many times later" -- not for resuming an interrupted fit.
        """
        import os

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_format_version": _ARTIFACT_FORMAT_VERSION,
            "features": self._features,
            "modality": self._modality,
            "columns": self._columns,
            "params": self._params,
            "mode": self._mode,
            "errors": self._errors,
            "memory_budget": self._memory_budget,
            "disk_budget": self._disk_budget,
            "cache_budget": self._cache_budget,
            "approximate_sample_size": self._approximate_sample_size,
            "fitted_states": self.fitted_states,
            "failed_stateful": self.failed_stateful,
        }
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: "str | Path") -> "FittedFeatureExtractor":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        legacy = payload.get("artifact_format_version") == 1
        if legacy and (payload.get("columns") is None or any(
            name == "tabular.encoding.category_frequency" or name.startswith("text.")
            for name in payload.get("fitted_states", {})
        )):
            raise ValueError("Legacy fitted artifact lacks bound/canonical input semantics; refit with v0.5.5")
        if payload.get("artifact_format_version") not in (1, _ARTIFACT_FORMAT_VERSION):
            raise ValueError(
                f"Unsupported fitted-extractor artifact_format_version="
                f"{payload.get('artifact_format_version')!r}; this featm build supports "
                f"version {_ARTIFACT_FORMAT_VERSION}."
            )
        return cls(
            features=payload["features"],
            modality=payload["modality"],
            columns=payload["columns"],
            params=payload["params"],
            mode=payload["mode"],
            errors=payload["errors"],
            memory_budget=payload["memory_budget"],
            disk_budget=payload["disk_budget"],
            cache_budget=payload["cache_budget"],
            approximate_sample_size=payload["approximate_sample_size"],
            fitted_states=payload["fitted_states"],
            failed_stateful=payload["failed_stateful"],
            fit_report=ExecutionReport(requested_features=list(payload["features"])),
        )


class FeatureExtractor:
    """Explicit, named feature selection with a leakage-safe ``fit`` step, reusable across sources.

    Mirrors ``featm.extract()``'s selection arguments exactly (explicit
    named features, no ``features="all"``); the only difference is that
    fitting and transforming are separate calls, so a stateful feature's
    state is fit once on a training source and can be applied, frozen, to
    any number of other sources afterward.
    """

    def __init__(
        self,
        *,
        features: str | Sequence[str],
        modality: str | None = None,
        columns: Mapping[str, Any] | None = None,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        mode: str = "exact",
        errors: str = "skip",
        memory_budget: "str | int | None" = None,
        disk_budget: "str | int | None" = None,
        cache_budget: "str | int | None" = None,
        approximate_sample_size: int = DEFAULT_APPROXIMATE_SAMPLE_SIZE,
    ) -> None:
        self.features = [features] if isinstance(features, str) else list(features)
        self.modality = modality
        self.columns = columns
        self.params = params
        self.mode = mode
        self.errors = errors
        self.memory_budget = memory_budget
        self.disk_budget = disk_budget
        self.cache_budget = cache_budget
        self.approximate_sample_size = approximate_sample_size

    def fit(self, source: Any, *, batch_rows: int | None = None) -> FittedFeatureExtractor:
        """Fit every selected stateful feature on ``source`` (a training source).

        Direct/stateless selected features need no fitting and simply pass
        through to :meth:`FittedFeatureExtractor.transform` unchanged. Runs
        out-of-core (bounded batching, adaptive memory retry) but, unlike
        ``featm.extract(..., checkpoint=)``, does not support resuming a
        partially-completed fit -- see this module's docstring.
        """
        prepared = _prepare_run(
            source,
            features=self.features,
            modality=self.modality,
            columns=self.columns,
            mode=self.mode,
            errors=self.errors,
            memory_budget=self.memory_budget,
            disk_budget=self.disk_budget,
            batch_rows=batch_rows,
            cache_budget=self.cache_budget,
            params=self.params,
        )
        fitted_states, failed_stateful = _run_fit_phase(
            prepared,
            errors=self.errors,
            approximate_sample_size=self.approximate_sample_size,
            params=self.params,
            checkpoint_mgr=None,
            manifest=None,
        )
        prepared.report.finalize()
        bound_columns = {k: list(v) for k, v in prepared.resolved_columns.items()}
        return FittedFeatureExtractor(
            features=self.features,
            modality=self.modality,
            columns=bound_columns,
            params=self.params,
            mode=self.mode,
            errors=self.errors,
            memory_budget=self.memory_budget,
            disk_budget=self.disk_budget,
            cache_budget=self.cache_budget,
            approximate_sample_size=self.approximate_sample_size,
            fitted_states=fitted_states,
            failed_stateful=failed_stateful,
            fit_report=prepared.report,
        )

    def fit_transform(self, source: Any, **transform_kwargs: Any) -> FeatureMatrix:
        """Convenience: fit and transform the same source in one call (like ``featm.extract()``)."""
        return self.fit(source).transform(source, **transform_kwargs)


def fit(
    source: Any,
    *,
    features: str | Sequence[str],
    modality: str | None = None,
    columns: Mapping[str, Any] | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "exact",
    errors: str = "skip",
    memory_budget: "str | int | None" = None,
    disk_budget: "str | int | None" = None,
    batch_rows: int | None = None,
    cache_budget: "str | int | None" = None,
    approximate_sample_size: int = DEFAULT_APPROXIMATE_SAMPLE_SIZE,
) -> FittedFeatureExtractor:
    """Function-call convenience for ``FeatureExtractor(...).fit(source)``. See :mod:`featm.extractor`."""
    return FeatureExtractor(
        features=features,
        modality=modality,
        columns=columns,
        params=params,
        mode=mode,
        errors=errors,
        memory_budget=memory_budget,
        disk_budget=disk_budget,
        cache_budget=cache_budget,
        approximate_sample_size=approximate_sample_size,
    ).fit(source, batch_rows=batch_rows)


def load_fitted(path: "str | Path") -> FittedFeatureExtractor:
    """Reload a fitted-state artifact written by :meth:`FittedFeatureExtractor.save`."""
    return FittedFeatureExtractor.load(path)
