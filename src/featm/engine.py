from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import warnings

import numpy as np
import pandas as pd

from .cache import scoped_cache
from .executor import FaultTolerantExecutor, TaskResult
from .planner import FeaturePlanner, ExecutionPlan
from .registry import REGISTRY, FeatureOutput, FeatureSpec
from .contracts import validate_params, numeric_series
from .resources import ResourceBudget
from .schema import DatasetProfile, infer_schema


@dataclass
class RunReport:
    profile: DatasetProfile | None
    plan: ExecutionPlan | None
    results: list[TaskResult]
    skipped: list[tuple[str, str]]

    @property
    def computed(self) -> int:
        return sum(r.status == "ok" for r in self.results)

    @property
    def failed_or_skipped(self) -> int:
        return sum(r.status != "ok" for r in self.results) + len(self.skipped)


class AutoFeatures:
    """Legacy automatic planner retained for compatibility.

    ``featm`` is moving to an explicit timm-like registry. This planner therefore
    intentionally has no ``all`` preset; users who want deterministic behavior
    should select exact feature names with :func:`extract` / :func:`create_feature`.
    """

    def __init__(
        self,
        *,
        preset: str = "standard",
        max_output_features: int = 1_000,
        errors: str = "skip",
        include_original: bool = True,
        families: Iterable[str] | None = None,
        budget: ResourceBudget | str = "auto",
    ) -> None:
        if preset not in {"minimal", "fast", "standard", "comprehensive"}:
            raise ValueError("preset must be minimal, fast, standard, or comprehensive; featm has no automatic 'all' mode")
        self.preset = preset
        self.max_output_features = max_output_features
        self.errors = errors
        self.include_original = include_original
        self.families = set(families) if families is not None else None
        self.budget = ResourceBudget.auto() if budget == "auto" else budget
        self._state: dict[str, Any] = {}
        self._plan: ExecutionPlan | None = None
        self._profile: DatasetProfile | None = None
        self.report_: RunReport | None = None

    def plan(self, X: pd.DataFrame) -> ExecutionPlan:
        self._profile = infer_schema(X)
        planner = FeaturePlanner(REGISTRY)
        self._plan = planner.plan_tabular(
            self._profile,
            preset=self.preset,
            budget=self.budget,
            max_output_features=self.max_output_features,
            include_families=self.families,
        )
        return self._plan

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self._plan = self.plan(X)
        executor = FaultTolerantExecutor(REGISTRY, errors=self.errors)
        generated, self._state, results = executor.execute_tabular(X, self._plan, state={}, fit=True)
        self.report_ = RunReport(self._profile, self._plan, results, list(self._plan.skipped))
        return pd.concat([X, generated], axis=1) if self.include_original else generated

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._plan is None:
            raise RuntimeError("Call fit_transform or fit before transform")
        executor = FaultTolerantExecutor(REGISTRY, errors=self.errors)
        generated, _, results = executor.execute_tabular(X, self._plan, state=self._state, fit=False)
        self.report_ = RunReport(self._profile, self._plan, results, list(self._plan.skipped))
        return pd.concat([X, generated], axis=1) if self.include_original else generated

    def fit(self, X: pd.DataFrame, y=None) -> "AutoFeatures":
        self.fit_transform(X, y=y)
        return self


def _selected_specs(features: str | Iterable[str]) -> list[FeatureSpec]:
    if features == "all":
        raise ValueError(
            "featm intentionally has no features='all' mode. Select explicit feature names, "
            "or use list_features/search to discover them."
        )
    if isinstance(features, str):
        names = [features]
    else:
        names = list(features)
    if not names:
        raise ValueError("Select at least one feature")
    return [REGISTRY.get(name) for name in names]


def _expand_result(name: str, result: Any) -> dict[str, np.ndarray]:
    if isinstance(result, FeatureOutput):
        arr = result.as_array()
        names = list(result.names) if result.names is not None else None
    else:
        arr = np.asarray(result)
        names = None

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return {name: arr}
    if arr.ndim != 2:
        raise ValueError(f"{name} returned shape {arr.shape}; expected 1D or 2D")
    if names is None:
        names = [f"{i:03d}" for i in range(arr.shape[1])]
    if len(set(names)) != len(names):
        raise ValueError(f"{name} returned duplicate output names")
    return {f"{name}__{suffix}": arr[:, i] for i, suffix in enumerate(names)}


def _wants_large_data_engine(
    X, *, modality, params, mode, memory_budget, disk_budget, output, batch_rows, checkpoint, cache_budget, columns
) -> bool:
    """Decide whether to route through the Step 2 out-of-core engine.

    The simple in-memory path (a plain array/DataFrame/list with none of the
    Step 2 kwargs set) is preserved exactly as in v0.1.0: same call, same
    ``pd.DataFrame`` return type, same behavior. Passing a file path/Source,
    or opting into any Step 2 kwarg, switches to the out-of-core engine,
    which returns a :class:`featm.matrix.FeatureMatrix` instead (with a
    ``.report`` attribute) so large-data users get bounded batching,
    checkpointing, and disk-backed output without changing the simple path.
    """
    from pathlib import Path

    from .sources import Source

    if isinstance(X, (Source, str, Path)):
        return True
    return any(
        v is not None
        for v in (memory_budget, disk_budget, batch_rows, checkpoint, cache_budget, columns)
    ) or mode != "exact" or output != "auto"


def extract(
    X,
    *,
    features: str | Iterable[str],
    modality: str | None = None,
    errors: str = "skip",
    params: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "exact",
    memory_budget: "str | int | None" = None,
    disk_budget: "str | int | None" = None,
    output: str = "auto",
    batch_rows: int | None = None,
    checkpoint: "str | Any | None" = None,
    cache_budget: "str | int | None" = None,
    columns: Mapping[str, Any] | None = None,
    approximate_sample_size: int = 200_000,
    allow_memory_budget_override: bool = False,
    backend: str = "cpu",
    gpu_fallback: str = "fallback",
    device: "str | int | None" = None,
    gpu_memory_budget: "str | int | None" = None,
    execution: str = "local",
    distributed_backend: str = "dask",
    distributed_workers: int = 4,
    distributed_partition_rows: int | None = None,
    distributed_max_in_flight: int | None = None,
    distributed_task_retries: int = 1,
    distributed_scheduler_address: str | None = None,
    distributed_threads_per_worker: int = 1,
    distributed_processes: bool = True,
    distributed_allow_multi_worker_gpu: bool = False,
):
    """Extract explicitly selected features.

    ``backend`` (Step 3): ``"cpu"`` (default, unchanged v0.1-v0.2 behavior)
    never touches a GPU. ``"gpu"`` tries a GPU implementation for every
    GPU-capable selected feature (``gpu_fallback="fallback"``, the default,
    runs a CPU-only-capable feature on CPU instead; ``gpu_fallback="strict"``
    raises/isolates it per ``errors=``). ``"auto"`` uses GPU only when a real
    CUDA device is available, the feature has a verified GPU implementation,
    and the batch is large enough for GPU execution to plausibly help
    (``FeatureSpec.gpu_min_rows``) -- otherwise CPU, silently in the sense of
    "no error" but never in the sense of "unreported": see
    ``result.attrs["backend_report"]`` for exactly what backend each
    selected feature actually used and why. The canonical feature NAME never
    changes between backends (see ``docs/SCALING.md``'s GPU section) --
    ``features=["timeseries.spectral.spectral_entropy"], backend="gpu"``
    computes the identical mathematical feature ``backend="cpu"`` would,
    within ``FeatureSpec.gpu_numerical_tolerance``.

    ``execution="distributed"`` (Step 4) keeps feature semantics in FEATM and delegates only
    partition task scheduling to ``distributed_backend`` (``"dask"`` for the real adapter, ``"local"`` for
    the standard-library reference runtime used by semantic tests). The existing ``backend=`` remains the
    per-worker CPU/GPU choice -- distribution and device execution are orthogonal. Stateful distributed fits
    use an explicit state-to-state merge contract; FULL_DATASET/STRUCTURED_SAMPLE/legacy single-shot features
    are rejected rather than naively mapped across workers. ``checkpoint=`` is supported for distributed
    execution: fit-phase raw per-partition states and transform-phase output shards are each committed
    transactionally (write-then-atomic-rename, then a manifest update), keyed by a deterministic partition
    index, so a Dask task retry or a resumed run after a crash can never double-count fitted state or
    duplicate output rows -- see ``featm.distributed.checkpoint`` and ``docs/DISTRIBUTED.md``.

    There is deliberately no ``features='all'`` shortcut. A selected feature may
    emit one scalar column or a fixed vector of columns.

    For a plain in-memory array/DataFrame/list with none of the Step 2
    keyword arguments set, this is exactly the v0.1.0 behavior and returns a
    ``pandas.DataFrame``. Passing a file path, an explicit
    :class:`featm.sources.Source`, or any of ``memory_budget``/``disk_budget``/
    ``output``/``batch_rows``/``checkpoint``/``cache_budget``/``columns``/a
    non-``"exact"`` ``mode`` switches to the out-of-core engine (bounded
    batching, adaptive memory retry, checkpoint/resume, memory- or
    disk-backed output) and returns a :class:`featm.matrix.FeatureMatrix`
    with a ``.report`` attribute describing exactly what happened.

    ``memory_budget`` is a hard planner envelope for out-of-core runs,
    including an explicit ``output="memory"``: if the actual output would
    exceed it, :class:`featm.execution.UnsafeOutputModeError` is raised
    rather than silently returning an over-budget in-memory result. Pass
    ``allow_memory_budget_override=True`` to explicitly accept that risk
    instead (e.g. because you know your budget estimate is conservative).
    """
    if errors not in {"skip", "warn", "raise"}:
        raise ValueError("errors must be skip, warn, or raise")
    if execution not in {"local", "distributed"}:
        raise ValueError("execution must be 'local' or 'distributed'")
    features = features if isinstance(features, str) else list(features)
    requested_specs = _selected_specs(features)
    features = [s.name for s in requested_specs]
    extra_params = set(params or {}) - {s.name for s in requested_specs}
    if extra_params:
        raise ValueError(f"Parameters supplied for unselected features: {sorted(extra_params)}")
    for spec in requested_specs:
        validate_params(spec, (params or {}).get(spec.name, {}))

    if execution == "distributed":
        if batch_rows is not None and distributed_partition_rows is not None and batch_rows != distributed_partition_rows:
            raise ValueError("Pass only one of batch_rows= or distributed_partition_rows= for distributed execution")
        from .distributed import distributed_extract

        return distributed_extract(
            X,
            features=[features] if isinstance(features, str) else list(features),
            modality=modality,
            columns=columns,
            params=params,
            mode=mode,
            errors=errors,
            memory_budget=memory_budget,
            disk_budget=disk_budget,
            output=output,
            partition_rows=distributed_partition_rows if distributed_partition_rows is not None else batch_rows,
            distributed_backend=distributed_backend,
            n_workers=distributed_workers,
            max_in_flight=distributed_max_in_flight,
            task_retries=distributed_task_retries,
            scheduler_address=distributed_scheduler_address,
            threads_per_worker=distributed_threads_per_worker,
            processes=distributed_processes,
            backend=backend,
            gpu_fallback=gpu_fallback,
            device=device,
            gpu_memory_budget=gpu_memory_budget,
            cache_budget=cache_budget,
            allow_memory_budget_override=allow_memory_budget_override,
            checkpoint=checkpoint,
            allow_multi_worker_gpu=distributed_allow_multi_worker_gpu,
        )

    if _wants_large_data_engine(
        X, modality=modality, params=params, mode=mode, memory_budget=memory_budget,
        disk_budget=disk_budget, output=output, batch_rows=batch_rows, checkpoint=checkpoint,
        cache_budget=cache_budget, columns=columns,
    ):
        from . import execution

        matrix, report = execution.run_extraction(
            X,
            features=[features] if isinstance(features, str) else list(features),
            modality=modality,
            columns=columns,
            params=params,
            mode=mode,
            errors=errors,
            memory_budget=memory_budget,
            disk_budget=disk_budget,
            output=output,
            batch_rows=batch_rows,
            checkpoint=checkpoint,
            cache_budget=cache_budget,
            approximate_sample_size=approximate_sample_size,
            allow_memory_budget_override=allow_memory_budget_override,
            backend=backend,
            gpu_fallback=gpu_fallback,
            device=device,
            gpu_memory_budget=gpu_memory_budget,
        )
        matrix.report = report
        return matrix

    specs = _selected_specs(features)
    if modality is not None:
        wrong = [s.name for s in specs if s.modality != modality]
        if wrong:
            raise ValueError(f"Selected features do not belong to modality={modality}: {wrong}")
    elif len({s.modality for s in specs}) > 1:
        raise ValueError("Mixed-modality extraction requires separate calls or a multimodal feature")

    if backend not in ("cpu", "gpu", "auto"):
        raise ValueError("backend must be cpu, gpu, or auto")
    if gpu_fallback not in ("fallback", "strict"):
        raise ValueError("gpu_fallback must be fallback or strict")

    outputs: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    params = params or {}
    backend_report: dict[str, dict] = {}
    failures: dict[str, str] = {}
    # One cache scope per call: features selected together on the same input
    # frequently share expensive intermediates (one FFT feeding several
    # spectral descriptors, one Delaunay triangulation feeding several simplex
    # statistics, ...). Scoping per-call means nothing can leak between calls.
    # Step 3: a matching GPU-cache scope is opened alongside it whenever GPU
    # dispatch is possible (backend != "cpu"), so several GPU-capable
    # features selected together share one device-resident FFT the same way
    # the CPU path already shares one host-resident FFT -- opened ONCE here,
    # not once per feature inside dispatch_one_feature (see that function's
    # docstring for the bug this avoids).
    import contextlib

    from .gpu.cache import gpu_cache_budget_bytes, scoped_gpu_cache

    # Audit-fix pass, Finding 1: the shared GPU intermediate cache used to
    # always use a fixed 64MiB cap regardless of `gpu_memory_budget=` -- a
    # user declaring a tiny total GPU budget could still get a nominal 64MiB
    # cache allowance. Resolve a nominal budget once per call (using the
    # call-level device/gpu_memory_budget, not any one feature's specific
    # backend requirement -- individual features still separately resolve
    # and enforce their own budget inside dispatch_one_feature) purely to
    # size this shared cache so it can never exceed the declared/resolved
    # total GPU working-memory envelope.
    gpu_cache_max_bytes = None
    if backend != "cpu":
        from .gpu.memory import GPUMemoryBudget

        gpu_cache_max_bytes = gpu_cache_budget_bytes(
            GPUMemoryBudget.resolve(gpu_memory_budget=gpu_memory_budget, device=device)
        )
    gpu_cache_ctx = (
        scoped_gpu_cache(max_bytes=gpu_cache_max_bytes) if backend != "cpu" else contextlib.nullcontext()
    )
    with scoped_cache(), gpu_cache_ctx:
        for spec in specs:
            fn = spec.implementation
            if fn is None:
                exc = RuntimeError(f"Feature is registered but has no implementation: {spec.name}")
                if errors == "raise":
                    raise exc
                if errors == "warn":
                    warnings.warn(str(exc), RuntimeWarning)
                continue
            try:
                validate_params(spec, params.get(spec.name, {}))
                if isinstance(X, pd.DataFrame) and spec.modality == "tabular":
                    from .execution import _resolve_columns_for_spec
                    cols = _resolve_columns_for_spec(spec, list(X.columns), lambda c: str(X[c].dtype), None)
                    args = [X[c] for c in cols]
                else:
                    args = [X]
                if spec.modality == "tabular" and spec.input_types == ("numeric",):
                    args = [numeric_series(arg) for arg in args]
                if backend == "cpu":
                    kwargs = dict(spec.default_params)
                    kwargs.update(params.get(spec.name, {}))
                    result = fn(*args, **kwargs)
                    if spec.modality == 'tabular' and isinstance(result, tuple) and len(result) == 2:
                        result = result[0]
                else:
                    # Step 3: GPU dispatch. _selected_specs already validated
                    # modality/existence; dispatch_one_feature (imported
                    # lazily, same lazy-import discipline as `execution`
                    # above) decides CPU vs GPU per backend/gpu_fallback
                    # policy and reports exactly which one ran.
                    from .gpu.dispatch import dispatch_one_feature, BackendInfo
                    if spec.modality == "tabular":
                        from .gpu.exceptions import GPUImplementationMissingError
                        if backend == "gpu" and gpu_fallback == "strict":
                            raise GPUImplementationMissingError(f"{spec.name} has no GPU implementation")
                        kwargs = {**spec.default_params, **params.get(spec.name, {})}
                        result = fn(*args, **kwargs)
                        if isinstance(result, tuple) and len(result) == 2:
                            result = result[0]
                        info = BackendInfo(requested_backend=backend, actual_backend="cpu",
                            cpu_fallback=backend == "gpu", fallback_reason="feature_not_gpu_capable")
                    else:
                        result, info = dispatch_one_feature(
                            spec, X, backend=backend, gpu_fallback=gpu_fallback, device=device,
                            gpu_memory_budget=gpu_memory_budget, params=params.get(spec.name, {}),
                        )
                    backend_report[spec.name] = info.as_dict()
                expanded = _expand_result(spec.name, result)
                for col, values in expanded.items():
                    values = np.asarray(values)
                    if n_rows is None:
                        n_rows = len(values)
                    if len(values) != n_rows:
                        raise ValueError(
                            f"{spec.name} produced {len(values)} rows; previous selected features produced {n_rows}"
                        )
                    outputs[col] = values
            except Exception as exc:
                from ._dependency_errors import annotate_missing_dependency
                annotate_missing_dependency(spec, exc)
                failures[spec.name] = f"{type(exc).__name__}: {exc}"
                if errors == "raise":
                    raise
                if errors == "warn":
                    warnings.warn(f"Skipping {spec.name}: {type(exc).__name__}: {exc}", RuntimeWarning)
    result_df = pd.DataFrame(outputs)
    if failures:
        result_df.attrs["execution_report"] = {"requested_features": [s.name for s in specs],
            "failed_features": failures, "errors_policy": errors}
    if backend != "cpu":
        # Step 3 spec §12: "do not silently fall back and leave users
        # thinking GPU executed." pandas.DataFrame.attrs is the officially
        # supported mechanism for attaching metadata to a DataFrame without
        # changing its return type/shape/API -- existing code that ignores
        # .attrs (the overwhelming majority) is completely unaffected.
        result_df.attrs["backend_report"] = backend_report
    return result_df


class CreatedFeature:
    def __init__(self, spec: FeatureSpec, params: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.params = {**spec.default_params, **(params or {})}
        self.name = spec.name

    def __call__(self, X, **kwargs):
        call_params = {**self.params, **kwargs}
        validate_params(self.spec, call_params)
        if self.spec.implementation is None:
            raise RuntimeError(f"Feature has no implementation: {self.spec.name}")
        try:
            result = self.spec.implementation(X, **call_params)
        except ModuleNotFoundError as exc:
            from ._dependency_errors import annotate_missing_dependency
            annotate_missing_dependency(self.spec, exc)
            raise
        if self.spec.modality == 'tabular' and isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result

    def __repr__(self) -> str:
        return f"CreatedFeature(name={self.name!r}, params={self.params!r})"


def create_feature(name: str, **params: Any) -> CreatedFeature:
    """Instantiate one named feature, mirroring ``timm.create_model`` ergonomics."""
    spec = REGISTRY.get(name)
    validate_params(spec, params)
    return CreatedFeature(spec, params)


def list_features(
    pattern: str | None = None,
    *,
    modality: str | None = None,
    family: str | None = None,
) -> list[str]:
    return [x.name for x in REGISTRY.search(pattern, modality=modality, family=family)]


def list_modalities() -> list[str]:
    return REGISTRY.modalities()


def list_families(*, modality: str | None = None) -> list[str]:
    return REGISTRY.families(modality=modality)


def search(query: str) -> list[FeatureSpec]:
    return REGISTRY.search(query)


def feature_info(name: str) -> FeatureSpec:
    return REGISTRY.get(name)
