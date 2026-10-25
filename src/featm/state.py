"""Stateful / two-pass feature protocol.

Features that need fitted state (category frequency encoding, z-score /
min-max scaling, corpus statistics for text features, and similar mergeable
global statistics) must work out-of-core and must not leak test-set
information into the fitted state. This module defines one general
lifecycle:

    state = encoder.init_state(params=fit_params)
    for batch in fit_batches:
        state = encoder.update_state(state, batch, params=fit_params)
    frozen = encoder.finalize_state(state, params=fit_params)
    for batch in transform_batches:
        output = encoder.transform_batch(batch, frozen, params=transform_params)

``update_state`` must be a genuine *merge* of the incoming batch into the
running accumulator (not a recompute-from-scratch), so a fit pass can be
driven by the out-of-core engine one batch at a time without ever
materializing the full training data. Z-score uses Welford/Chan online
mean-variance accumulation; min-max, frequency, and the text corpus-stats
encoders use exact, trivially mergeable accumulators (running min/max,
summed counts, document frequencies).

``params`` (fourth audit pass) is the merged per-call ``params={feature:
{...}}`` dict for this feature, threaded through every stage so an encoder
can read call-time configuration -- e.g. BM25's ``query`` (transform-time
only; corpus statistics are fit-time) or TF-IDF's ``max_features``
(fit-time, decides which terms the frozen vocabulary keeps). It defaults to
``None`` (treated as ``{}``) so encoders that need no configuration (Z-score,
min-max, frequency) can ignore the parameter entirely.

Each concrete encoder is also exposed as a normal registered ``tabular`` (or,
for the text corpus-stats encoders, ``text``) feature (see the bottom of this
module) via a thin adapter that runs ``init -> update(whole input) ->
finalize -> transform`` in one shot, so the identical encoder logic works for
both the simple small-data ``extract()`` path and genuine multi-batch
out-of-core fitting driven by the execution engine.
"""
from __future__ import annotations

import hashlib
import math
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import category_key, numeric_series, normalize_text, word_tokens, tfidf_names

import numpy as np
import pandas as pd

from .registry import Cost, FeatureOutput, register_feature
from .resources import ESTIMATED_BYTES_PER_CATEGORY


class StateBudgetExceededError(RuntimeError):
    """Raised when a stateful feature's exact fitted state cannot be bounded and cannot be spilled to disk."""


def _bounded_repr(s: str, *, max_len: int = 80) -> str:
    """A length-bounded, still-identifying representation of a string for an error/warning message.

    Fifth audit pass, Finding 8: the pre-fix error messages interpolated the
    complete ``repr()`` of the rejected category/term -- for a single 100KB
    or multi-MB string value, that turns one budget-exceeded error (whose
    entire point is reporting a resource-safety problem) into a 100KB/
    multi-MB error/warning string itself, amplifying exactly the kind of
    memory/log pressure it exists to report. Retains a short prefix (enough
    to recognize a short/medium value at a glance) plus the true length and
    a short stable digest (enough to tell two different long values apart,
    or confirm the same one recurring), without ever reproducing the value
    in full.
    """
    if len(s) <= max_len:
        return repr(s)
    digest = hashlib.blake2b(s.encode("utf-8", errors="surrogatepass"), digest_size=6).hexdigest()
    return f"{s[:max_len]!r}...<truncated, len={len(s):,}, blake2b={digest}>"


def _real_item_bytes(value: Any, *, overhead: int = 96) -> int:
    """Best-effort *actual* retained-memory estimate for one Python object, not a fixed average.

    ``sys.getsizeof`` reports the object's own allocation (accurate for a
    ``str``/``int``/``float``, which is what every state/reservoir accumulator
    in this module retains) plus a calibrated constant for the container
    overhead of holding it (a dict entry + a Python object pointer, or a list
    slot) -- fourth audit pass, Finding 11: a fixed per-item constant (the
    pre-fix ``200`` bytes/category) silently undercounts a long string by
    orders of magnitude, letting retained state exceed its declared budget
    with no warning. This is still an estimate (``sys.getsizeof`` does not
    follow references, irrelevant here since these accumulators only ever
    retain flat str/int/float values, never nested containers), not a
    guarantee against every possible adversarial object -- but it scales with
    the *actual* value, unlike a constant.
    """
    try:
        return int(sys.getsizeof(value)) + overhead
    except TypeError:
        return ESTIMATED_BYTES_PER_CATEGORY


class StatefulEncoder(ABC):
    """General mergeable fit/transform lifecycle for one stateful feature."""

    @abstractmethod
    def init_state(
        self, *, state_budget_bytes: int | None = None, spill_dir: "str | Path | None" = None, params: Mapping[str, Any] | None = None
    ) -> Any:
        """A fresh, empty accumulator.

        ``state_budget_bytes``/``spill_dir`` are optional resource-planning
        hints: most encoders (z-score, min-max) hold O(1) state and ignore
        them; :class:`FrequencyEncoder`/the text corpus-stats encoders use
        them to bound their exact in-memory state (see their docstrings).
        Both default to ``None`` (unbounded, in-memory only) so the
        small-data single-shot adapter below -- which calls ``init_state()``
        with no arguments -- keeps its existing zero-configuration behavior.
        ``params`` is this feature's merged call-time ``params={...}`` dict
        (fit-time configuration, e.g. TF-IDF's ``max_features``); most
        encoders ignore it.
        """

    @abstractmethod
    def update_state(self, state: Any, batch: pd.Series, *, params: Mapping[str, Any] | None = None) -> Any:
        """Fold one source-ordered batch into ``state``.

        Single-machine execution calls this sequentially. Distributed execution
        computes one partial state per partition and combines those partials via
        :meth:`merge_states`. Implementations should therefore keep the raw
        accumulator representation mergeable and avoid embedding process-local
        resources in it.
        """

    def merge_states(self, left: Any, right: Any, *, params: Mapping[str, Any] | None = None) -> Any:
        """Merge two raw accumulator states in source partition order.

        Step 4 contract: ``left`` represents rows that precede ``right`` in the
        logical source. Most accumulators are mathematically commutative, but
        order-sensitive defaults such as ``text.semantics.similarity``'s
        "first training document" deliberately preserve the left-most state.
        Stateful encoders that do not implement this method are not eligible
        for distributed fitting and are rejected by the distributed planner.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement distributed state merging")

    @property
    def distributed_merge_supported(self) -> bool:
        return type(self).merge_states is not StatefulEncoder.merge_states

    def finalize_state(self, state: Any, *, params: Mapping[str, Any] | None = None) -> Any:
        """Freeze the accumulator into the state used for transform. Default: identity."""
        return state

    @abstractmethod
    def transform_batch(self, batch: pd.Series, state: Any, *, params: Mapping[str, Any] | None = None) -> pd.Series:
        """Apply frozen ``state`` to one batch. Never mutates state. ``params`` is transform-time config (e.g. BM25's ``query``)."""

    def fit_transform_batch(self, batch: pd.Series, *, params: Mapping[str, Any] | None = None) -> tuple[pd.Series, Any]:
        """Single-shot fit+transform, for the small-data / single-array path."""
        state = self.finalize_state(self.update_state(self.init_state(params=params), batch, params=params), params=params)
        return self.transform_batch(batch, state, params=params), state


# --------------------------------------------------------------------------
# Z-score: Welford/Chan online mean-variance, numerically stable and mergeable.
# --------------------------------------------------------------------------
@dataclass
class _WelfordAccumulator:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations from the running mean


def _combine_welford(a: _WelfordAccumulator, b: _WelfordAccumulator) -> _WelfordAccumulator:
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    n = a.n + b.n
    delta = b.mean - a.mean
    mean = a.mean + delta * (b.n / n)
    m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n)
    return _WelfordAccumulator(n=n, mean=mean, m2=m2)


class ZScoreEncoder(StatefulEncoder):
    """Standardize to zero mean / unit variance, fit via mergeable Welford accumulation."""

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> _WelfordAccumulator:
        return _WelfordAccumulator()  # O(1) state regardless of cardinality; hints are irrelevant

    def update_state(self, state: _WelfordAccumulator, batch: pd.Series, *, params=None) -> _WelfordAccumulator:
        x = numeric_series(batch, finite_fit=True).dropna().to_numpy()
        if x.size == 0:
            return state
        batch_mean = float(x.mean())
        batch_m2 = float(np.sum((x - batch_mean) ** 2))
        return _combine_welford(state, _WelfordAccumulator(n=x.size, mean=batch_mean, m2=batch_m2))

    def merge_states(self, left: _WelfordAccumulator, right: _WelfordAccumulator, *, params=None) -> _WelfordAccumulator:
        return _combine_welford(left, right)

    def finalize_state(self, state: _WelfordAccumulator, *, params=None) -> dict:
        if state.n == 0:
            raise ValueError("Scaler training data contains no finite observations")
        variance = state.m2 / state.n
        std = math.sqrt(variance) if variance == variance and variance >= 0 else float("nan")
        if not math.isfinite(state.mean) or not math.isfinite(std):
            raise ValueError("Nonfinite scaler moments; rescale the training values")
        return {"mean": state.mean, "std": std, "n": state.n}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        x = numeric_series(batch)
        sd = state["std"]
        if not math.isfinite(sd) or not math.isfinite(state["mean"]):
            raise ValueError("Invalid nonfinite fitted scaler state")
        denom = sd if sd > 0 else 1.0
        return (x - state["mean"]) / denom


# --------------------------------------------------------------------------
# Missingness entropy: constant-size, exactly mergeable whole-column counts.
# --------------------------------------------------------------------------
class MissingnessEntropyEncoder(StatefulEncoder):
    """Fit binary Shannon entropy of missingness over the entire source column.

    Counts, unlike per-partition entropies, can be merged exactly; the frozen
    scalar is then broadcast to every row during transform. This prevents
    Source and distributed partition boundaries changing the answer.
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> dict:
        return {"n": 0, "missing": 0}

    def update_state(self, state: dict, batch: pd.Series, *, params=None) -> dict:
        values = pd.Series(batch)
        return {"n": state["n"] + len(values), "missing": state["missing"] + int(values.isna().sum())}

    def merge_states(self, left: dict, right: dict, *, params=None) -> dict:
        return {"n": left["n"] + right["n"], "missing": left["missing"] + right["missing"]}

    def finalize_state(self, state: dict, *, params=None) -> dict:
        n, missing = state["n"], state["missing"]
        p = missing / n if n else float("nan")
        entropy = 0.0 if p in (0.0, 1.0) else -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return {"entropy": float(entropy), "n": n, "missing": missing}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        values = pd.Series(batch)
        return pd.Series(state["entropy"], index=values.index, dtype=float)


# --------------------------------------------------------------------------
# Min-max: trivially mergeable running extrema.
# --------------------------------------------------------------------------
class MinMaxEncoder(StatefulEncoder):
    """Scale to [0, 1] using the training-set min/max."""

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> dict:
        return {"min": math.inf, "max": -math.inf}  # O(1) state; hints are irrelevant

    def update_state(self, state: dict, batch: pd.Series, *, params=None) -> dict:
        x = numeric_series(batch, finite_fit=True).dropna()
        if x.empty:
            return state
        return {"min": min(state["min"], float(x.min())), "max": max(state["max"], float(x.max()))}

    def merge_states(self, left: dict, right: dict, *, params=None) -> dict:
        return {"min": min(left["min"], right["min"]), "max": max(left["max"], right["max"])}

    def finalize_state(self, state: dict, *, params=None) -> dict:
        if not all(math.isfinite(state[k]) for k in ("min", "max")):
            raise ValueError("Scaler training data contains no finite observations")
        if not math.isfinite(state["max"] - state["min"]):
            raise ValueError("Nonfinite scaler range; rescale the training values")
        return dict(state)

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        x = numeric_series(batch)
        lo, hi = state["min"], state["max"]
        span = hi - lo
        if not (math.isfinite(lo) and math.isfinite(hi)) or span < 0:
            raise ValueError("Invalid nonfinite fitted scaler state")
        return (x - lo) / (span if span > 0 else 1.0)


# --------------------------------------------------------------------------
# Category frequency: exact, mergeable via summed counts. Unseen -> 0.
#
# A plain in-memory Counter (the pre-fix implementation) has memory that
# scales O(distinct categories), completely unbounded by memory_budget: a
# high-cardinality fit column (unique IDs, free-text-like categoricals, ...)
# can make the exact fitted state itself many times larger than a small
# declared budget with no warning. BoundedFrequencyCounter stays a plain
# in-memory dict, tracking its own REAL estimated size per key (not a fixed
# per-item constant -- Finding 11, fourth audit pass), and raises
# StateBudgetExceededError *before* growing past a budget-derived size
# rather than growing unbounded with no warning -- see its docstring for why
# this pass does not attempt a disk-spilled exact backend.
# --------------------------------------------------------------------------
class BoundedFrequencyCounter:
    """Exact (category -> count) accumulator that refuses to grow past an estimated memory budget.

    Tracks a running total of *actual* per-key byte estimates
    (:func:`_real_item_bytes`, based on ``sys.getsizeof`` of the real key
    value, not a fixed average) rather than ``len(keys) * constant`` --
    fourth audit pass, Finding 11: a handful of long strings (e.g. a few
    hundred distinct 10KB values) can exceed a small budget even though
    ``len(keys) * 200`` would report them as comfortably within it. New keys
    are admitted one at a time (not via an unbounded local ``Counter`` built
    first and checked afterward), so the budget check happens *before* a
    batch's full temporary aggregation could itself balloon past the limit.

    A disk-spilled exact backend (persisting overflow categories to e.g.
    SQLite) was considered for this class, since it would let exact
    high-cardinality frequency counting stay usable beyond what fits in
    memory. It was deliberately *not* implemented in this pass: making a
    disk-spilled accumulator correctly resumable is not just "write to a
    file instead of a dict" -- the existing fit-phase checkpoint protocol
    (see ``execution.py``'s versioned accumulator-snapshot commit) relies on
    the *entire* accumulator being nothing but a plain Python object that is
    only ever persisted via one atomic, versioned pickle snapshot per batch.
    A spill target written to *outside* that snapshot boundary (e.g. an
    immediately-committed SQLite upsert per batch) would be a durable
    side-channel not covered by that snapshot/rollback protocol: resuming
    from an older, still-valid snapshot after a crash would re-process the
    same batch's rows and re-apply them to the spill store, double-counting
    that batch's contribution -- reintroducing the exact class of bug
    Finding 9 (this project's third audit pass) fixed for every other
    accumulator type. A correct version of this (e.g. per-batch-tagged,
    idempotent upserts keyed by ``(batch_index, category)``, so a redo
    overwrites rather than double-adds) is a reasonable follow-up but is
    scoped out of this pass; see ``STEP2_AUDIT3_FIX_HANDOFF.md``. This class
    instead fails safely and honestly: it raises before an unbounded exact
    dict can exceed the declared budget, rather than silently exceeding it
    (the actual resource-safety bug this class exists to close) or silently
    switching to an approximate result under ``mode="exact"`` (which must
    never happen regardless of memory pressure).
    """

    def __init__(self, *, budget_bytes: int | None) -> None:
        self.budget_bytes = budget_bytes
        self._counts: dict[str, int] = {}
        self._bytes_used = 0
        self.total = 0

    def update(self, keys: "Sequence[str]") -> None:
        """Merge one batch's raw (not-yet-aggregated) key sequence into the accumulated exact state.

        Takes the raw per-row key sequence directly (not a pre-built
        ``Counter``), admitting and budget-checking one distinct new key at a
        time, so a batch containing many distinct long-string values cannot
        itself balloon past the budget before the check has a chance to run.
        """
        for k in keys:
            self.total += 1
            if k in self._counts:
                self._counts[k] += 1
                continue
            if self.budget_bytes is not None:
                projected = self._bytes_used + _real_item_bytes(k)
                if projected > self.budget_bytes:
                    raise StateBudgetExceededError(
                        f"Exact category-frequency fit state would grow to an estimated {projected:,} bytes "
                        f"admitting distinct category {_bounded_repr(k)} ({len(self._counts):,} distinct "
                        f"categories retained so far), exceeding its memory_budget-derived state budget ({self.budget_bytes:,} "
                        "bytes). Never silently approximating under mode='exact' -- increase memory_budget, "
                        "select mode='approximate'/'auto' to fit from a bounded sample instead, or pre-bucket "
                        "this high-cardinality column upstream."
                    )
                self._bytes_used = projected
            self._counts[k] = 1

    def merge(self, other: "BoundedFrequencyCounter") -> "BoundedFrequencyCounter":
        """Merge another exact counter into this one, enforcing this counter's budget."""
        for k, count in other._counts.items():
            if k not in self._counts:
                if self.budget_bytes is not None:
                    projected = self._bytes_used + _real_item_bytes(k)
                    if projected > self.budget_bytes:
                        raise StateBudgetExceededError(
                            f"Distributed exact category-frequency merge would grow to an estimated {projected:,} bytes "
                            f"admitting distinct category {_bounded_repr(k)}, exceeding its memory_budget-derived "
                            f"state budget ({self.budget_bytes:,} bytes)."
                        )
                    self._bytes_used = projected
                self._counts[k] = 0
            self._counts[k] += int(count)
        self.total += int(other.total)
        return self

    def get_many(self, keys: "Sequence[str]") -> dict[str, int]:
        return {k: self._counts.get(k, 0) for k in keys}


class FrequencyEncoder(StatefulEncoder):
    """Training-set category frequency; categories never seen during fit map to 0.

    Exact under every ``mode`` -- never silently approximates. If
    ``state_budget_bytes`` is given (the out-of-core engine derives one from
    ``memory_budget``), a fit column with more distinct categories than that
    real-byte-estimated budget can hold raises :class:`StateBudgetExceededError`
    (isolated per ``errors=`` like any other feature-level failure) instead of
    growing an unbounded Python ``dict`` with no warning -- see
    :class:`BoundedFrequencyCounter`.
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> BoundedFrequencyCounter:
        return BoundedFrequencyCounter(budget_bytes=state_budget_bytes)

    def update_state(self, state: BoundedFrequencyCounter, batch: pd.Series, *, params=None) -> BoundedFrequencyCounter:
        key = pd.Series(batch).map(category_key)
        state.update(key.tolist())
        return state

    def merge_states(self, left: BoundedFrequencyCounter, right: BoundedFrequencyCounter, *, params=None) -> BoundedFrequencyCounter:
        return left.merge(right)

    def finalize_state(self, state: BoundedFrequencyCounter, *, params=None) -> dict:
        return {"counts": state, "total": state.total}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        key = pd.Series(batch).map(category_key)
        total = state["total"] or 1
        counts: BoundedFrequencyCounter | dict = state["counts"]
        if isinstance(counts, BoundedFrequencyCounter):
            looked_up = counts.get_many(key.tolist())
            mapped = key.map(looked_up).fillna(0.0)
        else:  # legacy plain-dict state (e.g. an older pickled checkpoint) -- still supported
            mapped = key.map(counts).fillna(0.0)
        return (mapped / total).astype(float)


# --------------------------------------------------------------------------
# Text corpus statistics: mergeable (document-frequency, global term counts,
# document count, total token count) accumulator shared by the
# surprisal/BM25/TF-IDF/similarity stateful encoders below (fourth audit
# pass, Finding 1). All four features' pre-fix implementations recomputed
# these corpus-wide statistics independently inside every out-of-core batch,
# producing a genuinely different (and, for TF-IDF, differently-*shaped*)
# result per batch -- a silent mathematical-correctness bug, not merely an
# approximation. This accumulator is bounded the same way
# BoundedFrequencyCounter is (real per-term byte estimates, checked before
# admitting a new term), for the same reason: an unbounded vocabulary is a
# real resource-safety risk for a large/high-cardinality text corpus.
# --------------------------------------------------------------------------
class _CorpusTermStats:
    """Mergeable (term -> document frequency, term -> total count, N docs, total tokens)."""

    def __init__(self, *, budget_bytes: int | None) -> None:
        self.budget_bytes = budget_bytes
        self.doc_freq: dict[str, int] = {}
        self.term_total: dict[str, int] = {}
        self._bytes_used = 0
        self.n_docs = 0
        self.total_tokens = 0

    def _admit_term(self, term: str) -> None:
        if term in self.doc_freq:
            return
        if self.budget_bytes is not None:
            projected = self._bytes_used + _real_item_bytes(term) * 2  # doc_freq + term_total entries
            if projected > self.budget_bytes:
                raise StateBudgetExceededError(
                    f"Text corpus-statistics fit state would grow to an estimated {projected:,} bytes "
                    f"admitting term {_bounded_repr(term)} ({len(self.doc_freq):,} distinct terms retained so far), "
                    f"exceeding its memory_budget-derived state budget ({self.budget_bytes:,} bytes). "
                    "Increase memory_budget, or pre-filter/tokenize this corpus more coarsely upstream."
                )
            self._bytes_used = projected
        self.doc_freq[term] = 0
        self.term_total[term] = 0

    def update(self, docs_tokens: list[list[str]]) -> None:
        for tokens in docs_tokens:
            self.n_docs += 1
            self.total_tokens += len(tokens)
            seen_in_doc: set[str] = set()
            for t in tokens:
                self._admit_term(t)
                self.term_total[t] += 1
                if t not in seen_in_doc:
                    self.doc_freq[t] += 1
                    seen_in_doc.add(t)

    def merge(self, other: "_CorpusTermStats") -> "_CorpusTermStats":
        for term in other.doc_freq:
            self._admit_term(term)
            self.doc_freq[term] += int(other.doc_freq[term])
            self.term_total[term] += int(other.term_total[term])
        self.n_docs += int(other.n_docs)
        self.total_tokens += int(other.total_tokens)
        return self

    @property
    def avg_doc_length(self) -> float:
        return (self.total_tokens / self.n_docs) if self.n_docs else 0.0


def _word_tokens(s):
    return word_tokens(s)


class SurprisalEncoder(StatefulEncoder):
    """Mean negative log-probability of a document's words under the *training* corpus's global word distribution.

    Additive-smoothed unigram surprisal, fit via mergeable global term
    counts across as many out-of-core fit batches as needed -- the exact
    mathematics ``text.information.surprisal`` always intended, now computed
    once over the whole training corpus rather than independently per batch.
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> _CorpusTermStats:
        return _CorpusTermStats(budget_bytes=state_budget_bytes)

    def update_state(self, state: _CorpusTermStats, batch: pd.Series, *, params=None) -> _CorpusTermStats:
        state.update([_word_tokens(s) for s in batch])
        return state

    def merge_states(self, left: _CorpusTermStats, right: _CorpusTermStats, *, params=None) -> _CorpusTermStats:
        return left.merge(right)

    def finalize_state(self, state: _CorpusTermStats, *, params=None) -> dict:
        alpha = float((params or {}).get("alpha", 1.0))
        return {"term_total": dict(state.term_total), "total_tokens": state.total_tokens, "V": max(len(state.term_total), 1), "alpha": alpha}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        alpha = state["alpha"]
        V = state["V"]
        total = state["total_tokens"]
        term_total = state["term_total"]
        out = []
        for s in batch:
            w = _word_tokens(s)
            if not w:
                out.append(0.0)
                continue
            out.append(float(np.mean([-math.log((term_total.get(t, 0) + alpha) / (total + alpha * V)) for t in w])))
        return pd.Series(out, index=batch.index)


class Bm25Encoder(StatefulEncoder):
    """BM25 relevance score against a training-corpus-fitted document frequency / average length.

    ``query`` is transform-time configuration (different test queries reuse
    the same fitted corpus statistics); document frequency, corpus size, and
    average document length are training-corpus statistics, fit via
    mergeable accumulation across fit batches.
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> _CorpusTermStats:
        return _CorpusTermStats(budget_bytes=state_budget_bytes)

    def update_state(self, state: _CorpusTermStats, batch: pd.Series, *, params=None) -> _CorpusTermStats:
        state.update([_word_tokens(s) for s in batch])
        return state

    def merge_states(self, left: _CorpusTermStats, right: _CorpusTermStats, *, params=None) -> _CorpusTermStats:
        return left.merge(right)

    def finalize_state(self, state: _CorpusTermStats, *, params=None) -> dict:
        k1 = float((params or {}).get("k1", 1.5))
        b = float((params or {}).get("b", 0.75))
        return {"doc_freq": dict(state.doc_freq), "N": state.n_docs, "avg_doc_length": state.avg_doc_length, "k1": k1, "b": b}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        query = (params or {}).get("query")
        if query is None:
            raise ValueError('text.bm25.bm25 requires query="..." (transform-time parameter)')
        q = _word_tokens(str(query))
        df = state["doc_freq"]
        N = state["N"]
        avg = state["avg_doc_length"] or 1.0
        k1, b = state["k1"], state["b"]
        out = []
        for s in batch:
            tokens = _word_tokens(s)
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in q:
                d = df.get(t, 0)
                idf = np.log(1 + (N - d + 0.5) / (d + 0.5))
                f = tf[t]
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg) + 1e-15)
            out.append(float(score))
        return pd.Series(out, index=batch.index)


def _smooth_idf(doc_freq: int, n_docs: int) -> float:
    # scikit-learn's default TfidfVectorizer(smooth_idf=True) formula.
    return float(np.log((1 + n_docs) / (1 + doc_freq)) + 1)


def _fit_tfidf_vocabulary(state: _CorpusTermStats, max_features: int | None) -> tuple[list[str], dict[str, float]]:
    terms = list(state.term_total)
    if max_features is not None and len(terms) > max_features:
        terms = sorted(terms, key=lambda t: (-state.term_total[t], t))[:max_features]
    vocab = sorted(terms)
    idf = {t: _smooth_idf(state.doc_freq[t], state.n_docs) for t in vocab}
    return vocab, idf


def _tfidf_vectorize(tokens: list[str], vocab: list[str], idf: dict[str, float]) -> np.ndarray:
    tf = Counter(tokens)
    vec = np.array([tf.get(t, 0) * idf[t] for t in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class TfidfEncoder(StatefulEncoder):
    """TF-IDF vector, fit via a mergeable term/document-frequency accumulator, frozen vocabulary at transform.

    ``max_features`` (fit-time) selects the frozen vocabulary (by total
    training-corpus term frequency, matching scikit-learn's
    ``TfidfVectorizer(max_features=...)`` selection rule) once, at the end of
    the fit phase -- unlike the pre-fix per-batch ``TfidfVectorizer().fit_transform()``,
    every transform batch (train or test) is scored against that one frozen
    vocabulary/IDF, so output width and column identity never drift between
    batches or between train and test.
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> _CorpusTermStats:
        return _CorpusTermStats(budget_bytes=state_budget_bytes)

    def update_state(self, state: _CorpusTermStats, batch: pd.Series, *, params=None) -> _CorpusTermStats:
        state.update([_word_tokens(s) for s in batch])
        return state

    def merge_states(self, left: _CorpusTermStats, right: _CorpusTermStats, *, params=None) -> _CorpusTermStats:
        return left.merge(right)

    def finalize_state(self, state: _CorpusTermStats, *, params=None) -> dict:
        max_features = (params or {}).get("max_features", 128)
        if max_features is not None and (isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1):
            raise ValueError("max_features must be a positive integer or None")
        vocab, idf = _fit_tfidf_vocabulary(state, max_features)
        return {"vocab": vocab, "idf": idf}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> FeatureOutput:
        import re

        vocab, idf = state["vocab"], state["idf"]
        if not vocab:
            # An empty training vocabulary (e.g. an all-empty-string fit corpus) is a
            # degenerate but valid fit result -- emit one all-zero column rather than
            # a variable/empty width, so the output schema is still stable.
            return FeatureOutput(np.zeros((len(batch), 1)), ("_empty_vocab",))
        rows = np.asarray([_tfidf_vectorize(_word_tokens(s), vocab, idf) for s in batch]).reshape(len(batch), len(vocab))
        cols = tfidf_names(vocab)
        return FeatureOutput(rows, cols)


class SimilarityEncoder(StatefulEncoder):
    """Cosine similarity to a reference document, against a training-corpus-fitted TF-IDF vocabulary.

    ``reference`` (transform-time) is vectorized against the *frozen*
    training vocabulary/IDF; if omitted, the default reference is the
    *first training document seen during fit* (frozen at fit time, not
    "whichever document happens to be first in the current call/batch" --
    the pre-fix per-batch-local-first-document bug this exists to fix).
    """

    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None) -> dict:
        return {"stats": _CorpusTermStats(budget_bytes=state_budget_bytes), "first_doc": None}

    def update_state(self, state: dict, batch: pd.Series, *, params=None) -> dict:
        if state["first_doc"] is None and len(batch):
            state["first_doc"] = normalize_text(batch.iloc[0])
        state["stats"].update([_word_tokens(s) for s in batch])
        return state

    def merge_states(self, left: dict, right: dict, *, params=None) -> dict:
        # Distributed coordinator merges partitions in logical source order.
        # Preserve the first document from the left-most non-empty partition.
        left["stats"].merge(right["stats"])
        if left.get("first_doc") is None:
            left["first_doc"] = right.get("first_doc")
        return left

    def finalize_state(self, state: dict, *, params=None) -> dict:
        vocab, idf = _fit_tfidf_vocabulary(state["stats"], max_features=None)
        default_ref_vec = _tfidf_vectorize(_word_tokens(state["first_doc"] or ""), vocab, idf)
        return {"vocab": vocab, "idf": idf, "default_reference_vec": default_ref_vec}

    def transform_batch(self, batch: pd.Series, state: dict, *, params=None) -> pd.Series:
        vocab, idf = state["vocab"], state["idf"]
        reference = (params or {}).get("reference")
        ref_vec = _tfidf_vectorize(_word_tokens(reference), vocab, idf) if reference is not None else state["default_reference_vec"]
        out = [float(np.dot(_tfidf_vectorize(_word_tokens(s), vocab, idf), ref_vec)) for s in batch]
        return pd.Series(out, index=batch.index)


def _stateful_adapter(encoder: StatefulEncoder):
    """Wrap a StatefulEncoder as a plain tabular fit/state feature callable.

    Matches the existing tabular calling convention (``fn(s, *, fit, state) ->
    (values, state)``) so the same encoder is directly usable via
    ``featm.extract`` / ``featm.create_feature`` / the legacy ``AutoFeatures``
    planner on small in-memory data, in addition to genuine multi-batch
    out-of-core fitting driven by the new execution engine (which calls
    ``update_state``/``finalize_state``/``transform_batch`` directly).
    """

    def fn(s, *, fit: bool = False, state: Any = None, **params: Any):
        s = pd.Series(s)
        if fit or state is None:
            raw = encoder.update_state(encoder.init_state(params=params), s, params=params)
            state = encoder.finalize_state(raw, params=params)
        return encoder.transform_batch(s, state, params=params), state

    fn.__featm_parameters__ = ()
    return fn


_APPROX_NOTE = (
    "Approximate mode fits state by retaining a bounded reservoir sample of the fit rows "
    "(Algorithm R) in memory; every fit row is still read/streamed exactly once (Algorithm R "
    "must see each row to keep every row's inclusion probability correct), it is just not "
    "all retained -- this bounds retained memory, not input I/O. Every row seen has equal "
    "probability of being in the final sample, and the sample size (requested and effective, "
    "if clamped by memory_budget) is reported in the execution report."
)

register_feature(
    name="tabular.scaling.zscore",
    family="scaling",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.LOW,
    streamable=True,
    incremental=True,
    stateful=True,
    approximate_available=True,
    approximate_max_exact_rows=5_000_000,
    approximate_notes=_APPROX_NOTE,
    description="Z-score standardization fit via mergeable Welford accumulation across batches.",
    tags=("scaling", "zscore", "stateful", "welford"),
)(_stateful_adapter(ZScoreEncoder()))

register_feature(
    name="tabular.scaling.minmax",
    family="scaling",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.LOW,
    streamable=True,
    incremental=True,
    stateful=True,
    approximate_available=True,
    approximate_max_exact_rows=5_000_000,
    approximate_notes=_APPROX_NOTE,
    description="Min-max scaling fit via mergeable running extrema across batches.",
    tags=("scaling", "minmax", "stateful"),
)(_stateful_adapter(MinMaxEncoder()))

register_feature(
    name="tabular.encoding.category_frequency",
    family="encoding",
    modality="tabular",
    input_types=("categorical",),
    cost=Cost.LOW,
    streamable=True,
    incremental=True,
    stateful=True,
    approximate_available=True,
    approximate_max_exact_rows=5_000_000,
    approximate_notes=_APPROX_NOTE,
    description="Training-set category frequency encoding; categories unseen during fit map to 0.",
    tags=("encoding", "frequency", "categorical", "stateful", "high-cardinality"),
)(_stateful_adapter(FrequencyEncoder()))


STATEFUL_ENCODERS: dict[str, StatefulEncoder] = {
    "tabular.scaling.zscore": ZScoreEncoder(),
    "tabular.scaling.minmax": MinMaxEncoder(),
    "tabular.encoding.category_frequency": FrequencyEncoder(),
    "tabular.missingness.entropy": MissingnessEntropyEncoder(),
}
