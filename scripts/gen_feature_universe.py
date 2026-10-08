"""Regenerate docs/FEATURE_UNIVERSE.md from the live registry. Not a test; a doc-gen utility."""
from collections import Counter

import featm

specs = featm.REGISTRY.all()
by_modality: dict[str, list] = {}
for s in specs:
    by_modality.setdefault(s.modality, []).append(s)

counts = Counter(s.modality for s in specs)
total = len(specs)
modalities = sorted(counts)

lines = []
lines.append("# featm feature universe")
lines.append("")
lines.append(
    "This document is generated from the executable registry. Every entry below has "
    "an implementation callable. Some entries require an optional scientific backend, "
    "pretrained model, or user-supplied model at execution time."
)
lines.append("")
lines.append(f"**Total implemented entries:** {total}  ")
lines.append(f"**Modalities:** {len(modalities)}")
lines.append("")
lines.append("## Counts by modality")
lines.append("")
lines.append("| Modality | Features |")
lines.append("| --- | ---: |")
for m in sorted(counts, key=lambda m: -counts[m]):
    lines.append(f"| {m} | {counts[m]} |")
lines.append("")
lines.append("## Complete registry")
lines.append("")
for m in modalities:
    lines.append(f"### {m}")
    lines.append("")
    for s in sorted(by_modality[m], key=lambda s: s.name):
        bits = []
        if s.requires:
            bits.append("optional: " + ", ".join(s.requires))
        if s.output_kind != "scalar":
            bits.append(f"output={s.output_kind}")
        if s.cost.value != "low":
            bits.append(f"cost={s.cost.value}")
        if s.stateful:
            bits.append("stateful")
        if s.approximate_available:
            bits.append("approximate-available")
        suffix = f" ({'; '.join(bits)})" if bits else ""
        lines.append(f"- ✅ `{s.name}` — implemented{suffix}. {s.description or s.name}")
    lines.append("")

with open("docs/FEATURE_UNIVERSE.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", len(lines), "lines")
