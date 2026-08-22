"""Leakage-risk and measurability audit for a dataset or a recording folder.

This is a NEGATIVE instrument. It can prove a leak, it can prove a small number
of properties from structure, and for everything else it reports what it looked
at and what it could not see. It never says "clean", and the vocabulary is built
so that no reader can extract that claim from the output: every run ends with a
non-suppressible list of what was not checked, and the summary line counts
unchecked areas separately from passing ones.

Two things the design turns on:

1. **Overlap is decided from index ranges, never from content similarity.** Two
   windows that share half their samples hold those samples at different
   positions once flattened, so a similarity score compares sample *k* of one
   against sample *k+stride* of the other and rates real overlap like noise.
   `scripts/audit_leakage.py` has that blind spot; nothing here inherits it.

2. **In a dataset iqforge built, cross-split overlap is settled by proof rather
   than measurement.** Two windows can only overlap if they came from the same
   recording, so recording disjointness makes cross-split overlap impossible.
   That is stronger than any sample of the data, and it costs no reads.

What this deliberately does not do is train anything. Whether a task sits at the
ceiling or in a measurable band cannot be settled without a model; the audit can
only rule out the trivial case, and it says so and points at `measure-leakage`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from iqforge.io import META_EXT, Recording, load
from iqforge.labeling import resolve_exclude_labels

#: Report width. 78 rather than 80 so the block survives being pasted into a
#: LaTeX verbatim environment or a diff with a two-column gutter.
WIDTH = 78

#: Column widths of the checks table, summing to WIDTH with two-space gaps.
_STATUS_W, _CHECK_W = 11, 25
_DETAIL_W = WIDTH - _STATUS_W - _CHECK_W - 4

#: Samples read per recording for the axis measurements. Bounded on purpose:
#: the audit is meant to be cheap enough to run on every build, and the axes it
#: measures are stationary enough that a prefix answers them.
SAMPLE_LIMIT = 2_000_000

#: Block length for the power envelope used to separate active from idle.
_BLOCK = 1024

#: An axis that classifies at least this fraction of recordings on its own is
#: reported as a risk. Set high deliberately: the interesting case is an axis
#: that very nearly *determines* the class, and a lower bar would flag ordinary
#: correlation on every dataset until the warning stopped being read.
RISK_SEPARABILITY = 0.90

#: Above this, the task is reported as sitting at the ceiling: a single scalar
#: feature all but names the class, so any reasonable model saturates.
CEILING_SEPARABILITY = 0.95


class Status(StrEnum):
    """Outcome of one check.

    There is no `CLEAN`. `PASS_PROOF` means the property was established from
    structure and holds for the whole dataset; `PASS_SAMPLE` means it held on
    what was read and says nothing about the rest.
    """

    LEAK = "LEAK"
    RISK = "RISK"
    PASS_PROOF = "PASS/proof"
    PASS_SAMPLE = "PASS/sample"
    NOT_CHECKED = "NOT CHECKED"


#: Which summary bucket each status counts toward. `NOT_CHECKED` gets its own
#: bucket and is never folded into `passed` -- an unexamined area reading as a
#: pass is the specific failure this tool exists to avoid.
_BUCKET = {
    Status.LEAK: "leaks",
    Status.RISK: "risk",
    Status.PASS_PROOF: "passed",
    Status.PASS_SAMPLE: "passed",
    Status.NOT_CHECKED: "not checked",
}


@dataclass(frozen=True)
class Finding:
    """One check's outcome.

    Attributes:
        status: What the check concluded.
        check: Short name, shown in the table's second column.
        detail: One sentence of evidence. For `NOT_CHECKED` this must say
            *why*, because "not checked" without a reason is indistinguishable
            from an oversight.
    """

    status: Status
    check: str
    detail: str


@dataclass
class AuditReport:
    """Everything one audit run produced."""

    tool_version: str
    generated: str
    mode: str
    input_path: str
    input_lines: list[str]
    fingerprint: str
    findings: list[Finding] = field(default_factory=list)
    did_not_check: list[str] = field(default_factory=list)
    verdict: str = ""
    next_step: list[str] = field(default_factory=list)
    #: Measured axes, kept off the rendered report. `measure-leakage` uses them
    #: to decide refuse categories that need the raw timestamps and durations,
    #: not only the finding text.
    features: list[RecordFeatures] = field(default_factory=list, repr=False, compare=False)

    @property
    def summary(self) -> dict[str, int]:
        """Counts per bucket, in the order they are printed."""
        counts = dict.fromkeys(("leaks", "passed", "risk", "not checked"), 0)
        for finding in self.findings:
            counts[_BUCKET[finding.status]] += 1
        return counts


# --------------------------------------------------------------------------
# Feature measurement
# --------------------------------------------------------------------------


@dataclass
class RecordFeatures:
    """Per-recording scalars the axis checks compare against the class label.

    Every field is either measured from the samples or read from metadata, and
    every one of them can be `None` when the input does not support it. A `None`
    propagates into `NOT CHECKED` rather than into a default value.
    """

    record_id: str
    label: str | None
    carrier_offset_hz: float | None = None
    mean_power_db: float | None = None
    duration_samples: int | None = None
    duty_cycle: float | None = None
    burst_start_frac: float | None = None
    capture_time: dt.datetime | None = None
    occupied_bw_hz: float | None = None
    sample_rate: float | None = None
    #: Where `occupied_bw_hz` came from: "annotation" or "spectrum". Reported,
    #: because a declared bandwidth and an estimated one deserve different trust.
    bw_source: str | None = None
    #: `core:collection`, the SigMF field naming the Collection this recording
    #: belongs to. A hint about grouping, never a proof -- see `_collection_finding`.
    collection: str | None = None

    def axes(self) -> dict[str, float | None]:
        """The numeric axes, by the name the report shows."""
        return {
            "axis: carrier offset": self.carrier_offset_hz,
            "axis: mean power": self.mean_power_db,
            "axis: recording length": (
                None if self.duration_samples is None else float(self.duration_samples)
            ),
            "axis: burst duty cycle": self.duty_cycle,
            "axis: burst position": self.burst_start_frac,
        }


def _active_mask(samples: np.ndarray) -> np.ndarray:
    """Per-block boolean mask of where the signal is on.

    A power envelope with the threshold placed a third of the way up the dynamic
    range. Crude, but the alternative -- assuming the signal is always on -- is
    what makes a bandwidth estimate meaningless on a bursty capture: measured on
    the DASH7 set, 6.8% of a recording carries signal and the rest is noise
    floor, so an ungated spectrum describes the receiver, not the transmitter.
    """
    count = samples.size // _BLOCK
    if count < 2:
        return np.zeros(0, dtype=bool)
    blocks = samples[: count * _BLOCK].reshape(count, _BLOCK)
    power = (np.abs(blocks) ** 2).mean(axis=1)
    db = 10.0 * np.log10(power + 1e-30)
    span = float(db.max() - db.min())
    if span < 6.0:
        # Nothing resembling a burst structure: treat the whole thing as active
        # rather than inventing a threshold inside the noise.
        return np.ones(count, dtype=bool)
    return db > db.min() + span * 0.35


def _spectrum(samples: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Averaged periodogram of the active blocks, and of the idle ones.

    Returns:
        `(active_psd, idle_psd)`. `idle_psd` is None when every block is active,
        which is the case that leaves no in-capture noise reference.
    """
    count = active.size
    blocks = samples[: count * _BLOCK].reshape(count, _BLOCK)
    window = np.hanning(_BLOCK)
    spectra = np.abs(np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)) ** 2
    idle = spectra[~active].mean(axis=0) if (~active).any() else None
    return spectra[active].mean(axis=0), idle


def _occupied_band(
    active_psd: np.ndarray, idle_psd: np.ndarray | None, sample_rate: float
) -> tuple[float | None, float | None]:
    """Carrier offset and occupied bandwidth, both in Hz.

    Occupied bandwidth is the narrowest contiguous span holding 99% of the power
    that sits above the noise floor -- the standard definition, applied to the
    excess rather than the raw spectrum so that a wide noise pedestal does not
    count as signal.

    Returns `(None, None)` when there is no idle region to estimate a floor
    from, because without it every bin looks occupied and the answer would be
    the capture bandwidth for any input.
    """
    if idle_psd is None:
        return None, None
    excess = np.clip(active_psd - idle_psd, 0.0, None)
    total = float(excess.sum())
    if total <= 0.0:
        return None, None

    cumulative = np.cumsum(excess) / total
    lower = int(np.searchsorted(cumulative, 0.005))
    upper = int(np.searchsorted(cumulative, 0.995))
    bins = active_psd.size
    hz_per_bin = sample_rate / bins
    bandwidth = max(1, upper - lower + 1) * hz_per_bin

    centroid = float((np.arange(bins) * excess).sum() / total)
    offset = (centroid - bins / 2.0) * hz_per_bin
    return offset, bandwidth


def _sample_spread(rec: Recording, chunks: int = 16) -> np.ndarray:
    """Read `SAMPLE_LIMIT` samples spread evenly across the whole recording.

    A prefix would be cheaper and is what this did first, and it was wrong: on
    the DASH7 captures the first packet starts around 1.05 s into an 8 s
    recording, so a 0.26 s prefix contains nothing but noise floor. The duty
    cycle, the burst position and the bandwidth estimate would all have been
    measured on silence, and -- worse -- they would have come back as confident
    numbers rather than as `None`.

    Chunks are block-aligned so the envelope and the periodogram see whole
    blocks; the discontinuities between chunks fall on block boundaries and
    affect at most `chunks` blocks of the envelope.
    """
    total = min(rec.num_samples, SAMPLE_LIMIT)
    per_chunk = max(_BLOCK, (total // chunks) // _BLOCK * _BLOCK)
    if rec.num_samples <= total or per_chunk >= rec.num_samples:
        return rec.read(0, total)
    step = max(per_chunk, (rec.num_samples - per_chunk) // max(1, chunks - 1))
    pieces = []
    for index in range(chunks):
        start = min(index * step, rec.num_samples - per_chunk)
        pieces.append(rec.read(start, per_chunk))
        if start + per_chunk >= rec.num_samples:
            break
    return np.concatenate(pieces)


def _declared_band(rec: Recording) -> tuple[float | None, float | None]:
    """Carrier offset and bandwidth from the annotations, if they declare them.

    This is the preferred source: it is what the publisher meant, not what a
    threshold inferred. Excluded labels are skipped, so a reference tone that
    carries no class information does not become the measured signal.
    """
    exclude = resolve_exclude_labels(None)
    spans = [
        (a.freq_lower_edge, a.freq_upper_edge)
        for a in rec.annotations
        if a.freq_lower_edge is not None
        and a.freq_upper_edge is not None
        and (a.label or "") not in exclude
    ]
    if not spans or rec.center_frequency is None:
        return None, None
    centres = [(lo + hi) / 2.0 - rec.center_frequency for lo, hi in spans]
    widths = [hi - lo for lo, hi in spans]
    return float(np.median(centres)), float(np.median(widths))


def measure_recording(rec: Recording, record_id: str, label: str | None) -> RecordFeatures:
    """Measure one recording's axes from a bounded prefix of its samples."""
    features = RecordFeatures(
        record_id=record_id,
        label=label,
        duration_samples=rec.num_samples,
        sample_rate=rec.sample_rate,
        capture_time=_capture_time(rec),
    )
    collection = rec.global_info.get("core:collection")
    features.collection = str(collection) if collection else None
    offset, bandwidth = _declared_band(rec)
    if bandwidth:
        features.carrier_offset_hz = offset
        features.occupied_bw_hz = bandwidth
        features.bw_source = "annotation"

    samples = _sample_spread(rec)
    if samples.size < _BLOCK * 2:
        return features

    power = float(np.mean(np.abs(samples) ** 2))
    features.mean_power_db = 10.0 * math.log10(power + 1e-30)

    active = _active_mask(samples)
    if active.size:
        features.duty_cycle = float(active.mean())
        where = np.flatnonzero(active)
        if where.size:
            features.burst_start_frac = float(where[0] / active.size)
        if features.bw_source is None:
            active_psd, idle_psd = _spectrum(samples, active)
            offset, bandwidth = _occupied_band(active_psd, idle_psd, rec.sample_rate)
            if bandwidth:
                features.carrier_offset_hz = offset
                features.occupied_bw_hz = bandwidth
                features.bw_source = "spectrum"
    return features


def _capture_time(rec: Recording) -> dt.datetime | None:
    """`core:datetime` from the first capture segment, if it parses.

    SigMF puts this in `captures`, not `global`. An earlier version read
    `global_info` and therefore found nothing on every conforming file, which
    silently turned the capture-time check into NOT CHECKED across the whole
    dataset survey -- including on a set whose two classes were recorded a week
    apart. `global` is still consulted as a fallback for files that put it
    there anyway.
    """
    raw = rec.capture_datetime or rec.global_info.get("core:datetime")
    if not isinstance(raw, str):
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Axis separability
# --------------------------------------------------------------------------


def separability(values: list[float], labels: list[str]) -> tuple[float, float]:
    """How well one scalar axis predicts the class, and the chance rate.

    Leave-one-out nearest neighbour on the single axis: for each recording, find
    the closest *other* recording by this value alone and take its label. No
    threshold to tune, and it works for any number of classes.

    Returns:
        `(accuracy, chance)` where chance is the largest class's share -- what a
        constant predictor would score.
    """
    n = len(values)
    counts = Counter(labels)
    chance = max(counts.values()) / n if n else 0.0
    if n < 3:
        return 0.0, chance
    array = np.asarray(values, dtype=float)
    correct = 0
    for i in range(n):
        distance = np.abs(array - array[i])
        distance[i] = np.inf
        if labels[int(np.argmin(distance))] == labels[i]:
            correct += 1
    return correct / n, chance


def axis_findings(features: list[RecordFeatures]) -> list[Finding]:
    """One finding per numeric axis, plus the capture-time check."""
    labelled = [f for f in features if f.label is not None]
    findings: list[Finding] = []
    if len(labelled) < 3:
        return [
            Finding(
                Status.NOT_CHECKED,
                "class axes",
                "fewer than 3 labelled recordings; separability is not defined",
            )
        ]

    for name in next(iter(labelled)).axes():
        pairs = [(f.axes()[name], f.label) for f in labelled if f.axes()[name] is not None]
        if len(pairs) < 3:
            findings.append(
                Finding(
                    Status.NOT_CHECKED,
                    name,
                    "not measurable on these recordings (metadata absent or "
                    "no idle region to reference)",
                )
            )
            continue
        values = [float(v) for v, _ in pairs]
        labels = [str(lbl) for _, lbl in pairs]
        score, chance = separability(values, labels)
        detail = (
            f"this axis alone classifies {score:.0%} of {len(pairs)} recordings "
            f"(chance {chance:.0%})"
        )
        risky = score >= RISK_SEPARABILITY and score > chance
        findings.append(Finding(Status.RISK if risky else Status.PASS_SAMPLE, name, detail))

    findings.append(_capture_time_finding(labelled))
    return findings


def _capture_time_finding(features: list[RecordFeatures]) -> Finding:
    """Flag classes that are separated in time -- the session confound."""
    timed = [f for f in features if f.capture_time is not None]
    if len(timed) < 3:
        return Finding(
            Status.NOT_CHECKED,
            "axis: capture time",
            "core:datetime absent from the recordings",
        )
    values = [f.capture_time.timestamp() for f in timed if f.capture_time]
    labels = [str(f.label) for f in timed]
    score, chance = separability(values, labels)
    status = Status.RISK if score >= RISK_SEPARABILITY else Status.PASS_SAMPLE
    return Finding(
        status,
        "axis: capture time",
        f"capture time alone classifies {score:.0%} of {len(timed)} recordings "
        f"(chance {chance:.0%})",
    )


def processing_gain_finding(features: list[RecordFeatures]) -> Finding:
    """Report the processing gain the class separation has available.

    The number that matters for designing an SNR sweep is not the wideband SNR
    the noise was set to but the SNR where the signal lives, and the difference
    between them is `10*log10(sample_rate / occupied_bandwidth)`. Measured on
    the DASH7 cabled set that difference is 25.9 dB, which is why an SNR grid
    built around wideband figures came back at 100% on every row.
    """
    usable = [f for f in features if f.occupied_bw_hz and f.sample_rate]
    if not usable:
        return Finding(
            Status.NOT_CHECKED,
            "in-band SNR",
            "occupied bandwidth undetermined: no idle region to estimate a "
            "noise floor from, and no frequency edges in the annotations",
        )
    gains = [10.0 * math.log10(f.sample_rate / f.occupied_bw_hz) for f in usable if f.sample_rate]
    bandwidths = [f.occupied_bw_hz for f in usable if f.occupied_bw_hz]
    gain = float(np.median(gains))
    bandwidth = float(np.median(bandwidths))
    sources = sorted({f.bw_source for f in usable if f.bw_source})
    return Finding(
        Status.PASS_SAMPLE,
        "in-band SNR",
        f"median occupied bandwidth {bandwidth / 1e3:.1f} kHz (from "
        f"{'/'.join(sources)}) of {usable[0].sample_rate / 1e6:.2f} MHz, so up to "
        f"{gain:.1f} dB of processing gain is available. An SNR set over the full "
        f"capture is {gain:.0f} dB lower than what the task sees",
    )


def difficulty_verdict(features: list[RecordFeatures]) -> tuple[str, bool]:
    """Decide between `ceiling` and `unknown`.

    Returns:
        `(verdict text, is_ceiling)`. There is no `measurable` outcome: locating
        the band where a model is partly right requires training one, and the
        pilot that motivated this command found that band 7 dB wide with the
        same arm scoring 37% and 80% on two seeds.
    """
    labelled = [f for f in features if f.label is not None]
    best_name, best_score, best_chance, best_margin = "", 0.0, 0.0, -1.0
    for name in labelled[0].axes() if labelled else {}:
        pairs = [(f.axes()[name], f.label) for f in labelled if f.axes()[name] is not None]
        if len(pairs) < 3:
            continue
        score, chance = separability([float(v) for v, _ in pairs], [str(x) for _, x in pairs])
        # Rank by margin over chance, not by raw score. An axis measurable on
        # only one class scores 100% against a chance of 100% and told the
        # first survey that a WiFi dataset was at the ceiling "because of the
        # carrier offset" -- true verdict, wrong reason, and the reason is what
        # a reader acts on.
        if score - chance > best_margin:
            best_name, best_score, best_chance, best_margin = name, score, chance, score - chance

    if best_score >= CEILING_SEPARABILITY and best_margin > 0.0:
        return (
            f"ceiling - {best_name.removeprefix('axis: ')} alone classifies "
            f"{best_score:.0%} of recordings (chance {best_chance:.0%}), so a "
            f"trained model will saturate and leave no room to measure a leak",
            True,
        )
    if not best_name:
        return ("unknown - no axis was measurable, so nothing was ruled out", False)
    return (
        f"unknown - no single measurable feature separates the classes "
        f"(best: {best_name.removeprefix('axis: ')} at {best_score:.0%} against a "
        f"chance of {best_chance:.0%}, {best_margin * 100:+.0f} points)",
        False,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _wrap(text: str, width: int, indent: int) -> list[str]:
    """Wrap `text` to `width`, with continuation lines indented.

    Paths get their own packing. `textwrap` breaks a long unbroken token
    mid-word, which turns an absolute path into `...temp/claude/C--Users-pr` /
    `oject/...` and makes it unreadable exactly when it is longest. Splitting
    on the separator keeps every fragment meaningful.
    """
    lines = _pack_path(text, width) if _looks_like_path(text) else textwrap.wrap(text, width=width)
    lines = lines or [""]
    return [lines[0]] + [" " * indent + line for line in lines[1:]]


def _looks_like_path(text: str) -> bool:
    """A single token containing separators, i.e. something to break on `/`."""
    return "/" in text and " " not in text.strip()


def _pack_path(text: str, width: int) -> list[str]:
    """Greedily pack path segments, keeping each separator with its segment."""
    parts = [segment + "/" for segment in text.split("/")]
    parts[-1] = parts[-1][:-1]
    lines: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > width:
            lines.append(current)
            current = part
        else:
            current += part
    if current:
        lines.append(current)
    return lines


def render_text(report: AuditReport) -> str:
    """The quotable fixed-width block.

    ASCII only, no box-drawing and no typographic dashes: the block is meant to
    be pasted into a paper, a LaTeX verbatim environment or an issue, and this
    project has already been bitten once by a non-UTF-8 console.
    """
    rule, thin = "=" * WIDTH, "-" * WIDTH
    out = [rule, "iqforge audit report", rule]

    def field_lines(name: str, text: str) -> list[str]:
        """One `label   value` row, wrapped so nothing exceeds WIDTH.

        Paths are the reason this exists: a dataset in a deep temporary
        directory runs past 78 columns on its own and would break the block for
        every reader who pasted it somewhere fixed-width.
        """
        wrapped = _wrap(text, WIDTH - 14, 14)
        return [f"{name:<14}{wrapped[0]}", *wrapped[1:]]

    for name, value in (
        ("tool", f"iqforge {report.tool_version}"),
        ("generated", report.generated),
        ("mode", report.mode),
        ("input", report.input_path),
    ):
        out.extend(field_lines(name, value))
    out.extend(" " * 14 + line for line in report.input_lines)
    out.extend(field_lines("fingerprint", report.fingerprint))

    out += ["", "CHECKS", thin]
    out.append(f"{'status':<{_STATUS_W}}  {'check':<{_CHECK_W}}  detail")
    indent = _STATUS_W + _CHECK_W + 4
    for finding in report.findings:
        detail = _wrap(finding.detail, _DETAIL_W, indent)
        out.append(f"{finding.status.value:<{_STATUS_W}}  {finding.check:<{_CHECK_W}}  {detail[0]}")
        out.extend(detail[1:])

    summary = report.summary
    counts = ", ".join(f"{value} {name}" for name, value in summary.items())
    out += ["", f"{'SUMMARY':<14}{counts}"]

    out += ["", "WHAT THIS DID NOT CHECK", thin]
    for item in report.did_not_check:
        out.extend(_wrap(f"- {item}", WIDTH - 2, 2))

    out += ["", *field_lines("VERDICT", report.verdict)]
    for index, line in enumerate(report.next_step):
        wrapped = _wrap(line, WIDTH - 14, 14)
        out.append(f"{'NEXT':<14}{wrapped[0]}" if index == 0 else " " * 14 + wrapped[0])
        out.extend(wrapped[1:])
    out.append(rule)
    return "\n".join(out)


def render_json(report: AuditReport) -> str:
    """Machine-readable form. `did_not_check` is mandatory here too."""
    return json.dumps(
        {
            "tool": "iqforge",
            "tool_version": report.tool_version,
            "generated": report.generated,
            "mode": report.mode,
            "input": report.input_path,
            "input_detail": report.input_lines,
            "fingerprint": report.fingerprint,
            "checks": [
                {"status": f.status.value, "check": f.check, "detail": f.detail}
                for f in report.findings
            ],
            "summary": {
                "leak": report.summary["leaks"],
                "pass": report.summary["passed"],
                "risk": report.summary["risk"],
                "not_checked": report.summary["not checked"],
            },
            "did_not_check": report.did_not_check,
            "verdict": report.verdict,
            "next": report.next_step,
        },
        indent=2,
        ensure_ascii=True,
    )


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def fingerprint(paths: list[Path]) -> str:
    """Short digest of the input's identity: sorted names plus sizes.

    Cheap and stable, so two reports can be checked for describing the same
    input. It is not a content hash and does not pretend to be.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        size = path.stat().st_size if path.exists() else -1
        digest.update(f"{path.name}:{size}\n".encode())
    return f"sha256:{digest.hexdigest()[:8]}  (sorted source names + sizes)"


def data_digest(path: Path, chunk: int = 1 << 20) -> str:
    """Identity digest of a data file: size plus its first and last megabyte.

    A full hash of a multi-gigabyte capture costs more than the rest of the
    audit put together. This catches a file republished under two names, which
    is the case worth catching; it does not catch two files that differ only in
    the middle, and `did_not_check` says so.
    """
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size > chunk:
            handle.seek(max(0, size - chunk))
            digest.update(handle.read(chunk))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------

#: Stated on every run, whatever the input. These are the things no amount of
#: index arithmetic or metadata reading can reach.
UNIVERSAL_CAVEATS = [
    "Whether a SigMF Collection means its members are statistically dependent. "
    "The format expresses that recordings are related and not that they must "
    "stay together, so a collection check is a hint in either direction and "
    "never a proof.",
    "Whether the labels are correct. What was checked is that they are "
    "consistent with what the label source says, not that the source is right. "
    "A label table that is internally consistent and wrong throughout looks "
    "identical from here.",
    "Physical independence of recordings. Two files sharing no samples and no "
    "air time can still be near-duplicates: a static indoor path does not "
    "change between two recorder runs seconds apart. No index arithmetic "
    "finds this, and it is the failure that ruled out a public dataset which "
    "passed every count-based check.",
    "Nuisance axes that are not observable from the data: channel and "
    "multipath state, and transmitter identity as distinct from propagation "
    "path.",
    "Whether an unmeasured axis separates the classes. No measurable feature "
    "separating them is not evidence that the task is clean.",
    "Whether a high-scoring axis is a confound or the intended class. If the "
    "class IS the carrier offset, a high score on that axis is the task, not a "
    "leak. This tool reports association and cannot tell the two apart.",
]


def audit_dataset(root: Path, manifest: dict[str, Any], tool_version: str) -> AuditReport:
    """Audit a built dataset, using its manifest as the source of truth."""
    config = manifest.get("config", {})
    window = int(config.get("window", 0) or 0)
    stride = int(config.get("stride", 0) or 0)
    splits = manifest.get("splits", {})

    total_windows = sum(int(s.get("count", 0)) for s in splits.values())
    label_map = manifest.get("label_map", {})
    source_files = [Path(p) for p in manifest.get("source_files", [])]

    overlap = 0.0 if not window else max(0.0, 1.0 - stride / window)
    report = AuditReport(
        tool_version=tool_version,
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode=f"built dataset (manifest_schema {manifest.get('manifest_schema', 'absent')})",
        input_path=root.as_posix(),
        input_lines=[
            f"{len(source_files)} recordings, {len(label_map)} classes, {total_windows} windows",
            f"window {window}, stride {stride}, {overlap:.0%} overlap",
        ],
        fingerprint=fingerprint(source_files),
        did_not_check=list(UNIVERSAL_CAVEATS),
    )

    label_counts: Counter[str] = Counter()
    for entry in splits.values():
        for record in entry.get("records") or []:
            if record.get("label") is not None:
                label_counts[str(record["label"])] += 1
    report.input_lines.extend(class_distribution_lines(label_counts))

    report.findings.extend(_split_findings(splits, window, stride))
    features, unresolved = _dataset_features(root, splits, source_files)
    assignment = {
        str(record.get("id")): name
        for name, entry in splits.items()
        for record in entry.get("records") or []
    }
    if features:
        report.findings.append(_split_time_overlap(features, assignment))
        report.findings.append(_shared_timestamp(features, assignment))
        report.findings.append(_collection_finding(features, assignment))
        report.findings.extend(axis_findings(features))
        report.findings.append(processing_gain_finding(features))
    else:
        report.findings.append(
            Finding(
                Status.NOT_CHECKED,
                "class axes",
                "source recordings not found next to the manifest, so no axis could be measured",
            )
        )
    if unresolved:
        report.did_not_check.append(
            f"{unresolved} of {len(source_files)} source recordings could not be "
            f"opened from the paths in the manifest; the axis checks cover only "
            f"the rest."
        )

    report.findings.append(
        Finding(
            Status.NOT_CHECKED,
            "task difficulty",
            "requires a probe run; audit rules out the trivial case only",
        )
    )
    verdict, is_ceiling = difficulty_verdict(features)
    report.verdict = verdict
    report.next_step = _next_step(is_ceiling, root)
    report.features = features
    return report


def _split_findings(splits: dict[str, Any], window: int, stride: int) -> list[Finding]:
    """Recording disjointness, and the overlap conclusion that follows from it."""
    where: dict[str, set[str]] = {}
    provenance_missing = []
    for name, entry in splits.items():
        records = entry.get("records")
        if not records and int(entry.get("count", 0)) > 0:
            provenance_missing.append(name)
        for record in records or []:
            where.setdefault(str(record.get("id")), set()).add(name)

    if provenance_missing:
        reason = (
            f"split(s) {', '.join(sorted(provenance_missing))} hold windows but "
            f"list no recordings, so which recording each window came from is "
            f"not recoverable"
        )
        return [
            Finding(Status.NOT_CHECKED, "recording disjointness", reason),
            Finding(
                Status.NOT_CHECKED,
                "cross-split overlap",
                "depends on recording disjointness, which could not be established",
            ),
        ]

    shared = sorted(rid for rid, names in where.items() if len(names) > 1)
    if shared:
        overlap_samples = max(0, window - stride)
        listed = ", ".join(shared[:3]) + (" ..." if len(shared) > 3 else "")
        return [
            Finding(
                Status.LEAK,
                "recording disjointness",
                f"{len(shared)} recording(s) appear in more than one split: {listed}",
            ),
            Finding(
                Status.LEAK,
                "cross-split overlap",
                f"windows from those recordings are {stride} samples apart and "
                f"{window} long, so adjacent pairs split across sides share "
                f"{overlap_samples} samples",
            ),
        ]

    return [
        Finding(
            Status.PASS_PROOF,
            "recording disjointness",
            f"no recording appears in more than one split ({len(where)} recordings)",
        ),
        Finding(
            Status.PASS_PROOF,
            "cross-split overlap",
            "impossible: windows can only overlap within a recording, and no "
            "recording spans two splits. Proven from structure, not sampled",
        ),
    ]


def _dataset_features(
    root: Path, splits: dict[str, Any], source_files: list[Path]
) -> tuple[list[RecordFeatures], int]:
    """Measure the axes for a built dataset's source recordings.

    Features are keyed by the manifest's own record id -- a path relative to the
    build input, not a file name. Keying them by file name made every lookup
    against the manifest miss: labels came back empty, so the axis checks went
    quiet, and the split lookup in `_split_time_overlap` compared None to None
    and reported 465 overlapping pairs as correctly grouped when none of them
    had been checked at all.
    """
    labels: dict[str, str] = {}
    for entry in splits.values():
        for record in entry.get("records") or []:
            if record.get("label") is not None:
                labels[str(record["id"])] = str(record["label"])

    by_suffix = {p.as_posix(): p for p in source_files}
    features: list[RecordFeatures] = []
    unresolved = 0
    for record_id in sorted(labels) or [p.as_posix() for p in source_files]:
        source = next(
            (
                p
                for key, p in by_suffix.items()
                if key == record_id or key.endswith("/" + record_id)
            ),
            Path(record_id),
        )
        resolved = _resolve(source, root)
        if resolved is None:
            unresolved += 1
            continue
        try:
            rec = load(resolved)
        except Exception:  # noqa: BLE001 - an unreadable source is a skip, not a crash
            unresolved += 1
            continue
        features.append(measure_recording(rec, record_id, labels.get(record_id)))
    return features, unresolved


def _resolve(path: Path, root: Path) -> Path | None:
    """Find a manifest-recorded source path, trying the likely roots."""
    for candidate in (path, root / path.name, root.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return None


def audit_recordings(
    root: Path,
    features: list[RecordFeatures],
    window: int,
    stride: int,
    tool_version: str,
    meta_paths: list[Path],
    unreadable: list[tuple[str, str]] | None = None,
    label_source: dict[str, Any] | None = None,
) -> AuditReport:
    """Audit a folder of recordings that has not been built into a dataset.

    Args:
        unreadable: `(record id, error)` for recordings that could not be
            opened. They are reported rather than fatal: a set the tool cannot
            fully read is a finding about the set, and refusing to say anything
            about the other 327 files is not an improvement on saying so.
    """
    unreadable = unreadable or []
    overlap = max(0.0, 1.0 - stride / window) if window else 0.0
    classes = {f.label for f in features if f.label is not None}
    report = AuditReport(
        tool_version=tool_version,
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="recording folder (no manifest)",
        input_path=root.as_posix(),
        input_lines=[
            f"{len(features)} recordings, {len(classes)} classes, not yet built"
            + (f" ({len(unreadable)} unreadable, skipped)" if unreadable else ""),
            f"window {window}, stride {stride}, {overlap:.0%} overlap if built this way",
        ],
        fingerprint=fingerprint(meta_paths),
        did_not_check=list(UNIVERSAL_CAVEATS),
    )

    report.input_lines.extend(
        class_distribution_lines(Counter(f.label for f in features if f.label is not None))
    )
    report.findings.append(_readable_finding(len(meta_paths), unreadable))
    if label_source is not None:
        report.findings.append(_label_source_finding(label_source))
    report.findings.append(_predicted_overlap(features, window, stride))
    report.findings.append(_time_overlap(features))
    report.findings.append(_shared_timestamp(features))
    report.findings.append(_duplicate_finding(meta_paths))
    report.findings.append(
        Finding(
            Status.NOT_CHECKED,
            "recording disjointness",
            "nothing has been split yet; run this again on the built dataset",
        )
    )
    report.findings.extend(axis_findings(features))
    report.findings.append(processing_gain_finding(features))
    report.findings.append(
        Finding(
            Status.NOT_CHECKED,
            "task difficulty",
            "requires a probe run; audit rules out the trivial case only",
        )
    )

    report.did_not_check.append(
        "Duplicate detection compares file size and the first and last megabyte "
        "of each data file. Two recordings that differ only in the middle are "
        "reported as distinct."
    )
    verdict, is_ceiling = difficulty_verdict(features)
    report.verdict = verdict
    report.next_step = _next_step(is_ceiling, root)
    report.features = features
    return report


def _readable_finding(total: int, unreadable: list[tuple[str, str]]) -> Finding:
    """How much of the set the tool could actually open.

    Reported as its own row rather than folded into the others, because every
    check below it covers only the readable part and a reader needs to know the
    denominator.
    """
    if not unreadable:
        return Finding(
            Status.PASS_PROOF,
            "recordings readable",
            f"all {total} recording(s) opened; every check below covers all of them",
        )
    listed = ", ".join(rid for rid, _ in unreadable[:2]) + (" ..." if len(unreadable) > 2 else "")
    return Finding(
        Status.RISK,
        "recordings readable",
        f"{len(unreadable)} of {total} could not be opened and are excluded from "
        f"every check below ({listed}): {unreadable[0][1]}",
    )


def _label_source_finding(source: dict[str, Any]) -> Finding:
    """Do the labels the CSV declares survive into the labelling?

    `build` refuses a collapse outright. Here it is a finding, because the whole
    point of auditing a folder is to decide before building, and the answer is
    more useful than the refusal: a table whose file names do not identify
    recordings gives every one of them a plausible label, so nothing downstream
    looks wrong.
    """
    declared: set[str] = source["declared"]
    assigned: set[str] = source["assigned"]
    unlisted: list[str] = source["unlisted"]
    name = Path(source["path"]).name

    lost = sorted(declared - assigned)
    if lost:
        return Finding(
            Status.RISK,
            "label source",
            f"proven: '{name}' gives these recordings {len(declared)} distinct label(s) but "
            f"only {len(assigned)} survive the lookup. Missing: {', '.join(lost)}. The "
            f"'filename' column does not identify recordings uniquely -- write it as the "
            f"path relative to this directory. build refuses this outright",
        )
    if unlisted:
        listed = ", ".join(sorted(unlisted)[:2]) + (" ..." if len(unlisted) > 2 else "")
        return Finding(
            Status.RISK,
            "label source",
            f"{len(unlisted)} recording(s) are not in '{name}' and would be dropped by "
            f"build ({listed})",
        )
    return Finding(
        Status.PASS_PROOF,
        "label source",
        f"all {len(declared)} label(s) '{name}' gives these recordings survive the lookup, "
        f"and every recording is listed",
    )


def _predicted_overlap(features: list[RecordFeatures], window: int, stride: int) -> Finding:
    """What a build with these parameters would produce. Arithmetic, not a measurement."""
    if not window or not stride:
        return Finding(Status.NOT_CHECKED, "predicted overlap", "no window/stride given")
    shared = max(0, window - stride)
    counts = [
        (f.duration_samples - window) // stride + 1
        for f in features
        if f.duration_samples and f.duration_samples >= window
    ]
    total = f"{sum(counts):,}".replace(",", " ")
    if shared == 0:
        return Finding(
            Status.PASS_PROOF,
            "predicted overlap",
            f"stride {stride} >= window {window}: windows would share no samples, "
            f"{total} windows in total",
        )
    return Finding(
        Status.RISK,
        "predicted overlap",
        f"adjacent windows would share {shared} of {window} samples ({shared / window:.0%}); "
        f"{total} windows total. Harmless under recording-level splitting, which is "
        f"what build does; fatal under any window-level split",
    )


def _is_placeholder_time(timed: list[RecordFeatures]) -> bool:
    """Do all these recordings declare one identical timestamp?

    A generator that stamps every file with the same constant is common --
    `examples/` here ships 16 recordings all dated 2024-01-01T00:00:00Z -- and
    reading that as sixteen simultaneous captures turns a placeholder into a
    proven leak. Genuinely simultaneous captures do not agree to the
    microsecond and do not all overlap each other; a single distinct value
    across the whole set is a constant, not a measurement.
    """
    return len({f.capture_time for f in timed}) == 1


def _spans(timed: list[RecordFeatures]) -> list[tuple[float, float, RecordFeatures]]:
    """`(start, end, feature)` per recording, ordered by start.

    Sorted on an explicit key. Sorting the tuples directly falls through to
    comparing `RecordFeatures` whenever two recordings share a start and an end,
    which is not a rare tie -- it is what identical timestamps in a capture set
    look like -- and raises `TypeError` mid-audit.
    """
    spans = [
        (
            f.capture_time.timestamp(),
            f.capture_time.timestamp() + f.duration_samples / f.sample_rate,
            f,
        )
        for f in timed
        if f.capture_time and f.duration_samples and f.sample_rate
    ]
    spans.sort(key=lambda s: (s[0], s[1], s[2].record_id))
    return spans


def _collection_finding(features: list[RecordFeatures], assignment: dict[str, str]) -> Finding:
    """Do members of one `core:collection` sit in different splits?

    Never `PASS/proof`, however clean the answer, and the ceiling is deliberate.
    A Collection asserts that recordings are *related*; it does not assert that
    they are statistically dependent. A collection of "every recording in my
    paper" and one of "the four simultaneous receptions of one frame" are the
    same object here, and only the second is a constraint. So members landing
    together is worth reporting and is not evidence of anything, while members
    landing apart is worth flagging and is not proof of a leak either -- it is
    the strongest hint the format is able to give.
    """
    declared = [f for f in features if f.collection]
    if not declared:
        return Finding(
            Status.NOT_CHECKED,
            "collection members",
            "no recording declares core:collection, so SigMF's own grouping field "
            "says nothing about this dataset",
        )
    where: dict[str, set[str]] = {}
    for feature in declared:
        assert feature.collection is not None
        where.setdefault(feature.collection, set()).add(assignment.get(feature.record_id, "?"))
    split_apart = sorted(name for name, splits in where.items() if len(splits) > 1)
    if split_apart:
        listed = ", ".join(split_apart[:3]) + (" ..." if len(split_apart) > 3 else "")
        return Finding(
            Status.RISK,
            "collection members",
            f"{len(split_apart)} of {len(where)} collection(s) have members in more than "
            f"one split ({listed}). SigMF does not say whether a collection means "
            f"'not independent', so this is a hint rather than a proven leak; "
            f"--group-by collection holds them together",
        )
    return Finding(
        Status.PASS_SAMPLE,
        "collection members",
        f"all {len(where)} declared collection(s) are each within one split. Not proof "
        f"of independence: a collection asserts that recordings are related, not that "
        f"they are dependent, and unrelated recordings may share one",
    )


def _split_time_overlap(features: list[RecordFeatures], assignment: dict[str, str]) -> Finding:
    """Do two recordings that share air time sit in different splits?

    The acquisition-level counterpart of the window-level overlap proof. Two
    recordings can be separate files, separate rows in the manifest and still
    be the same instant of radio observed twice -- a LoRa frame heard by four
    receivers at once is four files and one event. Recording-level splitting
    does not help there, because the unit of independence is the transmission,
    not the file. `--group-by` is the fix, and this is the check that says
    whether it worked.
    """
    timed = [
        f
        for f in features
        if f.capture_time and f.duration_samples and f.sample_rate and f.sample_rate > 0
    ]
    if len(timed) < 2:
        return Finding(
            Status.NOT_CHECKED,
            "shared air time",
            "core:datetime missing or unparseable, so air time cannot be compared",
        )
    if _is_placeholder_time(timed):
        return Finding(
            Status.NOT_CHECKED,
            "shared air time",
            f"all {len(timed)} recordings declare the same core:datetime, which is a "
            f"placeholder rather than a capture time; air time cannot be compared",
        )
    spans = _spans(timed)
    together = 0
    split_apart: list[tuple[str, str]] = []
    for i, (_, end_i, a) in enumerate(spans):
        for start_j, _, b in spans[i + 1 :]:
            if start_j >= end_i:
                break
            if assignment.get(a.record_id) == assignment.get(b.record_id):
                together += 1
            else:
                split_apart.append((a.record_id, b.record_id))
    if split_apart:
        listed = ", ".join(f"{x} / {y}" for x, y in split_apart[:2])
        return Finding(
            Status.LEAK,
            "shared air time",
            f"{len(split_apart)} pair(s) share air time but landed in different "
            f"splits: {listed}. Group them with --group-by",
        )
    if together:
        return Finding(
            Status.PASS_PROOF,
            "shared air time",
            f"{together} pair(s) share air time and every one of them is in a single split",
        )
    return Finding(
        Status.PASS_SAMPLE,
        "shared air time",
        f"no two of {len(timed)} recordings claim overlapping air time",
    )


def _shared_timestamp(
    features: list[RecordFeatures], assignment: dict[str, str] | None = None
) -> Finding:
    """Do recordings that share a capture stamp land as the Vega-C pattern?

    Sibling of shared air time. That check looks for *intersecting* intervals;
    this one looks for recordings that carry the *same* `core:datetime` value.
    The difference is the Vega-C MEO Cubesats set (methodology §6.2): five
    satellites, the same three capture stamps, three passes that do not overlap
    each other. Shared air time is quiet there -- nothing intersects across
    splits, because a recording-level split puts a different pass in each bin.
    That is distribution shift, and it is what invalidated the first SNR grid.

    The placeholder guard that silences shared air time does not apply. A
    generator constant is one stamp across the whole set (`examples/` is dated
    2024-01-01T00:00:00Z). Here several distinct stamps each repeat in every
    class, and that repetition is the signal.

    Reported as RISK, never LEAK, and the distinction is not a severity dial.
    Every other LEAK in this tool means *the same material is on both sides* --
    overlapping air time, identical data, a recording split across bins. The
    Vega-C pattern is the opposite: *different material on each side*, a
    different pass with its own Doppler, elevation and SNR in every split.
    That is distribution shift, which is what methodology §6.2 calls it. Both
    invalidate a measurement, and they invalidate it in opposite directions --
    leakage inflates the score, shift depresses it. Filing one under the other
    costs the word "LEAK" its meaning, and this tool's value is that its
    statuses mean something precise.

    RISK is not a downgrade in consequence. `measure-leakage` refuses the set
    either way (category 2, `preflight.decide` accepts LEAK and RISK alike),
    and `audit --strict` still exits non-zero. What changes is that `audit`
    without `--strict` no longer exits 1 on it, which is correct: nothing here
    is proven to leak.

    Args:
        assignment: Split each recording landed in. Absent in folder mode, where
            nothing has been split yet. Without it the pattern is reported
            without reference to splits, because the bins do not exist to fall
            into.
    """
    timed = [f for f in features if f.capture_time is not None and f.label is not None]
    if len(timed) < 2:
        return Finding(
            Status.NOT_CHECKED,
            "shared timestamp",
            "core:datetime missing or unparseable, so timestamps cannot be compared",
        )
    if _is_placeholder_time(timed):
        return Finding(
            Status.NOT_CHECKED,
            "shared timestamp",
            f"all {len(timed)} recordings declare the same core:datetime, which is a "
            f"placeholder rather than a capture time; timestamps cannot be compared",
        )

    by_stamp: dict[dt.datetime, list[RecordFeatures]] = {}
    for feature in timed:
        assert feature.capture_time is not None
        by_stamp.setdefault(feature.capture_time, []).append(feature)
    crossed = {stamp: recs for stamp, recs in by_stamp.items() if len({f.label for f in recs}) > 1}
    if not crossed:
        return Finding(
            Status.PASS_SAMPLE,
            "shared timestamp",
            f"no core:datetime value of {len(by_stamp)} appears in more than one class",
        )

    if assignment is None:
        return Finding(
            Status.RISK,
            "shared timestamp",
            f"{len(crossed)} distinct core:datetime value(s) each appear in more than "
            f"one class. A recording-level split puts a different acquisition in "
            f"each split -- the Vega-C pattern (methodology §6.2)",
        )

    stamp_sets: dict[str, set[dt.datetime]] = {}
    for stamp, recs in crossed.items():
        for feature in recs:
            stamp_sets.setdefault(assignment.get(feature.record_id, "?"), set()).add(stamp)
    partitioned = len({frozenset(stamps) for stamps in stamp_sets.values()}) > 1
    if partitioned:
        return Finding(
            Status.RISK,
            "shared timestamp",
            f"{len(crossed)} distinct core:datetime value(s) each appear in more than "
            f"one class and the splits do not share the same set of them. A "
            f"recording-level split therefore puts a different acquisition in each "
            f"split -- the Vega-C pattern (methodology §6.2). This is distribution "
            f"shift, not leakage: the splits hold different material rather than "
            f"the same material twice",
        )
    return Finding(
        Status.PASS_SAMPLE,
        "shared timestamp",
        f"{len(crossed)} timestamp(s) appear in more than one class and every split "
        f"holds the same set of them",
    )


def _time_overlap(features: list[RecordFeatures]) -> Finding:
    """Do two recordings claim the same air time?

    A recording-level version of the same index-intersection idea: intervals in
    seconds instead of sample offsets. Only as good as `core:datetime`.
    """
    timed = [
        f
        for f in features
        if f.capture_time and f.duration_samples and f.sample_rate and f.sample_rate > 0
    ]
    if len(timed) < 2:
        return Finding(
            Status.NOT_CHECKED,
            "recording time overlap",
            "core:datetime missing or unparseable, so air time cannot be compared",
        )
    if _is_placeholder_time(timed):
        return Finding(
            Status.NOT_CHECKED,
            "recording time overlap",
            f"all {len(timed)} recordings declare the same core:datetime, which is a "
            f"placeholder rather than a capture time; air time cannot be compared",
        )
    spans = _spans(timed)
    clashes = [
        (a[2].record_id, b[2].record_id)
        for a, b in zip(spans, spans[1:], strict=False)
        if b[0] < a[1]
    ]
    if clashes:
        first = ", ".join(f"{x} / {y}" for x, y in clashes[:2])
        return Finding(
            Status.LEAK,
            "recording time overlap",
            f"{len(clashes)} pair(s) claim overlapping air time: {first}",
        )
    return Finding(
        Status.PASS_SAMPLE,
        "recording time overlap",
        f"no two of {len(timed)} recordings claim overlapping air time",
    )


def _duplicate_finding(meta_paths: list[Path]) -> Finding:
    """Republished-under-two-names detection."""
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    checked = 0
    for meta in meta_paths:
        data = meta.with_suffix(".sigmf-data")
        if not data.exists():
            continue
        checked += 1
        digest = data_digest(data)
        if digest in seen:
            duplicates.append((seen[digest], data.name))
        else:
            seen[digest] = data.name
    if not checked:
        return Finding(Status.NOT_CHECKED, "duplicate data files", "no data files found")
    if duplicates:
        listed = ", ".join(f"{a} / {b}" for a, b in duplicates[:2])
        return Finding(
            Status.LEAK,
            "duplicate data files",
            f"{len(duplicates)} pair(s) of recordings have identical data: {listed}",
        )
    return Finding(
        Status.PASS_SAMPLE,
        "duplicate data files",
        f"no two of {checked} data files match on size and their first and last megabyte",
    )


def class_distribution_lines(counts: Counter[str]) -> list[str]:
    """Class counts and the chance line, as input description rather than a check.

    Imbalance is not a finding. Rare-event detection is a normal thing to build
    a dataset for, so a threshold on skew would fire on a whole legitimate
    category until the warning stopped being read. What makes a distribution
    actionable is the number a constant predictor would score, and that needs no
    threshold at all -- it is reported always, and the reader decides.
    """
    if not counts:
        return []
    total = sum(counts.values())
    ranked = counts.most_common()
    shown = ", ".join(f"{name} {n}" for name, n in ranked[:4])
    if len(ranked) > 4:
        shown += f", +{len(ranked) - 4} more"
    return [
        f"classes: {shown}",
        f"chance {ranked[0][1] / total:.1%} (a constant predictor scoring "
        f"'{ranked[0][0]}' every time)",
    ]


def _next_step(is_ceiling: bool, root: Path) -> list[str]:
    """The handoff to `measure-leakage`, or the reason not to bother.

    Each entry is a paragraph, not a pre-broken line: the renderer wraps them,
    and pre-breaking here produced orphan words when the two disagreed.
    """
    if is_ceiling:
        return [
            "do not run measure-leakage on this data. A model will score at the "
            "ceiling in both arms and the measured inflation will be zero for a "
            "reason that has nothing to do with splitting."
        ]
    return [
        f"iqforge measure-leakage {root.as_posix()}",
        "audit cannot locate the band where a model is partly right, only rule "
        "out the trivial case. That band is what a leakage measurement needs.",
    ]


def collect_meta_paths(root: Path) -> list[Path]:
    """Every `.sigmf-meta` under `root`, sorted."""
    return sorted(root.rglob(f"*{META_EXT}"))
