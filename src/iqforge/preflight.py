"""Refuse path for `iqforge measure-leakage`.

This module does not train. It runs the same audit the user can run by hand,
classifies the outcome into the six categories that eliminated four public
datasets and let a fifth through (methodology §6), estimates the work a
measurement would do, and stops. The paired experiment lives in
`iqforge.measurement` and is not called from here.

Three outcomes, and no silent fourth:

- `REFUSED` — a category fired; measuring would report the wrong thing.
- `INCONCLUSIVE` — the sources the categories need are missing, so neither
  refusing nor measuring would be honest.
- `WOULD MEASURE` — nothing in the list fired. This version still does not
  train; it says so and reports the work a later version would do.

`--force` does not hide the category. It changes the header so a pasted block
cannot be mistaken for a clean run, and it changes the decision to
`WOULD MEASURE`. The reason that was overridden stays in the body.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from iqforge import __version__
from iqforge.audit import (
    WIDTH,
    AuditReport,
    Finding,
    RecordFeatures,
    Status,
    _spans,
    _wrap,
    difficulty_verdict,
)
from iqforge.io import IQForgeError
from iqforge.measurement import DEFAULT_SPLIT, DEFAULT_WINDOW, EPOCHS
from iqforge.splitting import parse_ratios

#: Gap at or below which two same-class captures are treated as the DASH7
#: indoor pattern (methodology §6.3): separate recorder runs, physically the
#: same channel. 43 seconds in that set; two minutes is the documented bar.
NEAR_DUPLICATE_GAP = dt.timedelta(seconds=120)

#: Frame snippets (LoRaIQ's 15 ms segments) are not this pattern. The indoor
#: captures were 8 s long; one second is the floor that keeps a burst of
#: short, labelled frames from looking like a repeated room.
NEAR_DUPLICATE_MIN_DURATION_S = 1.0

#: Durations of a near-duplicate pair must agree this closely. A 8 s run
#: next to a 0.05 s snippet is not the same acquisition repeated.
NEAR_DUPLICATE_DURATION_TOLERANCE = 0.10

#: Batch size the baseline trains at. The throughput probe uses the same
#: number so the extrapolation is about the loop that would actually run.
PROBE_BATCH = 64

#: Dummy steps timed after warmup. Enough to drown startup, not a benchmark.
PROBE_STEPS = 5


class DecisionStatus(StrEnum):
    """What `measure-leakage` concluded, before any training."""

    REFUSED = "REFUSED"
    INCONCLUSIVE = "INCONCLUSIVE"
    WOULD_MEASURE = "WOULD MEASURE"


class Category(IntEnum):
    """The six reasons a measurement is refused, or not.

    Categories 1-4 are the four datasets methodology §6 eliminated, numbered
    as that section numbers them, so the command can cite `category 4` rather
    than a paragraph. Category 5 is any remaining proven leak the audit
    already names. Category 6 is a split that `build` would refuse. There is
    no category for LoRaIQ: that is the case that is not refused (§6.5).
    """

    UNREADABLE = 1
    SHARED_TIMESTAMP = 2
    INDEPENDENCE = 3
    CEILING = 4
    STRUCTURAL_LEAK = 5
    CANNOT_SPLIT = 6


#: Short name and the citation a reader can follow. Category 5 cites the
#: audit finding, not §6.5 — that section is the dataset that passed.
CATEGORY_META: dict[Category, tuple[str, str]] = {
    Category.UNREADABLE: ("unreadable format", "methodology 6.1"),
    Category.SHARED_TIMESTAMP: ("shared timestamp", "methodology 6.2"),
    Category.INDEPENDENCE: ("physical independence", "methodology 6.3"),
    Category.CEILING: ("ceiling", "methodology 6.4"),
    Category.STRUCTURAL_LEAK: ("structural leak", "audit LEAK"),
    Category.CANNOT_SPLIT: ("cannot split", "SPEC 5.6"),
}


@dataclass(frozen=True)
class WorkEstimate:
    """What a default paired cell would train, if this command trained.

    Attributes:
        train_windows: Predicted training-split size from recording lengths.
        test_windows: Predicted test-split size.
        total_windows: All recordings, before the split.
        arms: Always 2 (recording-level and window-level).
        epochs: The published-table length.
        seconds: Wall time from the dummy-batch probe, or None if it could
            not run (no torch, or the probe failed).
    """

    train_windows: int
    test_windows: int
    total_windows: int
    arms: int = 2
    epochs: int = EPOCHS
    seconds: float | None = None


@dataclass
class Decision:
    """One refuse-path conclusion."""

    status: DecisionStatus
    reason: str
    category: Category | None = None
    forced: bool = False
    forced_past: str | None = None
    work: WorkEstimate | None = None
    audit: AuditReport | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def name(self) -> str | None:
        """Short name of the category, or None when nothing fired."""
        if self.category is None:
            return None
        return CATEGORY_META[self.category][0]

    @property
    def cites(self) -> str | None:
        """Where to read the case this category was numbered from."""
        if self.category is None:
            return None
        return CATEGORY_META[self.category][1]


def _finding(report: AuditReport | None, check: str) -> Finding | None:
    if report is None:
        return None
    return next((f for f in report.findings if f.check == check), None)


def _is_unreadable(report: AuditReport | None, unreadable_error: str | None) -> str | None:
    """Return a reason if the set is the AirID case, else None."""
    if unreadable_error:
        return unreadable_error
    if report is None:
        return None
    readable = _finding(report, "recordings readable")
    if readable is None or readable.status is not Status.RISK:
        return None
    # A conversion step between the publisher and the measurement is the
    # thing §6.1 is unwilling to do. Any unsupported datatype in the set
    # is that step, even when the rest of the files opened.
    if "Unsupported" in readable.detail or "cf16" in readable.detail:
        return readable.detail
    return None


def _near_duplicate_pairs(features: list[RecordFeatures]) -> list[tuple[str, str, float]]:
    """Same-class long captures whose gaps are the DASH7 indoor pattern.

    Returns `(id_a, id_b, gap_seconds)` for each pair that fired. Short
    frame snippets are excluded so LoRaIQ's consecutive transmissions do
    not look like a repeated indoor channel.
    """
    by_label: dict[str, list[RecordFeatures]] = defaultdict(list)
    for feature in features:
        if feature.label is None:
            continue
        if not feature.capture_time or not feature.duration_samples or not feature.sample_rate:
            continue
        if feature.sample_rate <= 0:
            continue
        duration_s = feature.duration_samples / feature.sample_rate
        if duration_s < NEAR_DUPLICATE_MIN_DURATION_S:
            continue
        by_label[feature.label].append(feature)

    pairs: list[tuple[str, str, float]] = []
    for recs in by_label.values():
        recs = sorted(recs, key=lambda f: f.capture_time or dt.datetime.min)
        for index, current in enumerate(recs):
            assert current.capture_time and current.duration_samples and current.sample_rate
            end = current.capture_time + dt.timedelta(
                seconds=current.duration_samples / current.sample_rate
            )
            duration = current.duration_samples / current.sample_rate
            for other in recs[index + 1 :]:
                assert other.capture_time and other.duration_samples and other.sample_rate
                gap = other.capture_time - end
                if gap > NEAR_DUPLICATE_GAP:
                    break
                if gap < dt.timedelta(0):
                    # Overlap is shared air time, not this pattern.
                    continue
                other_duration = other.duration_samples / other.sample_rate
                longer = max(duration, other_duration)
                if longer <= 0:
                    continue
                if abs(duration - other_duration) / longer > NEAR_DUPLICATE_DURATION_TOLERANCE:
                    continue
                pairs.append((current.record_id, other.record_id, gap.total_seconds()))
    return pairs


def _overlap_pairs(features: list[RecordFeatures]) -> list[tuple[str, str]]:
    """Recording ids whose capture intervals intersect."""
    timed = [
        f
        for f in features
        if f.capture_time and f.duration_samples and f.sample_rate and f.sample_rate > 0
    ]
    spans = _spans(timed)
    pairs: list[tuple[str, str]] = []
    for index, (_, end_i, left) in enumerate(spans):
        for start_j, _, right in spans[index + 1 :]:
            if start_j >= end_i:
                break
            pairs.append((left.record_id, right.record_id))
    return pairs


def _ungrouped_overlap_pairs(
    features: list[RecordFeatures], group_keys: dict[str, str] | None
) -> list[tuple[str, str]]:
    """Overlapping pairs that `--group-by` would not hold together.

    Folder-mode time overlap is a LEAK because the files claim the same air
    time. With `--group-by` those files are one unit, which is the LoRaIQ
    fix, and refusing the measurement for a leak the flag already holds
    together would refuse the one dataset the search found.
    """
    pairs = _overlap_pairs(features)
    if not group_keys:
        return pairs
    return [pair for pair in pairs if group_keys.get(pair[0]) != group_keys.get(pair[1])]


def _leaks_not_held(
    report: AuditReport | None,
    features: list[RecordFeatures],
    group_keys: dict[str, str] | None,
) -> list[Finding]:
    if report is None:
        return []
    held_overlap = bool(group_keys) or not _ungrouped_overlap_pairs(features, group_keys)
    leaks: list[Finding] = []
    for finding in report.findings:
        if finding.status is not Status.LEAK or finding.check == "shared timestamp":
            continue
        if finding.check in {"recording time overlap", "shared air time"} and held_overlap:
            continue
        leaks.append(finding)
    return leaks


def _cannot_split_reason(
    features: list[RecordFeatures],
    split: str,
    group_keys: dict[str, str] | None,
) -> str | None:
    """Mirror of the `build` error, without performing the split."""
    labelled = {f.record_id: f.label for f in features if f.label is not None}
    if not labelled:
        return "no labelled recordings; a recording-level split has nothing to allocate"
    try:
        ratios = parse_ratios(split)
    except IQForgeError as exc:
        return str(exc)
    active = sum(1 for ratio in ratios if ratio > 0)
    if group_keys:
        members: dict[str, list[str]] = defaultdict(list)
        for record_id in labelled:
            members[group_keys.get(record_id, record_id)].append(record_id)
        for key, ids in members.items():
            labels = {labelled[i] for i in ids}
            if len(labels) > 1:
                return (
                    f"group '{key}' spans classes {sorted(labels)}; a unit cannot carry two "
                    f"strata into one split. --group-by holds recordings together, so the "
                    f"key must not cross a class boundary"
                )
        by_label: dict[str, set[str]] = defaultdict(set)
        for record_id, label in labelled.items():
            by_label[str(label)].add(group_keys.get(record_id, record_id))
        noun = "group"
    else:
        by_label = defaultdict(set)
        for record_id, label in labelled.items():
            by_label[str(label)].add(record_id)
        noun = "recording"
    for label, units in sorted(by_label.items()):
        if len(units) < active:
            return (
                f"class '{label}' has only {len(units)} {noun}"
                f"{'' if len(units) == 1 else 's'}, but a {split} split needs "
                f"at least {active}"
            )
    return None


def predicted_windows(
    features: list[RecordFeatures], window: int, stride: int
) -> tuple[int, int, int]:
    """`(total, train, test)` window counts from recording lengths.

    Arithmetic, not a build. Uses `DEFAULT_SPLIT`'s train and test ratios.
    Recordings shorter than one window contribute nothing, matching `build`.
    """
    if window <= 0 or stride <= 0:
        return 0, 0, 0
    total = 0
    for feature in features:
        if feature.duration_samples and feature.duration_samples >= window:
            total += (feature.duration_samples - window) // stride + 1
    ratios = parse_ratios(DEFAULT_SPLIT)
    train = int(round(total * ratios[0]))
    test = int(round(total * ratios[2]))
    return total, train, test


def probe_seconds_per_window_epoch(window: int = DEFAULT_WINDOW) -> float | None:
    """Time one dummy batch of the baseline; return seconds per window-epoch.

    This is a throughput probe, not a measurement of the user's data. It
    trains on zeros. Returns None when torch is missing or the probe fails,
    so a machine without the extra still gets a refuse decision.
    """
    try:
        import torch

        from iqforge.models import BaselineCNN
        from iqforge.training import DEFAULT_DEVICE, resolve_device
    except ImportError:
        return None
    try:
        device = resolve_device(DEFAULT_DEVICE)
        model = BaselineCNN(num_classes=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.zeros(PROBE_BATCH, 2, window, device=device)
        labels = torch.zeros(PROBE_BATCH, dtype=torch.long, device=device)
        model.train()
        for _ in range(2):
            optimiser.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimiser.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(PROBE_STEPS):
            optimiser.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimiser.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except Exception:  # noqa: BLE001 - a failed probe is a missing estimate
        return None
    windows = PROBE_BATCH * PROBE_STEPS
    if elapsed <= 0 or windows <= 0:
        return None
    return elapsed / windows


def estimate_work(
    features: list[RecordFeatures],
    window: int,
    stride: int,
    *,
    seconds_per_window_epoch: float | None | object = ...,
) -> WorkEstimate:
    """Window counts plus, when possible, a wall-time from the dummy probe."""
    total, train, test = predicted_windows(features, window, stride)
    rate: float | None
    if seconds_per_window_epoch is ...:
        rate = probe_seconds_per_window_epoch(window)
    else:
        rate = seconds_per_window_epoch  # type: ignore[assignment]
    seconds = None
    if rate is not None and train > 0:
        seconds = rate * train * EPOCHS * 2
    return WorkEstimate(
        train_windows=train,
        test_windows=test,
        total_windows=total,
        seconds=seconds,
    )


def _verdict_token(report: AuditReport | None) -> str:
    if report is None or not report.verdict:
        return "unknown"
    return report.verdict.split(" ", 1)[0]


def decide(
    report: AuditReport | None,
    *,
    force: bool = False,
    split: str = DEFAULT_SPLIT,
    window: int = DEFAULT_WINDOW,
    stride: int = 512,
    group_keys: dict[str, str] | None = None,
    unreadable_error: str | None = None,
    seconds_per_window_epoch: float | None | object = ...,
) -> Decision:
    """Classify an audit into a refuse category, or say it would measure.

    Args:
        report: The audit that was just run. None when the folder could not
            be opened at all (the AirID case of every file unreadable).
        force: Override a refusal. The category remains in the body.
        split: Ratios a later measurement would pass to `build`.
        window: Window length used for the work estimate.
        stride: Step used for the work estimate.
        group_keys: `--group-by` map, so overlapping files that already share
            a unit are not refused as a structural leak.
        unreadable_error: The error from a fully unreadable folder.
        seconds_per_window_epoch: Injected probe rate. Omit to run the probe;
            pass None to skip it.
    """
    features = list(report.features) if report is not None else []
    # Window counts are cheap and always useful. The dummy-batch probe is
    # only worth running when a measurement would actually start.
    work = estimate_work(features, window, stride, seconds_per_window_epoch=None)

    category: Category | None = None
    status = DecisionStatus.WOULD_MEASURE
    reason = (
        "audit did not fire a refuse category. This command does not train; "
        "a later version would run the paired experiment at the default "
        "operating point"
    )

    unreadable = _is_unreadable(report, unreadable_error)
    timestamp = _finding(report, "shared timestamp")
    near = _near_duplicate_pairs(features)
    _, is_ceiling = difficulty_verdict(features) if features else ("", False)
    leaks = _leaks_not_held(report, features, group_keys)
    split_reason = _cannot_split_reason(features, split, group_keys)

    if unreadable:
        category = Category.UNREADABLE
        status = DecisionStatus.REFUSED
        reason = unreadable
    elif timestamp is not None and timestamp.status in {Status.LEAK, Status.RISK}:
        category = Category.SHARED_TIMESTAMP
        status = DecisionStatus.REFUSED
        reason = timestamp.detail
    elif near and not group_keys:
        category = Category.INDEPENDENCE
        status = DecisionStatus.REFUSED
        first = near[0]
        reason = (
            f"{len(near)} same-class pair(s) are long captures {first[2]:.0f} s apart "
            f"({first[0]} / {first[1]}). They pass every structural independence "
            f"test and are the same channel realisation physically -- the DASH7 "
            f"ds_indoor pattern (methodology 6.3). Measuring leakage on top of "
            f"that would measure the sum of the two"
        )
    elif is_ceiling:
        category = Category.CEILING
        status = DecisionStatus.REFUSED
        reason = report.verdict if report is not None else "ceiling"
    elif leaks:
        category = Category.STRUCTURAL_LEAK
        status = DecisionStatus.REFUSED
        reason = leaks[0].detail
    elif report is not None and not features and report.mode.startswith("built dataset"):
        status = DecisionStatus.INCONCLUSIVE
        reason = (
            "source recordings could not be measured, so ceiling and independence "
            "were not assessed and refusing would be a guess"
        )
    elif split_reason:
        category = Category.CANNOT_SPLIT
        status = DecisionStatus.REFUSED
        reason = split_reason
    elif report is None:
        status = DecisionStatus.INCONCLUSIVE
        reason = unreadable_error or "audit produced no report"

    forced_past = None
    if force and status is not DecisionStatus.WOULD_MEASURE:
        if category is Category.CEILING:
            forced_past = f"audit VERDICT '{_verdict_token(report)}'"
        elif category is not None:
            forced_past = f"category {int(category)} '{CATEGORY_META[category][0]}'"
        else:
            forced_past = "an INCONCLUSIVE audit"
        status = DecisionStatus.WOULD_MEASURE

    if status is DecisionStatus.WOULD_MEASURE and seconds_per_window_epoch is not None:
        work = estimate_work(
            features, window, stride, seconds_per_window_epoch=seconds_per_window_epoch
        )

    return Decision(
        status=status,
        reason=reason,
        category=category,
        forced=bool(forced_past),
        forced_past=forced_past,
        work=work,
        audit=report,
        findings=list(report.findings) if report is not None else [],
    )


def _field_lines(name: str, text: str) -> list[str]:
    wrapped = _wrap(text, WIDTH - 14, 14)
    return [f"{name:<14}{wrapped[0]}", *wrapped[1:]]


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


def _title(decision: Decision) -> str:
    if decision.forced and decision.forced_past:
        return f"iqforge leakage measurement -- FORCED PAST {decision.forced_past}"
    return "iqforge leakage measurement"


def render_text(decision: Decision) -> str:
    """The quotable fixed-width block. ASCII, 78 columns, same contract as audit."""
    rule, thin = "=" * WIDTH, "-" * WIDTH
    out = [rule, _title(decision), rule]
    report = decision.audit
    out.extend(_field_lines("tool", f"iqforge {__version__}"))
    generated = (
        report.generated
        if report is not None
        else dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    out.extend(_field_lines("generated", generated))
    if report is not None:
        out.extend(_field_lines("mode", report.mode))
        out.extend(_field_lines("input", report.input_path))
        out.extend(" " * 14 + line for line in report.input_lines)
        out.extend(_field_lines("audit", report.fingerprint))
        out.extend(_field_lines("verdict", report.verdict))
    else:
        out.extend(_field_lines("mode", "unreadable folder"))

    out += ["", "DECISION", thin]
    out.extend(_field_lines("status", decision.status.value))
    if decision.category is not None:
        out.extend(
            _field_lines(
                "category",
                f"{int(decision.category)}  {decision.name}  ({decision.cites})",
            )
        )
    else:
        out.extend(_field_lines("category", "none"))
    out.extend(_field_lines("reason", decision.reason))
    if decision.forced:
        out.extend(
            _field_lines(
                "forced",
                "yes. The category above still stands; this run is not a clean measurement",
            )
        )

    out += ["", "WORK", thin]
    work = decision.work
    if work is None or work.total_windows == 0:
        out.extend(
            _field_lines(
                "estimate",
                "no window count from these recordings; this command does not train",
            )
        )
    else:
        out.extend(
            _field_lines(
                "cell",
                f"2 arms x {work.epochs} epochs, ~{work.train_windows} train / "
                f"~{work.test_windows} test windows (split {DEFAULT_SPLIT}, "
                f"{work.total_windows} windows total)",
            )
        )
        if work.seconds is None:
            out.extend(
                _field_lines(
                    "estimate",
                    "not timed on this machine (torch missing, or the dummy-batch "
                    "probe failed). This command does not train",
                )
            )
        else:
            out.extend(
                _field_lines(
                    "estimate",
                    f"~{_format_seconds(work.seconds)} on this machine for one pair "
                    f"at the default operating point (dummy-batch probe; not a "
                    f"measurement of the recordings)",
                )
            )
        if decision.status is DecisionStatus.REFUSED and not decision.forced:
            out.extend(_field_lines("started", "no"))
        elif decision.status is DecisionStatus.WOULD_MEASURE:
            out.extend(
                _field_lines(
                    "started",
                    "no. This version of the command stops before training",
                )
            )
        else:
            out.extend(_field_lines("started", "no"))
    out.append(rule)
    # Audit findings this command quotes still carry U+00A7. Translate them
    # so the pasted block stays ASCII, which is the contract inherited from
    # `audit` even when a finding text has not been converted yet.
    return "\n".join(out).replace("\u00a7", "section ")


def render_json(decision: Decision) -> str:
    """Machine-readable form of the same fields."""
    report = decision.audit
    work = decision.work
    payload: dict[str, Any] = {
        "tool": "iqforge",
        "command": "measure-leakage",
        "tool_version": __version__,
        "status": decision.status.value,
        "category": int(decision.category) if decision.category is not None else None,
        "name": decision.name,
        "cites": decision.cites,
        "reason": decision.reason,
        "forced": decision.forced,
        "forced_past": decision.forced_past,
        "work": None
        if work is None
        else {
            "train_windows": work.train_windows,
            "test_windows": work.test_windows,
            "total_windows": work.total_windows,
            "arms": work.arms,
            "epochs": work.epochs,
            "seconds": work.seconds,
        },
        "audit": None
        if report is None
        else {
            "fingerprint": report.fingerprint,
            "verdict": report.verdict,
            "mode": report.mode,
            "input": report.input_path,
            "summary": {
                "leak": report.summary["leaks"],
                "pass": report.summary["passed"],
                "risk": report.summary["risk"],
                "not_checked": report.summary["not checked"],
            },
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)
