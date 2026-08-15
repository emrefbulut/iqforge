"""Resolving which recordings must not be split apart (`--group-by`).

The counterpart to `--balance-by`. Balancing spreads a nuisance variable ACROSS
splits; grouping keeps related recordings TOGETHER in one split, because they
are not independent of each other and separating them leaks.

Three schemes, and the first two exist because of what real datasets turned out
to look like rather than what would be tidy. In every public set examined for
`docs/methodology.md` the information identifying an acquisition lived in the
file path, not in the SigMF metadata: DASH7 puts the location in a directory and
the channel in the file name while every file declares the same centre
frequency; AirID encodes the burst in the file name; the Vega-C recordings carry
their session as a timestamp.

`collection:` is the third, and it reads the one field SigMF already has for
this: `core:collection` in the Global Object, which names a `.sigmf-collection`
file listing member recordings. Its own specification example is "channels from
a phased array" -- simultaneous multi-channel acquisition, exactly the case that
recording-level splitting cannot see.

**It is a hint, not a proof, and the code must not pretend otherwise.** A
Collection asserts that recordings are *related*; it does not assert that they
are statistically dependent. To this module, a collection holding "every
recording in my paper" and one holding "the four simultaneous receptions of one
frame" are the same object, and only the second is a constraint. Grouping on it
is therefore safe -- holding independent recordings together costs split
flexibility, never correctness -- but a passing check on it is never evidence
that a dataset is free of acquisition leakage. `iqforge audit` reports it as
`PASS/sample` at best for that reason.

A second limit worth knowing before relying on it: `core:collection` is a single
string, so a recording belongs to at most one collection. A dataset needing
nested levels -- this transmission, within this session, within this deployment
-- can express exactly one of them.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from iqforge.io import IQForgeError

#: Accepted `--group-by` prefixes.
GROUP_SCHEMES = ("path", "csv", "collection")

#: Schemes that take no argument -- the key is read from the recordings.
ARGUMENTLESS_SCHEMES = ("collection",)

#: Group key given to a recording the spec did not match. Kept distinct per
#: recording so an unmatched recording is its own unit rather than being pooled
#: with every other unmatched one, which would silently glue unrelated
#: recordings together -- the exact failure grouping exists to prevent.
UNMATCHED_PREFIX = "(ungrouped) "


def _parse_spec(spec: str) -> tuple[str, str]:
    """Split `scheme:argument`, or explain what the accepted forms are."""
    scheme, _, argument = spec.partition(":")
    if scheme in ARGUMENTLESS_SCHEMES and not argument:
        return scheme, ""
    if not argument or scheme not in GROUP_SCHEMES:
        raise IQForgeError(
            f"--group-by must be '<scheme>:<argument>' with scheme one of "
            f"{', '.join(GROUP_SCHEMES)}; got '{spec}'.\n"
            "  path:<regex>   group by a pattern over the recording's relative path,\n"
            "                 e.g. --group-by 'path:(CH\\d+)'\n"
            "  csv:<file>     group by an explicit table with 'recording,group' columns"
        )
    return scheme, argument


def _from_path(record_ids: list[str], pattern: str) -> dict[str, str]:
    """Group by a regex over the recording id (its path relative to the input).

    The key is the concatenation of the capture groups, or the whole match when
    the pattern has none. `re.search` rather than `fullmatch`: the pattern names
    the part that identifies the acquisition, not the whole path.
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise IQForgeError(f"--group-by path pattern is not a valid regex: {exc}") from exc

    keys: dict[str, str] = {}
    for record_id in record_ids:
        match = regex.search(record_id)
        if match is None:
            keys[record_id] = f"{UNMATCHED_PREFIX}{record_id}"
        elif match.groups():
            keys[record_id] = "|".join(g or "" for g in match.groups())
        else:
            keys[record_id] = match.group(0)
    return keys


def _from_csv(record_ids: list[str], path_text: str) -> dict[str, str]:
    """Group by an explicit `recording,group` table.

    A row is matched first on its value as written -- normally the path relative
    to the input directory -- and only then on the bare file name, which is
    offered so a flat layout need not spell out directories. A bare name that
    two rows disagree on is not used at all: on a public LoRa set where
    `3.sigmf-meta` exists under every session and receiver, name-only matching
    collapsed 23 554 groups into a handful and would have grouped unrelated
    recordings together while reporting nothing.
    """
    path = Path(path_text)
    if not path.exists():
        raise IQForgeError(f"--group-by csv file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"recording", "group"} <= fields:
            raise IQForgeError(
                f"'{path.name}' must have 'recording' and 'group' columns. "
                f"Columns found: {', '.join(sorted(fields)) or '(none)'}."
            )
        rows = [
            (row["recording"], row["group"])
            for row in reader
            if row.get("recording") and row.get("group")
        ]

    table: dict[str, str] = {}
    by_name: dict[str, set[str]] = {}
    for raw, group in rows:
        table[raw] = group
        table[Path(raw).as_posix()] = group
        by_name.setdefault(Path(raw).name, set()).add(group)
    ambiguous = set()
    for name, groups in by_name.items():
        if len(groups) == 1:
            table.setdefault(name, next(iter(groups)))
            table.setdefault(Path(name).stem, next(iter(groups)))
        else:
            ambiguous.add(name)

    # An ambiguous bare name only matters for a recording that has to fall back
    # to it. A table written as relative paths resolves every record exactly and
    # never reaches the fallback, so rejecting it up front would refuse the one
    # form that is unambiguous.
    keys: dict[str, str] = {}
    unresolvable: list[str] = []
    for record_id in record_ids:
        exact = table.get(record_id) or table.get(Path(record_id).as_posix())
        if exact:
            keys[record_id] = exact
            continue
        name = Path(record_id).name
        if name in ambiguous:
            unresolvable.append(record_id)
            continue
        value = table.get(name) or table.get(Path(record_id).stem)
        keys[record_id] = value if value else f"{UNMATCHED_PREFIX}{record_id}"

    if unresolvable:
        listed = ", ".join(sorted(unresolvable)[:3]) + (" ..." if len(unresolvable) > 3 else "")
        raise IQForgeError(
            f"{len(unresolvable)} recording(s) match no row of '{path.name}' by path, and "
            f"their file names appear under more than one group ({listed}). The bare name "
            f"does not identify them. Write the 'recording' column as the path relative to "
            f"the input directory, e.g. 'session/rrh1/3.sigmf-meta'."
        )
    return keys


def resolve_group_keys(
    record_ids: list[str],
    spec: str,
    *,
    collections: dict[str, str | None] | None = None,
) -> dict[str, str]:
    """Map each recording id to the key of the unit it belongs to.

    Args:
        record_ids: Recording ids, as they appear in the manifest.
        spec: The `--group-by` argument: `path:<regex>`, `csv:<file>` or
            `collection`.
        collections: Recording id -> its `core:collection` value, required by
            the `collection` scheme and ignored by the others. The caller reads
            it, because this module does not open recordings.

    Returns:
        Recording id -> group key. Recordings the spec did not match get a key
        of their own, so they stay independent units.

    Raises:
        IQForgeError: If the spec is malformed, the regex invalid, the CSV
            missing or short of columns, or `collection` used without the map.
    """
    scheme, argument = _parse_spec(spec)
    if scheme == "path":
        return _from_path(record_ids, argument)
    if scheme == "collection":
        return _from_collection(record_ids, collections)
    return _from_csv(record_ids, argument)


def _from_collection(
    record_ids: list[str], collections: dict[str, str | None] | None
) -> dict[str, str]:
    """Group by `core:collection`, the SigMF field meant for exactly this."""
    if collections is None:
        raise IQForgeError(
            "--group-by collection needs the recordings' metadata, which was not available here."
        )
    return {
        rid: (collections.get(rid) or f"{UNMATCHED_PREFIX}{rid}").strip()
        or f"{UNMATCHED_PREFIX}{rid}"
        for rid in record_ids
    }


def grouping_warnings(keys: dict[str, str], spec: str) -> list[str]:
    """Report a grouping that did nothing, or that missed some recordings.

    A spec that matches nothing leaves every recording its own unit, which is
    the same as not passing the flag at all -- and the user would be left
    believing recordings are being held together when they are not. Silent
    ineffectiveness is the failure mode this project keeps running into, so it
    gets a warning rather than a quiet no-op.
    """
    if not keys:
        return []
    unmatched = [r for r, k in keys.items() if k.startswith(UNMATCHED_PREFIX)]
    distinct = len(set(keys.values()))
    warnings: list[str] = []

    if len(unmatched) == len(keys):
        warnings.append(
            f"--group-by '{spec}' matched no recording; every recording is its own unit, "
            f"which is the same as not grouping at all."
        )
    elif unmatched:
        warnings.append(
            f"--group-by '{spec}' did not match {len(unmatched)} of {len(keys)} recording(s); "
            f"each of those stays its own unit: {', '.join(sorted(unmatched)[:3])}"
            f"{' …' if len(unmatched) > 3 else ''}"
        )
    elif distinct == len(keys):
        warnings.append(
            f"--group-by '{spec}' produced one group per recording, so nothing is being held "
            f"together. Check the pattern if you expected recordings to share a unit."
        )
    return warnings
