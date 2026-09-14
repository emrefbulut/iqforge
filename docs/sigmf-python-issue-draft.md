# Upstream report — sigmf/sigmf-python#159

Filed as sigmf/sigmf-python#159 on 10.08.2026:
https://github.com/sigmf/sigmf-python/issues/159

Kept as the local record of what was reported. The text below is what was
submitted; the live discussion is upstream, so edit the issue there rather than
this file.

## Outcome

**Accepted, fixed upstream, released in sigmf 1.13.0.**

- The report was discussed at SigMF's monthly call and turned into code within
  **seven days**.
- **Both** suggested fixes below were taken, not just the preferred one:
  [sigmf-python#160](https://github.com/sigmf/sigmf-python/pull/160)
  deep-copies the metadata in `__init__` *and* preserves the declared value.
  It also removes the errant `self.version` attribute in favour of reading it
  from the metadata, and closes #159 explicitly.
- One maintainer approved it — `777arc`, 2026-08-17T21:52:30Z, nine minutes
  after the pull request was opened — and it shipped in **v1.13.0**. A second
  maintainer, `Teque5`, drove the discussion and wrote the PR; that is
  participation rather than a second review, and this line previously said
  "two maintainers approved" on the strength of it.

**Verified on the released version, not on the pull request.** CI resolves
dependencies fresh (`uv.lock` is not committed), so 1.13.0 reached CI before it
reached any developer machine and the tripwire in `tests/test_io.py` fired the
morning after publication — with the "behaviour CHANGED" message rather than
the routine one, which is the distinction that test exists to make. What was
then measured against an installed 1.13.0:

| | 1.12.0 | 1.13.0 |
|---|---|---|
| caller's dict after `SigMFFile(metadata=d)` | rewritten to `1.2.6` | **untouched, still `1.0.0`** |
| declared value recoverable from the handle | no | **yes, `handle.declared_version`** |
| `handle.get_global_info()["core:version"]` | `1.2.6` | `1.2.6` (unchanged) |

One detail worth recording, because it is the half of the fix that did *not*
change what a reader sees:

`get_global_info()` **still returns the library's spec version**. The deepcopy
moved the normalisation into the handle's own copy rather than removing it,
which is Option B in the maintainer's own framing. This was predicted from a
stand-in before 1.13.0 existed and is now confirmed against the real thing — so
`iqforge info` continues to print `1.0.0 (file); 1.2.6 (reader)`, and that
display is correct rather than a leftover.

**Where the name came from.** The maintainer's 2026-08-17 comment closed with
three candidates — `.declared_version`, `.file_version`, `.original_version` —
and asked which. The reply from here on 2026-08-19 was that "`declared_version`
reads better to me than `original_version`, but that's a detail". The pull
request had already used `self._declared_version` internally since it was
opened on 2026-08-17, and the released property is public `declared_version`.

So the shortlist was upstream's and the pick was endorsed from here; neither
side invented it alone. An earlier version of this section claimed the pull
request had called the attribute `__original_version` and that reading the diff
would therefore have given the wrong name. That was wrong on both counts:
`__original_version` appears nowhere upstream in that spelling, and the PR body
named `_declared_version` from the day it was opened.

**Checked against the three real captures this report cites.** Their metadata
was re-fetched and read under 1.13.0; all three declare `core:version: 1.0.0`,
and on all three the caller's dict survives intact, `declared_version` returns
`1.0.0`, and the handle still reports `1.2.6`. Nothing regressed:

| capture | datatype | in file | after `SigMFFile` | `declared_version` | `get_global_info()` |
|---|---|---|---|---|---|
| `cellular_downlink_880MHz` | `ci16_le` | 1.0.0 | 1.0.0 | 1.0.0 | 1.2.6 |
| `space/GNSS L1 E1 band recording` | `ci8` | 1.0.0 | 1.0.0 | 1.0.0 | 1.2.6 |
| `estevez/Vega-C MEO Cubesats/ASTROBIO_2022-07-24T19_25_49` | `ci16_le` | 1.0.0 | 1.0.0 | 1.0.0 | 1.2.6 |

**How long the workaround is needed.** Not needed on 1.13.0, and kept anyway.
`pyproject.toml` still admits 1.11.1 and 1.12.0, which do mutate, and the floor
was deliberately not raised: `load()` reads `core:version` out of the parsed
JSON before the dict is handed to `SigMFFile`, which is correct whether or not
the library would have overwritten it, so the fix makes the workaround
*redundant* rather than *wrong*. Forcing an upgrade on users for a bug this
package does not expose them to would be a cost with no benefit.

Whether `declared_version` opens a cleaner route once the floor does move is
tracked separately rather than decided here.

---

**Title:** `SigMFFile(metadata=...)` mutates the caller's dict and overwrites the declared `core:version`

**Labels:** bug

---

### Summary

`SigMFFile(metadata=d)` modifies `d` in place. Among the changes it overwrites
`core:version` — the version the recording declares — with the spec version the
installed library implements. After construction there is no way to recover what
the file actually said, from either the dict or the handle.

This is specific to the `metadata=` path, which reads an *existing* recording.
The `global_info=` path, which creates a *new* one, is not affected and should
not change; see "Relation to #76" below.

### Reproduction

```python
import json
from sigmf import SigMFFile

meta = {
    "global": {"core:datatype": "cf32_le", "core:version": "1.0.0"},
    "captures": [],
    "annotations": [],
}
before = json.dumps(meta["global"])
handle = SigMFFile(metadata=meta)

print("before:", before)
print("after :", json.dumps(meta["global"]))
print("get_global_info():", handle.get_global_info()["core:version"])
```

Output with sigmf 1.11.1:

```
before: {"core:datatype": "cf32_le", "core:version": "1.0.0"}
after : {"core:datatype": "cf32_le", "core:version": "1.2.6", "core:num_channels": 1, "core:offset": 0}
get_global_info(): 1.2.6
```

### Expected

Constructing a `SigMFFile` from a metadata dict should not modify the caller's
object. Defaults such as `core:num_channels` and `core:offset` being filled in is
reasonable and useful — doing it to the caller's dict is the surprising part.

`core:version` is different in kind from the other two: it is a value the file
asserts about itself, not a default to be supplied. Replacing it discards
information that cannot be recovered afterwards.

### Actual

The caller's dict is rewritten, and the declared version is lost.

### Why it matters

`core:version` is what you reach for when a recording does not parse the way you
expect. Tools that read it back after construction — the obvious thing to do —
report the reader's version for every file, so every recording looks like it was
written against whatever the local library implements.

This is not hypothetical. `cellular_downlink_880MHz`, a public 40 MS/s USRP B210
capture in the GNU Radio SigMF collection, declares `"core:version": "1.0.0"` and
is reported as `1.2.6`:

```
https://www.iqengine.org/api/datasources/local/local/cellular_downlink_880MHz.sigmf-meta
```

(browsable at https://www.iqengine.org). Two others from the same collection —
`space/GNSS L1 E1 band recording` and
`estevez/Vega-C MEO Cubesats/ASTROBIO_2022-07-24T19_25_49` — behave identically.

The in-place mutation is the wider problem. A caller that parses the JSON itself,
hands the dict to `SigMFFile` for validation, and then keeps using its own dict —
which is not an unusual pattern — silently gets different data back than it
parsed.

### Relation to #76

[#76](https://github.com/sigmf/sigmf-python/issues/76) asked for the opposite and
was right to: when building a new file from `global_info=`, the library should
fill in `core:version` and `core:num_channels` rather than leaving them out. That
behaviour is correct and this report does not ask for it to change.

The two paths differ in what the version means. Creating a file, there is no
declared version and supplying the current one is the only sensible choice.
Reading a file, the version is data — the writer's statement about which
revision of the spec the recording follows — and overwriting it destroys the
only record of that.

```text
SigMFFile(global_info={...})   # no version present -> filling it in is correct
SigMFFile(metadata={...})      # version present    -> overwriting it loses data
```

### Suggested fix

Either of, in order of preference:

1. Deep-copy the metadata on construction, so the caller's dict is untouched, and
   leave an existing `core:version` as the file declared it.
   `SigMFFile(metadata=copy.deepcopy(meta))` already avoids the mutation today,
   which suggests the copy belongs inside the constructor.
2. Keep the current behaviour but expose the declared value separately, so the
   information is at least recoverable.

### Environment

- sigmf 1.11.1 (latest on PyPI at the time of writing)
- Python 3.11 and 3.12
- Reproduced on Windows 11 and on Linux (ubuntu-latest, GitHub Actions)
- Re-verified unchanged on sigmf 1.12.0; fixed in sigmf 1.13.0

---

Worked around downstream by reading `core:version` out of the parsed JSON before
the dict is handed to `SigMFFile`. Both suggested fixes were accepted upstream
in #160; see **Outcome** at the top.
