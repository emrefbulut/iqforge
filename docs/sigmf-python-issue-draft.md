# Upstream report — sigmf/sigmf-python#159

Filed as sigmf/sigmf-python#159 on 10.08.2026:
https://github.com/sigmf/sigmf-python/issues/159

Kept as the local record of what was reported. The text below is what was
submitted; the live discussion is upstream, so edit the issue there rather than
this file.

## Outcome

**Accepted, fixed upstream, awaiting release.**

- The report was discussed at SigMF's monthly call and turned into code within
  **seven days**.
- **Both** suggested fixes below were taken, not just the preferred one:
  [sigmf-python#160](https://github.com/sigmf/sigmf-python/pull/160)
  deep-copies the metadata in `__init__` *and* preserves the declared value in
  `__original_version`. It also removes the errant `self.version` attribute in
  favour of reading it from the metadata, and closes #159 explicitly.
- Two maintainers approved it. It targets **v1.13.0** and is not merged as of
  this writing.

**The bug is still present in the latest release.** `sigmf 1.12.0` shipped after
the report and still mutates the caller's dict — verified, not assumed, by the
tripwire in `tests/test_io.py`, which fired on the version bump and was
rechecked by hand.

**How long the workaround is needed.** Until `1.13.0` is released *and* the
floor in `pyproject.toml` is raised past it — and even then removing it is
optional rather than forced. `load()` reads `core:version` out of the parsed
JSON before the dict is handed to `SigMFFile`, which is correct whether or not
the library would have overwritten it, so the fix landing makes the workaround
*redundant* rather than *wrong*. `tests/test_io.py` distinguishes the two cases
and will say which one has happened.

One thing measured while preparing for it, worth knowing before deleting
anything: a deepcopy **alone** does not change what `get_global_info()` returns
— the library still normalises `core:version` inside its own copy. It is
`__original_version`, the second half of #160, that would change what a reader
sees. So `iqforge info` may keep showing both values after 1.13.0, and that is
correct rather than a leftover.

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
- Re-verified unchanged on sigmf 1.12.0

---

Worked around downstream by reading `core:version` out of the parsed JSON before
the dict is handed to `SigMFFile`. Both suggested fixes were accepted upstream
in #160; see **Outcome** at the top.
