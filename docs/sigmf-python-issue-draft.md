# Issue draft for sigmf-python — NOT FILED

Draft only. Review before opening at https://github.com/sigmf/sigmf-python/issues.

---

**Title:** `SigMFFile(metadata=...)` mutates the caller's dict and overwrites the declared `core:version`

**Labels:** bug

---

### Summary

`SigMFFile(metadata=d)` modifies `d` in place. Among the changes it overwrites
`core:version` — the version the recording declares — with the spec version the
installed library implements. After construction there is no way to recover what
the file actually said, from either the dict or the handle.

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

Concretely: three public captures from the GNU Radio / IQEngine repository all
declare `"core:version": "1.0.0"` and are reported as `1.2.6`.

The in-place mutation is the wider problem. A caller that parses the JSON itself,
hands the dict to `SigMFFile` for validation, and then keeps using its own dict —
which is not an unusual pattern — silently gets different data back than it
parsed.

### Suggested fix

Either of, in order of preference:

1. Deep-copy the metadata on construction, so the caller's dict is untouched.
   `SigMFFile(metadata=copy.deepcopy(meta))` already avoids the problem today,
   which suggests the copy belongs inside the constructor.
2. Keep filling in defaults, but leave `core:version` as the file declared it,
   and expose the implemented spec version separately (e.g. `sigmf.__specification__`).

### Environment

- sigmf 1.11.1
- Python 3.12
- Windows 11 (not platform specific)

---

## Workaround in iqforge

`iqforge.io.load` reads `core:version` from the parsed JSON *before* handing the
dict to `SigMFFile`, and `iqforge info` shows both values when they differ:

```
│ SigMF version │ 1.0.0 (file); 1.2.6 (reader) │
```

Covered by `tests/test_io.py::test_declared_version_survives_the_sigmf_library`,
which fails if the read is moved after the constructor.
