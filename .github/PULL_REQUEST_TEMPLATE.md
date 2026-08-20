## What changed

<!-- One or two sentences, in terms of behaviour rather than files. -->

## Why

<!-- What was wrong, or what this makes possible. If it corrects something
     that already shipped, say what the wrong output looked like -- the
     correction is more useful than the fix. -->

## How it was verified

<!-- The commands you ran and what you saw. For a new test, say how you
     confirmed it can fail: the mutation you made and the failure it
     produced. A test that has never been red has not been checked. -->

```bash
# exact commands, and the result
```

## Conventions it touches

<!-- The numbered conventions in CONTRIBUTING.md. Tick the ones this change
     is answerable to and say in a line how it satisfies them; delete the
     rest. If none apply, say so -- an empty list is a claim too. -->

- [ ] 1 Never fall back silently
- [ ] 2 A passing test does not prove it can fail
- [ ] 3 Do not claim what you did not measure
- [ ] 4 Protect published artifacts (no file under `artifacts/` loses runs)
- [ ] 5 Do not enlarge a sample after seeing the result
- [ ] 6 Report your own mistakes
- [ ] 7 Ask before changing scope
- [ ] 8 Nothing session- or machine-specific is committed
