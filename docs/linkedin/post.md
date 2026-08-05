# LinkedIn post — iqforge v0.1.0 alpha

Copy the text below. Attach `docs/linkedin/linkedin.png` (or export from `linkedin.svg`).

---

**Post text (recommended):**

I built **iqforge** — a CLI that turns real SDR captures (SigMF) into PyTorch-ready datasets you can actually trust.

The gap in RF/ML today:
→ TorchSig generates synthetic data
→ IQEngine inspects recordings in the browser
→ But the step between "I have a capture" and "I can train on it" is where things quietly go wrong

Window your signal randomly and neighbouring windows land in both train and test. Accuracy looks great. The model gets worse. Nothing warns you.

**iqforge** makes the dangerous decisions explicit:
• Recording-level train/val/test splits (no window leakage)
• Refuses to run instead of silently falling back to a broken split
• `--balance-by` to spread nuisance variables (carrier freq, hardware) across splits
• SigMF-native: cf32_le, ci16_le, ci8 with memory-mapped I/O

```bash
iqforge info    capture.sigmf-meta
iqforge inspect capture.sigmf-meta
iqforge build   recordings/ -o dataset/ --balance-by core:freq_lower_edge
iqforge stats   dataset/
iqforge train   dataset/ --epochs 20
```

Ships with 16 example recordings — no hardware needed to try it.

Open source (MIT): https://github.com/emrefbulut/iqforge

#SoftwareDefinedRadio #MachineLearning #PyTorch #SignalProcessing #OpenSource #RF #SigMF #Python

---

**Short variant (if character limit matters):**

New project: **iqforge** — turn SigMF SDR captures into leak-safe PyTorch datasets.

Recording-level splits. No silent window leakage. Errors instead of inflated accuracy.

Open source: https://github.com/emrefbulut/iqforge

#SDR #MachineLearning #PyTorch #OpenSource

---

**First comment (optional, paste after posting):**

Clone and run in ~2 minutes:

```
git clone https://github.com/emrefbulut/iqforge
cd iqforge
uv sync --extra torch
uv run iqforge build examples/ -o dataset/ --balance-by core:freq_lower_edge
```

PyPI coming soon — for now install from source.
