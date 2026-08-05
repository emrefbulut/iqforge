# LinkedIn post — iqforge

Attach `docs/linkedin/linkedin.png`.

---

**Post text:**

I am releasing **iqforge**, an open-source CLI that converts SigMF SDR recordings into labelled PyTorch datasets.

Most RF/ML tooling either generates synthetic data or visualises captures. The step from a real recording to a trustworthy train/val/test set is still easy to get wrong: window-level splits leak neighbouring samples across splits and inflate reported accuracy without warning.

iqforge addresses this directly. Splits are performed at the **recording level**, not the window level. When a valid stratified split is impossible, the tool fails with an explicit error rather than falling back silently. Optional `--balance-by` spreads nuisance variables such as carrier frequency across splits while preserving class stratification.

Built on SigMF (`cf32_le`, `ci16_le`, `ci8`), with memory-mapped I/O, terminal inspection, and an optional PyTorch Dataset interface.

Repository: https://github.com/emrefbulut/iqforge

#SoftwareDefinedRadio #MachineLearning #PyTorch #SignalProcessing #SigMF #OpenSource

---

**First comment (optional):**

```
git clone https://github.com/emrefbulut/iqforge
cd iqforge && uv sync --extra torch
uv run iqforge build examples/ -o dataset/ --balance-by core:freq_lower_edge
```
