# LinkedIn post — iqforge

## Images

| File | Size | Shows |
|---|---|---|
| `linkedin.png` | 1200×627 | Spectrogram fracturing into windows. Used for the first post. |
| `split_card.png` | 1200×1200 | Random window split vs recording-level split, side by side. |

`split_card.png` is the one to use for a post about the split guarantee: it draws
the claim rather than illustrating the domain. A reader who never opens the repo
still leaves knowing what the tool does differently.

Square beats 1200×627 in the feed — LinkedIn gives a square image roughly twice
the height on mobile, where most of the scrolling happens. The 627-tall version
is sized for link previews, not for an attached image.

Regenerate with:

```bash
uv run python docs/linkedin/make_linkedin.py     # linkedin.png
uv run python docs/linkedin/make_split_card.py   # split_card.png
```

---

## As published (2026-08-05)

Attached `linkedin.png`.

> I am releasing **iqforge**, an open-source CLI that converts SigMF SDR recordings into labelled PyTorch datasets.
>
> Most RF/ML tooling either generates synthetic data or visualises captures. The step from a real recording to a trustworthy train/val/test set is still easy to get wrong: window-level splits leak neighbouring samples across splits and inflate reported accuracy without warning.
>
> iqforge addresses this directly. Splits are performed at the **recording level**, not the window level. When a valid stratified split is impossible, the tool fails with an explicit error rather than falling back silently. Optional `--balance-by` spreads nuisance variables such as carrier frequency across splits while preserving class stratification.
>
> Built on SigMF (`cf32_le`, `ci16_le`, `ci8`), with memory-mapped I/O, terminal inspection, and an optional PyTorch Dataset interface.
>
> Repository: https://github.com/emrefbulut/iqforge
>
> #SoftwareDefinedRadio #MachineLearning #PyTorch #SignalProcessing #SigMF #OpenSource

---

## Revision for the next post

Three changes, in order of how much they matter.

**Lead with the problem, not the announcement.** Only the first two lines survive
above the "…see more" fold. "I am releasing X" spends them on the least
interesting sentence in the post. The leak is the hook — it makes a reader
recognise their own pipeline and open the rest.

**Move the repository link to the first comment.** LinkedIn suppresses reach on
posts with outbound links in the body.

**Fewer hashtags.** Three or four beat six. `#SoftwareDefinedRadio` and `#SigMF`
reach the people who care; `#OpenSource` and `#MachineLearning` are too broad to
do anything but dilute.

Attach `split_card.png`.

> Split your IQ windows at random and neighbouring windows land in both train and test. Your reported accuracy goes up. Your model gets worse. Nothing warns you.
>
> This is the step between "I have an SDR recording" and "I have a dataset", and it is easy to get wrong quietly.
>
> I have been building **iqforge**, an open-source CLI that turns SigMF captures into PyTorch datasets. It splits at the **recording** level, stratified by class — never the window level. When a valid split is not possible it stops with an error and tells you your options, instead of falling back to something that would inflate the number.
>
> `--balance-by <sigmf-field>` goes further: it spreads a nuisance variable — carrier frequency, receiver hardware, capture date — across the splits, so the model is not evaluated on a condition it never trained on.
>
> Reads SigMF natively (`cf32_le`, `ci16_le`, `ci8`), memory-maps large files, inspects captures in the terminal, and hands you a `torch.utils.data.Dataset`. Example recordings ship with the repo, so it runs without hardware.
>
> Link in the comments.
>
> #SoftwareDefinedRadio #SigMF #RFMachineLearning

**First comment:**

> https://github.com/emrefbulut/iqforge
>
> ```
> git clone https://github.com/emrefbulut/iqforge
> cd iqforge && uv sync --extra torch
> uv run iqforge build examples/ -o dataset/ --balance-by core:freq_lower_edge
> ```
