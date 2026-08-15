# Methodology

How the claims `iqforge` makes were measured, what the measurements returned, and
what they do not cover.

Everything below is reproducible from this repository. Where a number appears it
came from a run whose output is in [`artifacts/`](../artifacts/); where a
statement has no number behind it, it is described as a design decision rather
than a finding.

---

## 1. The problem

An IQ recording is cut into fixed-length windows. Those windows become the
examples in a dataset, and the dataset is split into train, validation and test.

The natural implementation splits the windows: shuffle them, deal them out in the
requested proportions. It is the same code you would write for images, and for
independent images it is correct.

IQ windows are not independent. `iqforge` defaults to a window of 1024 samples
and a stride of 512, so consecutive windows share half their samples. Window *k*
and window *k+1* are two views of one stretch of signal — the same symbols, the
same noise realisation, the same fading state, offset by half a window. A
window-level split puts one of them in train and can put the other in test.

The test set then contains material the model has already seen. Test accuracy
measures memorisation as much as generalisation, and it moves in the wrong
direction: it goes **up** as the leak gets worse, so nothing about the number
looks alarming.

`iqforge` splits at the level of the **recording** instead. Every window from one
recording goes to one split, and the split is stratified by class over recordings.
When the requested ratios cannot be satisfied that way, `build` stops with an
error rather than falling back.

Sections 2 and 3 measure what that refusal buys.

---

## 2. Measurement 1 — accuracy inflation against SNR

**Setup.** 48 synthetic recordings, two classes (BPSK, QPSK) matched in symbol
rate, occupied bandwidth, burst duration and mean burst power, so modulation is
the only class-carrying difference. Four carrier offsets, balanced across splits.
Split 0.6/0.2/0.2 at the recording level, giving 28/10/10 recordings. The same
window pool is then split two ways — once by recording, once by window with the
per-split window counts and class balance held identical — and both are trained
with the same model for the same 20 epochs. 15 seed pairs per SNR, 180 runs.

Noise is additive complex Gaussian. "Burst SNR" is the burst power against the
noise power in the full band, `10·log10(BURST_RMS² / 2σ²)`.

| burst SNR | recording-level | window-level | inflation (paired) | n |
|---|---|---|---|---|
| −4.1 dB | 51.4% ± 1.9% | 59.9% ± 7.7% | **+8.5 pp** ± 2.2 | 15 |
| −2.2 dB | 59.0% ± 9.1% | 72.5% ± 12.4% | **+13.6 pp** ± 3.7 | 15 |
| −0.8 dB | 74.9% ± 15.7% | 86.4% ± 8.1% | **+11.5 pp** ± 4.0 | 15 |
| +0.9 dB | 88.5% ± 9.3% | 95.7% ± 1.9% | **+7.2 pp** ± 2.3 | 15 |
| +3.0 dB | 96.0% ± 4.4% | 97.7% ± 1.6% | **+1.7 pp** ± 1.1 | 15 |
| +5.8 dB | 98.4% ± 1.4% | 98.9% ± 0.8% | **+0.5 pp** ± 0.3 | 15 |

Accuracy columns are mean ± standard deviation across runs. The inflation column
is the mean **paired** difference ± its standard error (§4).

**Shape.** The inflation is an inverted U. It is near zero at both ends and
largest in the middle, peaking at +13.6 pp around −2.2 dB.

Both ends are compressed, for different reasons:

- **At high SNR the honest split is already at the ceiling.** At +5.8 dB the
  recording-level arm scores 98.4%. There are 1.6 points available, so no split
  strategy can gain more than that, whatever it does.
- **At low SNR neither arm can do much.** At −4.1 dB the honest arm sits at
  51.4%, near chance for two balanced classes. The leak still helps — +8.5 pp —
  but the task is hard enough that a memorised test window is not worth as much,
  because there is less signal in it to memorise.

The peak is in between, where the model has enough signal to learn something but
not enough to solve the task outright. That is the regime where a few points
decide whether an approach looks viable, and it is where the number is least
trustworthy.

**Read the shape, not the peak.** +13.6 pp is a property of this signal pair,
this architecture and this window length, not a constant to be quoted. What
transfers is the shape: the inflation is largest exactly where the measurement
matters most, and it is smallest where a benchmark is easiest to run. Anyone who
evaluates a leaky split on easy data will conclude the concern is overblown, and
their measurement will honestly support that conclusion.

Reproduce: `uv run python scripts/leakage_experiment.py`. Raw runs in
[`artifacts/leakage_runs.json`](../artifacts/leakage_runs.json).

---

## 3. Measurement 2 — accuracy inflation against overlap

Section 2 shows *when* window-level splitting inflates accuracy. It does not show
*why*. If overlap is the mechanism, then removing the overlap should remove the
effect, and that is a prediction the experiment can be pointed at directly.

**Setup.** Identical to §2 except that the noise is fixed at σ = 0.17 (−0.8 dB
burst SNR, chosen because §2 found a large gap there, so there is room for the
stride to move the number) and the window is fixed at 1024 samples. Only the
stride varies. 15 seed pairs per stride, 150 runs.

| stride | overlap | recording-level | window-level | inflation (paired) | n |
|---|---|---|---|---|---|
| 1024 | 0% | 59.6% ± 7.2% | 59.8% ± 9.8% | **+0.2 pp** ± 2.7 | 15 |
| 768 | 25% | 71.8% ± 6.4% | 72.3% ± 11.4% | **+0.5 pp** ± 2.8 | 15 |
| 512 | 50% | 74.9% ± 15.7% | 86.4% ± 8.1% | **+11.5 pp** ± 4.0 | 15 |
| 256 | 75% | 71.3% ± 13.8% | 94.1% ± 10.0% | **+22.9 pp** ± 4.5 | 15 |
| 128 | 88% | 82.2% ± 12.1% | 95.6% ± 4.8% | **+13.4 pp** ± 3.1 | 15 |

**At zero overlap the inflation is +0.2 pp ± 2.7 — indistinguishable from
noise.** When windows share no samples, a window-level split separates genuinely
distinct material, and splitting by window rather than by recording costs
nothing measurable. This is the cleanest statement the experiment produces: the
leak is not a property of window-level splitting in the abstract, it is a
property of window-level splitting **applied to overlapping windows**.

At 25% overlap the gap is still within noise (+0.5 pp ± 2.8). From 50% it is
unmistakable. `iqforge`'s own default stride of 512 sits at 50% overlap, in the
row that costs 11 points.

**One caveat on reading down the column.** Shrinking the stride does not only
increase overlap — it also multiplies the number of windows. At stride 128 there
are eight times as many windows as at stride 1024, so the recording-level arm is
training on eight times more data and improves on its own (59.6% → 82.2%). The
smaller gap in the last row is partly that catching-up, not overlap mattering
less. The column is therefore not a clean dose-response curve for overlap alone.

Within each row the comparison is exact — same windows, same counts, same class
balance, only the assignment differs — so each row's inflation figure stands on
its own. The zero-overlap row is the one that carries the causal claim, and it
needs no cross-row comparison to do so.

Reproduce: `uv run python scripts/leakage_experiment.py --sweep stride`.

### The same sweep on a real capture

§6 explains why the SNR sweep of §2 cannot be reproduced on any of the public
datasets assessed. The stride sweep is a different question — it asks
what the mechanism is, not how large the effect gets — and it does not need a
graded task, so it was run on the DASH7 cabled set: 3 channels, 10 independent
recorder runs each, added noise fixed at −19 dB wideband, 15 seed pairs per
stride, 150 runs.

| stride | overlap | recording-level | window-level | inflation (paired) | n |
|---|---|---|---|---|---|
| 1024 | 0% | 59.7% ± 10.6% | 55.9% ± 12.8% | **−3.7 pp** ± 3.7 | 15 |
| 768 | 25% | 52.3% ± 10.4% | 48.9% ± 12.9% | **−3.4 pp** ± 4.4 | 15 |
| 512 | 50% | 48.4% ± 10.7% | 52.6% ± 16.0% | **+4.3 pp** ± 3.5 | 15 |
| 256 | 75% | 53.2% ± 14.6% | 61.9% ± 13.5% | **+8.7 pp** ± 5.9 | 15 |
| 128 | 88% | 53.1% ± 11.3% | 60.8% ± 17.2% | **+7.7 pp** ± 5.1 | 15 |

**The zero-overlap prediction holds.** At stride 1024 the inflation is −3.7 pp
± 3.7, indistinguishable from zero, exactly as on synthetic data. Nothing in
this dataset produces a leak when the windows share no samples.

**The positive half is consistent but not established.** Inflation rises with
overlap, and the trend the design licenses — a per-seed regression of inflation
on overlap fraction, paired so that seed scatter cancels — gives a slope of
**+15.8 pp per unit overlap, standard error 7.7, t = 2.07**. That is the
direction §3 predicts and a magnitude compatible with it, but no individual row
clears t = 2, and only 8 of 15 seed pairs show high overlap beating zero
overlap at all.

So this run **does not independently establish the effect on real data**. It
rules out the alternative that the synthetic result was an artefact of
synthetic signals, and it confirms the null where the causal claim lives. It
does not reproduce the effect size.

The reason is the one §6 documents, and it is a property of the dataset rather
than of the experiment. The honest arm here scores 48–60% on a task whose chance
line is 33.3%, with a standard deviation of 10–15 points between seeds — an
order of magnitude more scatter than the synthetic arm at comparable accuracy,
because this dataset's classes are separated by a 2.3 MHz carrier offset and the
model therefore either resolves them or does not. There is no stable region of
partial competence, so there is little statistical power to detect an effect
in. Measuring a 10-point effect through 15 points of seed noise needs more than
15 seed pairs, and the way to get it is a dataset with graded difficulty, not
more compute on this one.

**This is a limit, not a failed result.** The run answers the question it was
pointed at — is the zero-overlap null real outside synthetic data — and it
answers yes. What it cannot do is size the effect, and §6 explains in advance
why no run on this dataset could.

Reproduce: `uv run python scripts/leakage_real.py --sweep stride`.

### The same sweep on a dataset that can carry it

DASH7 confirmed the null and could not size the effect, because its task is a
step function (§6). LoRaIQ is the first assessed dataset `iqforge audit` returns
`unknown` for — no single measurable axis separates the classes — and it has the
region of partial competence the measurement needs: the honest arm sits around
65%, against a 46% chance line.

**Setup.** 312 recordings over 13 capture sessions, class = propagation
environment (`drone_los`, `drone_nlos`, `pedestrian_partial_los`,
`pedestrian_nlos`, `indoor`), split 0.6/0.2/0.2 and **grouped by transmission
id**, because one LoRa frame is heard by up to four rooftop receivers at the same
instant and those four files are one event. A fixed 15 244-sample segment is
taken around each frame so every recording contributes equally. No noise added.
15 seed pairs per stride, 150 runs.

| stride | overlap | recording-level | window-level | inflation (paired) | t | n |
|---|---|---|---|---|---|---|
| 1024 | 0% | 66.8% ± 6.7% | 68.3% ± 2.1% | **+1.5 pp** ± 1.7 | 0.9 | 15 |
| 768 | 25% | 67.8% ± 9.0% | 67.3% ± 2.8% | **−0.4 pp** ± 2.3 | −0.2 | 15 |
| 512 | 50% | 65.8% ± 9.0% | 69.4% ± 3.2% | **+3.7 pp** ± 2.3 | 1.6 | 15 |
| 256 | 75% | 69.2% ± 5.8% | 68.8% ± 4.8% | **−0.4 pp** ± 1.7 | −0.2 | 15 |
| 128 | 88% | 64.5% ± 10.0% | 74.1% ± 4.5% | **+9.6 pp** ± 2.7 | **3.5** | 15 |

**What this establishes.** Two things, and they are the two the experiment was
pointed at.

*The null holds a third time.* At zero overlap the inflation is +1.5 pp ± 1.7 —
indistinguishable from zero, as on synthetic data (+0.2 ± 2.7) and on DASH7
(−3.7 ± 3.7). Three independent datasets, three times, the row the causal claim
rests on.

*The effect is now individually significant on real data.* At 88% overlap,
+9.6 pp ± 2.7, **t = 3.5**. The DASH7 repeat could only report a trend at
t = 2.07 and had to be written up as consistent-with rather than established;
this one does not. Across all 75 seed pairs, a per-seed regression of inflation
on overlap gives **+6.2 pp per unit overlap, t = 2.49**, the 88%-versus-0%
contrast is **+8.1 pp ± 3.3 (t = 2.48)**, and 12 of 15 seeds show 88% beating 0%.

**What this does not establish: the shape of the dose-response curve.** The
intermediate rows do not line up — 25% and 75% both came out at −0.4 pp. Nor do
they line up on the other two datasets:

| overlap | synthetic | DASH7 | LoRaIQ |
|---|---|---|---|
| 0% | +0.2 | −3.7 | +1.5 |
| 25% | +0.5 | −3.4 | −0.4 |
| 50% | +11.5 | +4.3 | +3.7 |
| 75% | **+22.9** | **+8.7** | −0.4 |
| 88% | +13.4 | +7.7 | **+9.6** |

Synthetic peaks at 75% and falls; DASH7 peaks at 75% and falls; LoRaIQ is
non-monotone with its maximum at 88%. The per-pair spread says why n = 15 cannot
resolve this: at stride 256 the fifteen paired differences run from −11 to +14 pp,
at stride 128 from −6 to +38 pp. Only the largest effect clears that noise.

So the honest statement is narrower than a curve: **overlap is the mechanism —
remove it and the effect goes, increase it to 7/8 and the effect is large and
significant — but how the effect grows in between is not measured.**

**The stride column still must not be read downward.** Shrinking the stride
multiplies the window count (2 926 windows at stride 1024, 20 048 at stride 128
for one split seed), so the recording-level arm trains on more data as the rows
descend. Each row's comparison is exact — same windows, same counts, same class
balance, only the assignment differs — and each row's inflation figure stands on
its own. The column is not a dose-response curve and no row's number should be
subtracted from another's.

**This was a confirmatory test, not an exploratory one.** The direction was
predicted in advance: §3's synthetic result says inflation is zero at zero
overlap and rises with overlap, and this run was designed to check that
prediction on a dataset chosen before any of its numbers were seen. That is the
answer to the multiple-comparisons objection — five strides were measured
because the synthetic sweep measured five, not because five were tried until one
worked.

**The intermediate points were not re-run with more seeds**, although 45 minutes
of compute would have done it. Enlarging a sample after seeing the result means
letting the result choose the stopping rule, and a +6.2 pp trend that only
becomes significant once the analyst has decided to keep going is not a finding.
The rows stand at the n they were planned with.

**Measured on** CPU (`torch 2.13.0+cpu`, 8 threads, AMD64). No GPU was used, and
`iqforge train` pins the device on purpose (§8) — mixing devices inside one
paired experiment would break the "only the assignment differs" property that
the whole design rests on.

Reproduce: `uv run python scripts/leakage_loraiq.py`. Raw runs in
[`artifacts/leakage_loraiq_runs.json`](../artifacts/leakage_loraiq_runs.json).

---

## 4. Experimental design

**The wrong split lives outside the tool.** `iqforge build` refuses to split at
the window level. Demonstrating why that refusal is right requires doing the
wrong thing deliberately, so the leaky splitter lives in
[`scripts/leakage_experiment.py`](../scripts/leakage_experiment.py) and is never
reachable from the CLI. Users cannot produce a leaky dataset with this tool even
by accident.

**Pairing.** For one split seed and one training seed, both arms use the same
recordings, the same window pool, the same per-split window counts, the same
class balance and the same weight initialisation. Only the assignment of windows
to splits differs. The difference within such a pair isolates the effect.

This matters because seed-to-seed scatter dominates everything else here. At
−2.2 dB the two arms have per-arm standard deviations of 9.1 and 12.4 points
against an effect of 13.6. Comparing the two group means folds that scatter into
the comparison; pairing cancels it, because both members of a pair drew the same
recordings into train. The tables therefore report the mean paired difference and
its standard error, not the difference of the two means.

**Why n went from 6 to 15.** The first full grid used 3 split seeds × 2 training
seeds = 6 pairs per SNR. It produced a paired standard error near 6 pp against an
effect around 10 pp — the direction was clear, the magnitude was not. More
seriously, two of the six rows came out with the **wrong sign**: at +3.0 dB the
n=6 grid measured −1.7 pp ± 0.6, and at n=15 the same cell measures +1.7 pp ± 1.1.

That sign flip is the reason the grid grew rather than a curiosity about it. Six
pairs was not enough to distinguish a small positive effect from noise, and a
confidently reported negative number would have been worse than no number at all.
At 15 pairs (5 split seeds × 3 training seeds) the SNR curve has a single sign
throughout and both ends behave as the mechanism predicts.

**Preconditions are asserted, not assumed.** The recording-level arm is only a
fair baseline if it is not handicapped by something other than the split. An
early version of this experiment used ratios under which every carrier offset
landed in exactly one split, so the honest arm was tested on carriers it had
never trained on and sat at chance — a distribution shift, not leakage, and one
that would have swamped the effect. `iqforge` warned about it; the script had
captured the subprocess output and never looked at it.

The script now aborts if `build` emits any warning, and separately verifies from
the manifest that every carrier offset present in test also appears in train.
The second check is not redundant: `iqforge` warns when a split collapses to a
single group, but a *partially* confounded split passes that check while still
leaving some evaluation on unseen carriers.

---

## 5. Validation against real recordings

The synthetic experiments say nothing about whether the reader is correct. Three
public recordings from the GNU Radio SigMF collection (browsable at
[iqengine.org](https://www.iqengine.org)) were used to check that.

| | datatype | hardware | size |
|---|---|---|---|
| `space/GNSS L1 E1 band recording` | `ci8` | Ettus B210, 6 MS/s | 12 000 000 B |
| `estevez/Vega-C MEO Cubesats/ASTROBIO_2022-07-24T19_25_49` | `ci16_le` | USRP B205mini, 40 kS/s | 11 170 468 B |
| `cellular_downlink_880MHz` | `ci16_le` | USRP B210, 40 MS/s | 80 000 000 B |

**Integer conversion, checked against the bytes.** The `ci16_le` and `ci8`
conversion paths had until then only been exercised by synthetic round-trips —
tests that confirm the reader agrees with the writer that wrote the same
assumption. Reading the files independently with `numpy.fromfile` and comparing
sample by sample:

- `ci8`: raw `[-8, 3, 4, -1, …]` → `[-0.0625+0.0234375j, …]`, ratio exactly
  **128** on every non-zero sample.
- `ci16_le`: ratio exactly **32768**.

Both are full-scale normalisation, matching the divisors declared in `io.py`.
Interleaving is correct in both: `raw[0]` becomes the real part, `raw[1]` the
imaginary part. Sample counts derived from file size match the declared datatype
in all three files.

**Frequency metadata, checked against the signal.** `cellular_downlink_880MHz`
carries 8 annotations with frequency edges. Averaging the spectrogram over time
and separating bins by whether any annotation covers them:

| | mean power |
|---|---|
| bins inside an annotated band | −46.15 dB |
| bins outside every annotation | −77.98 dB |
| **difference** | **+31.83 dB** |

Every gap between annotations is a genuine spectral null — −11.21…−10.59 MHz at
−78.2 dB, +2.27…+3.24 MHz at −78.1 dB, +13.55…+19.96 MHz at −79.0 dB. This is an
end-to-end check rather than a metadata check: if the integer scaling, the I/Q
order or the byte layout were wrong, the energy would not land where the
metadata says it should.

**What the annotation path does on a real recording.** Labelling
`cellular_downlink_880MHz` from its annotations yields **293 usable windows out
of 39 061 (0.75%)**: 38 085 unmatched and 683 dropped as ambiguous. Five of the
eight annotations share one time range and differ only in frequency, so every
window in that stretch falls inside more than one of them. Labelling is
time-based, so those windows cannot be assigned without guessing, and they are
dropped and counted instead. The one annotation that survives is the only one
that does not share its time range with another.

This is the designed behaviour meeting its limit on real data rather than a
defect, and it is recorded in the README's known limitations. Frequency-aware
labelling is the fix and is not implemented.

---

## 6. What it took to find a dataset that could carry the measurement

The obvious next step after §2 is to run the same comparison on a real capture.
It took **five public datasets** to find one that could carry it, and the fifth
did: the LoRaIQ result in §3 reproduces the zero-overlap null and reaches
t = 3.5 at high overlap. This section is the record of the four that could not,
because the reasons they failed are not accidents of this particular search and
they are more instructive than the success.

An earlier version of this section was titled "why the measurement was not
repeated on real data", which was accurate when it was written and is not now.
What survives the correction is the search cost: the criteria below are real,
they eliminated four datasets in a row, and only one of them can be checked
before downloading anything.

A leakage measurement needs real hardware, several **independent** recordings
per class — enough that a recording-level split has something to split — a
format the reader can be trusted on, and a task that is neither trivial nor
impossible. Format turned out to be the easy one.

**AirID** (GENESYS Lab, 4 UAV transmitters with deliberately distinct IQ
imbalances). Recording count is not the problem: the sibling *hovering-uavs*
release is 7 UAVs × 4 distances × 4 bursts × ~140 transmissions, ~13k files. The
format is. GENESYS's own converter writes `"core:datatype": "cf16_le"` and
`"core:version": "0.0.1"`. The SigMF schema constrains the datatype to
`^(c|r)(f32|f64|i32|i16|u32|u16|i8|u8)(_le|_be)?` — there is no `f16`, and
`0.0.1` is not a spec version either. `iqforge` rejects the file, correctly:

```
Unsupported datatype 'cf16_le'. Supported: cf32_le, ci16_le, ci8.
```

The class label sits in `core:device_id_genesys_lab`, a key invented for the
dataset, so annotation labelling finds nothing. No licence is stated. Each
obstacle is surmountable — convert to `cf32_le`, copy the label — but each
conversion is a step where the data can be altered between the publisher and the
measurement, which is precisely what the measurement is trying to be careful
about.

**Vega-C MEO Cubesats** (GNU Radio SigMF collection, 5 satellites, `ci16_le`,
CC BY 4.0). Format is ideal: `iqforge` reads it as shipped, and the byte-level
check in §5 was done on a recording from this collection. There are 3 recordings
per satellite — at the bottom of what a three-way split can use. The structure is
what rules it out: all five satellites carry the *same three* capture timestamps,
`2022-07-24T18_47_38`, `19_25_49` and `19_29_02`, with identical file sizes. The
sessions are crossed perfectly with class. Splitting by recording therefore puts
a different pass — different Doppler, elevation and SNR — in each split. That is
distribution shift, and it is the failure that invalidated the first version of
the SNR grid (§4). Measuring leakage on top of it would measure the sum of the
two.

**DASH7 `ds_indoor`** (Zenodo 10961311, `ci16_le`, CC BY 4.0, USRP B210). This
one passed the count and still failed, which is the instructive case.

Its published description gives 10 locations, "60 packets per location, 10
packets per file pair" and "3 Lo-Rate channel recordings per location" — 6 file
pairs per location. Six independent recordings per class clears any reasonable
threshold. The archive layout agrees: 10 nested zips, one per location, each
~1.39 GB, holding files of exactly 245,760,000 bytes — 8.0 s at 7.68 MS/s.

The filenames do not agree. The first two channel-0 recordings at location 1 are
timestamped `11.35.36` and `11.36.19`: **43 seconds apart**, 8 seconds long, in a
static indoor setup, same antenna, same position, same channel. They are separate
recorder invocations, so by any structural definition they are two independent
recordings. Physically they are near-duplicates — the multipath a room presents
does not change in 43 seconds. Put one in train and the other in test and the
model sees the same channel realisation on both sides.

Grouping those pairs, as physical independence requires, leaves 3 units per
location, which is the Vega-C situation again. And the three differ only by
channel, which is a carrier offset within one captured band: every file declares
`core:frequency: 866500000.0`, so the distinction that separates the three units
is **not in the metadata at all** and `--balance-by` cannot see it.

**DASH7 `ds_indoor_cabled`** (Zenodo 10961311, 1.9 GB, `ci16_le`, CC BY 4.0,
USRP B210). The cabled companion to the set above: same three Lo-Rate channels,
transmitter wired to receiver instead of over the air, and **ten** separate
recorder runs per channel instead of six. It clears every bar the other three
failed. `iqforge` reads it as shipped, ten independent recordings per class is
more than a three-way split needs, `--dirname-level 2` reads the channel out of
the `CH0/rec1/` layout, and `--group-by` now exists in case the runs turn out
not to be independent after all.

So this one was not assessed and set aside — it was downloaded, prepared and
trained on. That is what makes it the instructive case: it failed at the far
end, with the model already running.

Two properties of the data had to be handled before any number meant anything,
and both were measured rather than assumed:

- **The carrier is only on air 6.8% of the time.** Each recording holds ten
  packets of 207,872 samples (27.1 ms); the remaining 93% is noise floor, where
  no window carries class information at all. Packet timing differs between
  runs, so each recording's first packet is located by its power envelope rather
  than assumed to start at a fixed offset.
- **Noise has to be added before windowing, not per window.** Two overlapping
  windows must share the noise in the samples they share. Adding noise per
  window would give them independent draws in that shared region and destroy the
  correlation the experiment exists to measure — the result would come out clean
  for the wrong reason.

With that in place the recordings as captured were classified at **100.0% by
both arms**, and so was the same task with noise added at 0 dB wideband SNR.
Below is why, and it is arithmetic, not a surprise:

| quantity | measured |
|---|---|
| capture bandwidth | 7.68 MHz |
| occupied bandwidth per channel | 19.5 kHz |
| carrier separation between classes | 2.33 MHz (−3.485, −1.160, +1.166 MHz) |
| processing gain | **25.9 dB** |

The class *is* a carrier offset, the offsets are 2.3 MHz apart, and each signal
fills 19.5 kHz of a 7.68 MHz band. A first convolutional layer that learns
anything frequency-selective at all gets about 26 dB for free, so "0 dB
wideband" is roughly **+26 dB where the signal actually lives**. The SNR grid as
designed — 6, 3, 0, −3, −6 dB — would have spent an hour and a half returning
six rows of 100%.

Pushing down to find the usable band gave the rest of the answer:

| wideband SNR | in-band | recording-level accuracy |
|---|---|---|
| −15 dB | +11 dB | 96.4% |
| −22 dB | +4 dB | 40.8% |
| −28 dB | −2 dB | 33.4% |
| −40 dB | −14 dB | 33.3% |

Chance is 33.3%. The task goes from **solved to impossible inside about 7 dB**,
and the transition band is not merely narrow but unstable: at −19 dB two split
seeds of the same arm returned 37.4% and 79.7%.

That is what disqualifies the dataset for §2. The curve in §2 exists because
synthetic difficulty is *graded* — the model half-learns the task over a wide
band of SNR, and leakage is what fills the gap between half-learned and
reported. Here the task is effectively **binary**: the network either resolves a
2.3 MHz carrier separation or it does not. There is no wide region of partial
competence for a shortcut to exploit, so an accuracy-against-SNR curve measured
on this data would be reporting the width of a cliff, not the size of a leak.

The stride sweep of §3 asks a different question — what the mechanism is rather
than how large the effect gets — and it does not need graded difficulty. That
one *was* run on this dataset, and its result is in §3.

### What it cost to find out, and what that bought

Everything above was learned the expensive way. The processing-gain arithmetic
was done by hand after a pilot returned 100% on both arms; the 7 dB transition
was found by a bracket probe; the seed instability only became visible once the
probe was repeated. The SNR grid that was designed before any of that would have
run for 1.6 hours and returned six rows of 100%.

None of those findings needed a model. The carrier offsets are in the spectrum,
the occupied bandwidth is in the spectrum, and the ratio between them is the
processing gain. That is what `iqforge audit` was built to compute, and pointing
it at the same 30 recordings gives:

```
VERDICT       ceiling - carrier offset alone classifies 100% of recordings
              (chance 33%), so a trained model will saturate and leave no room
              to measure a leak
NEXT          do not run measure-leakage on this data. A model will score at
              the ceiling in both arms and the measured inflation will be zero
              for a reason that has nothing to do with splitting.
```

**Eight seconds, no training.** It also reports the occupied bandwidth it
measured — 30.0 kHz of 7.68 MHz, 24.1 dB of available processing gain — against
the 19.5 kHz and 25.9 dB arrived at by hand. The estimate is coarser, because it
takes the span holding 99% of the above-floor power rather than a hand-placed
band, and it is close enough to have made the same decision.

That is the empirical case for the command existing. It does not find leaks that
a careful person would miss; it finds, in seconds, the reasons a measurement was
never going to work, which is the part that otherwise costs hours.

### What this actually shows

The recurring obstacle was not licensing, size or format. It was that **none of
the first three datasets documents its recording structure** — which published files
came from one continuous capture, and which are genuinely separate acquisitions.

That had to be reconstructed indirectly in every case:

- reading the ZIP64 central directory over HTTP range requests, to see that
  `ds_indoor.zip` is ten nested per-location archives rather than a flat set;
- reading the GNU Radio flowgraphs that produced the data — a `blocks_head`
  limiting each run to a fixed duration, a `blocks_file_source` with
  `length: 61440000` cutting an 8-second strip out of a longer raw file, and a
  filename built from `datetime.now()` at record time, which is what makes the
  timestamps mean anything;
- arithmetic on those timestamps and on file sizes, to decide whether two files
  are two captures or two slices of one.

A leak-free split requires knowing which recordings are independent. Published RF
datasets state how many files they contain; they do not state which of those
files share an acquisition. The two are not the same number, and only the second
one is the one a split needs.

The DASH7 case takes it one step further. There the documented structure was
sufficient — six separate recorder runs per location, verifiable, not a matter of
interpretation. Physical independence still was not, and no amount of reading the
description would have revealed it; the 43-second gap only became visible after
recovering the filenames from inside a 13.9 GB archive. **Counting recordings is
not enough.** Independence is a property of how the data was acquired, and it
survives into the published artefact only if someone writes it down.

The cabled set adds a requirement that has nothing to do with provenance. It
documents its structure, it has ten independent runs per class, and it reads
without conversion — and it still cannot carry §2, because the *task* it poses
is a step function. A dataset is suitable for measuring leakage only if there is
a regime where the model is partly right; a task that is either trivial or
impossible has no room for a shortcut to show up in, whatever its file count. So
the search criteria are four, not three: real hardware, documented independence,
a readable format, and **a task whose difficulty is graded**. Only the first
three can be checked before downloading anything.

### A fifth dataset that changes the claim

**LoRaIQ** (Zenodo 20341802, CC BY 4.0, 71 GB, four rooftop receivers at EPFL,
30 000+ LoRa frames) is the case the paragraphs above did not anticipate. It
publishes exactly what they say nobody publishes.

Its `dataset.csv` carries, per frame, the columns `sigmf_file`,
`sigmf_file_offset` and `sigmf_file_n_samples` — which capture a frame came
from, where in it, and how long. That is acquisition provenance at sample
resolution, for 103 802 files, written down by the authors. The claim that
published RF datasets do not record which of their files share an acquisition
is, for this dataset, simply false.

So the thesis narrows rather than weakening. **The information sometimes
exists, and where the format can hold it, it cannot say what it means.**

SigMF is not silent here, and an earlier draft of this section overstated the
gap. `core:collection` in the Global Object names a Collection file, and that
file lists member recordings in `core:streams` — whose own example is "channels
from a phased array", which is simultaneous multi-channel acquisition, the
LoRaIQ case exactly. `core:offset` is documented as "typically used when a
Recording is split over multiple files". The `capture_details` extension carries
`source_file`, the recording a file was cut from. NTIA's `ntia-scos` carries
`schedule.id`, `task` and `recording`, which together identify an acquisition
run.

What none of them carries is the **constraint**. A Collection asserts that
recordings are *related*; it does not assert that they are statistically
dependent and must not be separated. To a tool, a collection of "every recording
in my paper" and a collection of "the four simultaneous receptions of one frame"
are the same object. The first must not constrain a split; the second must. The
format cannot tell them apart, so neither can a reader of the format.

Five limits, concretely, from reading the spec against what LoRaIQ needs:

1. **No dependence semantics.** As above: relatedness is expressible, and it is
   the wrong predicate. Grouping a split correctly requires knowing that members
   are not independent, which no field states.
2. **One collection per recording.** `core:collection` is a single string, so a
   recording belongs to at most one collection. LoRaIQ needs three nested
   levels — this transmission, within this session, within this deployment —
   and only one of them can be recorded.
3. **One group per collection file.** A Collection holds a single `core:streams`
   array. Expressing LoRaIQ's 23 554 transmission groups means 23 554
   `.sigmf-collection` files.
4. **Co-location is mandatory.** The spec requires a collection file to sit "in
   the same directory as the Recordings that it references, or in the top-level
   directory of an Archive", which a grouping that cuts across directories
   cannot satisfy without restructuring the tree.
5. **Membership is sealed with a hash.** A Recording Object "MUST contain both a
   `name` field ... and a `hash` which is the SHA512 hash of the Recording
   Metadata file". Correcting one label in one `.sigmf-meta` invalidates every
   collection that references it — so the grouping breaks whenever the metadata
   is fixed.

Points 3 to 5 are friction and could be lived with. Points 1 and 2 are the
substance: the format can say *these belong together* but not *these must not be
separated*, and it can say it only once per recording.

LoRaIQ's provenance therefore lives in a sidecar CSV under column names its
authors invented, and `iqforge` reads it because someone wrote a converter for
that one file, not because a tool can know where to look.

**The thesis narrows; it does not weaken.** The original claim here — that
there is no standard place to record which files share an acquisition — was too
strong, and surveying the spec is what corrected it. The accurate claim is
sharper and harder to dismiss: there *is* a place, and it carries no meaning a
splitter can act on. "Nobody records it" invites the reply that authors should
be more careful. "Authors do record it, and the standard field that could hold
it cannot distinguish a bibliography from a constraint" is a gap in the standard,
and it is the kind of gap an extension exists to close: a `core:` or
`recording:` namespace naming the acquisition a file belongs to would let the
DASH7 43-second pairs, the Vega-C shared passes and the LoRaIQ simultaneous
receptions all be stated in the same field instead of reconstructed from
timestamps, flowgraphs and filenames.

The dataset also demonstrates why the field would earn its place. A LoRa frame
is heard by up to four receivers at the same instant, so one transmission is
four files and one event: 23 742 of its transmissions have exactly four
receptions. Split those files independently and the same instant of radio lands
on both sides of the split — recording-level splitting does not help, because
the unit of independence is the transmission, not the file. Built from 312 of
its recordings without grouping, `iqforge audit` reports **271 of 465
air-time-sharing pairs split across sides**; built with `--group-by` on the
transmission id from that same CSV, it reports **all 465 held together**. The
constraint is expressible and checkable — but only because one dataset chose to
write it down, in a form only it uses.

### More data is not more independent units

Only 13 of LoRaIQ's 22 capture sessions were gathered at first. Completing the
set quadrupled the recordings — 312 to 1194 — and moved the number that
actually constrains a split not at all:

| environment | recordings | independent sessions |
|---|---|---|
| drone_los | 757 | 12 |
| drone_nlos | 269 | 3 |
| pedestrian_partial_los | 96 | 4 |
| pedestrian_nlos | 48 | 2 |
| **indoor** | **24** | **1** |

Every session was already represented in the first 13 for the classes that
matter; the remaining nine added more recordings from environments that already
had sessions. `indoor` is one session, and no quantity of files drawn from it
makes it two — a three-way split cannot put that class in train, validation and
test, and `iqforge build` says so rather than pretending otherwise.

This is the concrete version of a distinction that is easy to state and easy to
forget when a dataset page advertises its size. **The number that bounds a
leak-free split is not how many files there are but how many independent
acquisitions they came from**, and the two can differ by any factor at all: a
class with 757 recordings and 12 sessions and a class with 24 recordings and 1
session sit in the same table. Collecting more of the same session buys
statistical precision within it and buys nothing at all against the failure a
recording-level split exists to prevent.

It is also why the criteria in this section are stated in sessions and
acquisitions rather than in file counts, and why the survey in §6 keeps
reporting "N recordings, M independent units" as two numbers instead of one.

### A dataset can be unusable for one task and fine for another

LoRaIQ produced a disqualification that is not on the list above. The first
four are properties of the *dataset*: the format, the recording count, the
physical twinning, the task's difficulty gradient. This one is a property of the
**pairing of a dataset with a class definition**, and it rules out no dataset at
all — only a combination.

Take the class to be the **receiver**, which is the natural reading of a set
with four rooftop radios: four classes, 22 capture sessions each, sessions
perfectly crossed with class. On the count criteria it is the best-structured
dataset in this survey.

It cannot be split without leaking, and the reason is arithmetic rather than
empirical. Every transmission is heard by all four receivers simultaneously, so
each transmission yields one recording per class. Holding a transmission
together — which physical independence requires, since the four files are one
instant of radio — puts one recording of *every* class into whichever split the
transmission lands in. The unit of independence therefore spans all four
classes, and a unit cannot belong to one class and to four classes at once.
`iqforge build` refuses the combination outright:

```
A group goes to one split as a whole, so it cannot belong to two classes at
once. Either the grouping key is wrong, or the labels are.
```

Nothing is wrong with the data. The same 312 recordings, labelled by
**propagation environment** instead, split cleanly: a transmission is received
from one location, so a transmission group carries exactly one environment
label, and the constraint and the class definition stop competing. That is the
build §3's LoRaIQ sweep uses.

So the fifth criterion is not another hurdle for datasets to clear. It is a
check to run on the *question* being asked of one: **is the class you have
chosen constant within the unit of independence?** When it is not — when the
thing that must stay together is exactly the thing that must be told apart — no
amount of data fixes it, and no split exists. The failure is worth naming
because it is invisible from a dataset description: file counts, licences,
formats and difficulty all look fine, and the contradiction only appears when
the grouping key and the label are written down side by side.

## 7. Silent failures found along the way

Each of these produced plausible output. None was found by reading code.

**Class bandwidth inequality.** The synthetic generator initially used
`BPSK_SYMBOL_RATE = 64_000` and `QPSK_SYMBOL_RATE = 128_000`. The two classes
therefore differed by a factor of two in occupied bandwidth, and a classifier
could separate them from the spectrum alone without learning anything about
modulation. Any accuracy measured on that data would have been real and
meaningless. *Why it was silent:* the numbers would have looked good. Nothing in
a training curve distinguishes "learned the modulation" from "learned the
bandwidth". *How it was found:* by auditing what the two classes were allowed to
differ in, before trusting any result from them. Both classes now share symbol
rate, occupied bandwidth (86.4 kHz), burst duration and mean burst power.

**A nuisance-variable balancer that manufactured a shortcut.** The first
`--balance-by` implementation shared group counters across classes so that splits
would "complement each other". The effect was that within a split, carrier offset
predicted the class exactly — train held BPSK at positive offsets and QPSK at
negative ones, with the relationship inverted in test. The model learned the
shortcut, reached 100% on training data and **0% on test**, below chance for two
classes. *Why it was silent:* the class distribution was perfect and the split
report looked correct; the defect was in the joint distribution of class and
offset within each split, which no per-class count reveals. *How it was found:*
by an accuracy far enough below chance to be impossible under an honest split —
0% is not bad luck, it is an inverted rule. The rule is now that group and label
must be independent **within** each split, and a regression test checks it across
seeds.

**Terminal markup deleting part of a command.** `rich` interprets bracketed text
as style tags. The error message telling a user to run
`pip install 'iqforge[torch]'` was printed as `pip install 'iqforge'` — a valid
command that installs the wrong thing. *Why it was silent:* the source string was
correct, and the rendered string was still a runnable command. *How it was found:*
by testing what reached the screen instead of what was in the source. Measurement
showed `[not a tag]`, `label[a]` and `core:hw[ext]` are consumed as well, while
`['x']` and `[1024]` survive, so every user-derived value flowing into `rich` is
now escaped. A regression test captures the rendered output and asserts `[torch]`
is present in it.

**A library mutating its caller's data.** `SigMFFile(metadata=d)` modifies `d`
in place, replacing `core:version` with the spec version the installed library
implements. Three real captures declaring `1.0.0` were reported as `1.2.6`.
*Why it was silent:* the reported version is a plausible version, and it is the
same for every file, so nothing looks inconsistent. *How it was found:* by
comparing `iqforge info` output against the raw JSON of a downloaded file. The
version is now read before the dict is handed over, and `info` shows both values
when they differ. A test pins the upstream behaviour so that a future fix
upstream is noticed; an issue draft is in
[`docs/sigmf-python-issue-draft.md`](sigmf-python-issue-draft.md).

**A recording identified by its file name.** `--labels csv` and `--group-by
csv:` both reduced their lookup key to the bare file name. On LoRaIQ, where
`3.sigmf-meta` exists under every capture session and every receiver, a
312-row label table collapsed to 47 distinct keys and **310 of 312 recordings
came out carrying one label**. The build printed no warning. `iqforge audit`
carried the same blindness against its own manifest — it keyed features by file
name while the manifest keys records by relative path, so every lookup missed,
the class-axis checks fell silent, and the split lookup compared `None` to
`None` and reported 465 pairs of recordings as correctly grouped without having
checked one of them.

*Why it was silent:* the collapsed table is still a valid table. Every
recording gets a label, every window is labelled, the class counts are
plausible, and the split satisfies every constraint the tool checks — it is
simply a dataset of confidently wrong labels, which is exactly the outcome the
annotation path refuses to produce and documents refusing (README, *Known
limitations*). The audit's version was worse: `None == None` is `True`, so an
unchecked property reported as a **pass**, in the one command written on the
principle that an unexamined area must never read as a pass.

*How it was found:* not by reading the code, and not reachable from the test
suite as it stood. Every fixture in this repository is a flat directory of
uniquely named recordings — `bpsk_01`, `qpsk_03` — and in a flat layout the file
name *is* a unique key, so the bug is invisible. It appeared the first time the
tool was pointed at a real nested layout, and even then only because the class
distribution in the manifest was implausible enough to read twice: 310 of one
class and 2 of another, from a table that had four balanced classes in it.

*What it cost to fix:* both tables now match the value as written — normally the
path relative to the input directory — and fall back to the bare name only when
that name is unambiguous, refusing with a message that names the fix when it is
not. A flat layout is unaffected, which is why nothing caught it.

The general lesson is the one this section keeps producing in different forms:
**the tool violated its own stated principle inside its own code.** iqforge
exists because a silent fallback to a plausible-but-wrong answer is worse than
an error, and here it had one, in the identifier that decides what every
downstream guarantee is about. Fixtures that share the shape of the code's
assumptions cannot find that class of bug. Real data with an inconvenient
layout can.

**An acquisition method that broke the experiment.** Fetching a 71 GB archive's
contents over HTTP range requests, one request per file, ran at 20 KB/s: the
files of one capture session are spread across half a gigabyte of the archive,
so each request paid full latency for 70 KB of payload. Sorting the archive's
entries by their stored offset and taking one contiguous 25 MB request per
(session, receiver) cut 480 requests to 32 and ran 25-35 times faster. Every
byte extracted was correct, verified against the ZIP64 central directory's
recorded sizes.

It also destroyed the experimental design. Selecting files by *storage
adjacency* rather than by *transmission id* means the six files taken for
receiver 1 are a different set of transmissions from those taken for receiver 2,
so the simultaneous receptions that make a transmission one event stop landing
in the sample together. Of 479 transmission groups in the resulting set only 115
hold all four receivers, against a clean four-per-group in the set gathered the
slow way; and the number of files per (session, receiver) went from a uniform 6
to anywhere between 1 and 25.

*Why it was silent:* nothing about the result looks wrong. The files are
byte-correct, the directory layout is identical, the audit still returns
`unknown`, the class counts are plausible and larger than before, and every
per-file check passes. The damage is entirely in the *relationships between*
files, which no per-file check can see, and it was introduced by an operation
whose only stated purpose was to go faster.

*How it was found:* by counting group sizes after the download, because the
transmission grouping was going to be used next. Had the set been used directly,
the `--group-by` constraint would have been satisfied trivially — most groups
had one member — and the experiment would have reported a clean split that
enforced nothing.

This is not leakage, but it belongs to the same family: **an operation that is
technically correct per item and invalid in aggregate, and that cannot be
detected by looking at the result.** A dataset assembled by a sampling procedure
carries that procedure's structure whether or not anyone wrote it down, which is
§6's argument arriving from the other direction — there about published data,
here about data we gathered ourselves.

---

## 8. Methods

**Mutation testing.** A test that has never failed has not been shown to test
anything. Each of the guarantees below was verified by deliberately breaking the
implementation and confirming the test fails:

- swapping I and Q in the reader — the reference-tone test must fail, and must
  fail on the *sign* of the frequency offset, not merely its magnitude
- moving the `core:version` read to after the library constructor
- disabling the UTF-8 stream reconfiguration
- changing `__version__` so it disagrees with the packaged metadata

**Control experiments.** A check that fires is only meaningful if it also stays
quiet when it should. The out-of-range annotation warning was tested against an
annotation ending exactly at the last sample (no warning); the locale encoding
tests include a UTF-8 case that must pass while the cp1254, cp1252 and ASCII
cases fail without the fix.

**Treating an implausible accuracy as a symptom, in either direction.** A
standing rule during development: a test accuracy above 98% is a prompt to audit,
not to celebrate. It prompted the leakage audit script, which checks recording
disjointness, cosine similarity between test and training windows, and the class
× nuisance contingency per split.

The same reflex applied downward caught the balancer bug in §7: 0% on two
balanced classes is not bad luck, because bad luck averages to 50%. A score that
far below chance means the model learned a rule that is *inverted* in the test
set, which points straight at the joint distribution rather than at the model.

**Never swallowing a warning.** The experiment harness captures subprocess output
and inspects it; any warning from `build` aborts the run. This exists because the
first version discarded that output and consequently measured a confounded split
for an entire grid. A warning that no one reads is equivalent to no warning.

**Measuring rather than reasoning.** Where a claim could be checked by running
something, it was — including claims that turned out to be wrong. The initial
diagnosis of a "uniform spectrogram bug" on the cellular recording was incorrect:
the renderer was working and the terminal capture had stripped the colour that
carries the information. Counting the distinct colours in a forced-colour render
(338) settled it in one step.

---

## 9. Limits

The measurements in §2 and §3 are narrow. They should be read as a demonstration
of a mechanism, not as an estimate of an effect size that transfers.

**One architecture.** A single baseline CNN, 13 490 parameters, global average
pooling, 20 epochs, Adam. Nothing here says how the effect scales with model
capacity. A larger model memorises more readily and would plausibly show a larger
gap, but that is a hypothesis, not a result.

**One window length.** 1024 samples throughout. The stride sweep varies overlap
at that fixed length; it does not vary the length itself.

**One signal pair, synthetic.** BPSK against QPSK, root-raised-cosine shaped,
additive white Gaussian noise, no fading, no interference, no hardware
impairments, one receiver. Real captures were used to validate the *reader*
(§5), not to measure leakage.

**One SNR definition.** Burst power against noise power in the full band. The
burst occupies a fraction of the band and a fraction of each recording, so the
in-band, in-burst SNR is higher than the quoted figure. The numbers are
comparable within these tables and not directly comparable with SNR figures
computed differently elsewhere.

**The stride sweep confounds overlap with dataset size** (§3). The zero-overlap
row is unaffected by this and carries the causal claim on its own.

Open questions this repository does not answer:

- **How large is the inflation on a real capture, with real channel effects?**
  **Answered for the mechanism, open for the magnitude.** On LoRaIQ (§3) the
  zero-overlap null holds at +1.5 pp ± 1.7 and the 7/8-overlap inflation is
  +9.6 pp ± 2.7, t = 3.5 — the first individually significant real-data result
  in this project. What no dataset has yet produced is the *shape* of the
  dose-response curve: the intermediate overlaps disagree across all three
  datasets, and n = 15 does not resolve them.

  Getting there took five datasets: AirID ruled
  out on format, Vega-C on sessions crossed with class, DASH7 `ds_indoor` on
  recordings that are structurally independent and physically near-duplicate,
  DASH7 cabled on a task whose difficulty is a step function rather than a
  gradient, and LoRaIQ — which carried it. §6 gives the reasoning for each.

  The DASH7 cabled set carried the stride sweep first (§3) and reproduced the
  zero-overlap null, but could not size the effect: the trend was in the right
  direction at t = 2.07 with no individual row significant. LoRaIQ closed that
  gap. The two obstacles that made the search long are different from each
  other and both remain: published RF datasets mostly do not state which of
  their files share an acquisition, so independence has to be inferred; and a
  dataset can satisfy every structural criterion and still pose a task with no
  region of partial competence to measure in.
- How does it scale with model capacity, window length, or the number of
  recordings?
- Does recording-level splitting remain sufficient when recordings share a
  receiver, a session, or a calibration? The DASH7 case in §6 says probably not:
  two captures 43 seconds apart in a static room are separate recordings and the
  same channel realisation. Recording-level splitting removes within-recording
  leakage; it says nothing about correlations *between* recordings, and
  `--balance-by` spreads a nuisance variable rather than keeping related
  recordings together. How much that residual costs has not been measured.
- Is 0.75% usable windows typical for dense real spectrum, or particular to this
  capture? One recording is not a sample.

---

## Reproducing

```bash
git clone https://github.com/emrefbulut/iqforge && cd iqforge
uv sync --extra torch

uv run python scripts/leakage_experiment.py                  # section 2
uv run python scripts/leakage_experiment.py --sweep stride   # section 3
uv run python scripts/leakage_real.py --sweep stride         # section 3, real
```

All three write to `artifacts/`. None touches `examples/`; the synthetic
experiment generates its own recordings at each noise level, and the real one
reads a DASH7 capture set you supply with `--source`. The grids are 180, 150 and
150 training runs; runtime is dominated by training on CPU and is measured in
hours rather than minutes. Results are checkpointed after every run, so an
interrupted grid keeps what it has completed.

The audit in §6 needs no training and takes seconds:

```bash
uv run iqforge audit <recording folder> --dirname-level 2
```

See also the [README](../README.md) for what the tool does and its known
limitations.
