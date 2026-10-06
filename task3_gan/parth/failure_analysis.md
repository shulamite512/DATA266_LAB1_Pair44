# Task 3: Failure Analysis (Parth)

Run: EXP-05, `task3_parth_exp05_5090_baseline_control_20260930-145628` (checkpoint `final_generators.pt`, epoch 60). The handout (3.1–3.2) asks for a convergence and stability analysis, a check of cycle consistency, and a discussion of shortcomings such as mode collapse or artifacts.

Notebook section 18 cites entries F1 to F7 of an earlier `failure_analysis.md` from the original Task 3 project folder. That file is not in this repository and was not recreated; section 18 states each finding itself, and the table below lists what each entry covers according to the notebook and whether its supporting evidence is packaged here.

| Entry | Topic (from notebook section 18) | Evidence in this repository |
|---|---|---|
| F1, F2 | Checkerboard texture in EXP-00 and the resize-conv experiment EXP-01 | EXP-01 course score in `experiments.csv`; its spike and saturation counts in `exp05_control/EXP-05_RESULT.md`. The spectral-peak measurements and EXP-01 run files are not packaged |
| F3 | Longer decay, EXP-02 | Course score in `experiments.csv`; `checkpoint_sweep_exp02.csv` is not packaged |
| F4 | Identity weight 2.5, EXP-03 | Course score in `experiments.csv` |
| F5 | Cycle weight 7.5 with discriminator LR 1e-4, EXP-04 | Course score in `experiments.csv`, counts in `EXP-05_RESULT.md`; its checkerboard measurement is not packaged |
| F6 | Run-to-run variation, EXP-00 against EXP-05 | `experiments.csv`, `exp05_control/EXP-05_RESULT.md` |
| F7 | The mini-search did not transfer to full scale | `mini_search/` is not packaged |

The sections below analyse the final run, EXP-05.

## Evidence

| Evidence | Value | Source |
|---|---|---|
| Non-finite steps | 0 of 60,000 | run manifest `totals` |
| Loss-spike steps | 38 of 60,000 | run manifest `totals` |
| Discriminator-saturated steps | 6 of 60,000 | run manifest `totals` |
| Largest per-epoch maximum of the generator gradient norm | 283.95 | `outputs/runs/<run_id>/epoch_metrics.csv` (`gn_G_total_max`) |
| Epoch 60 cycle L1 (training, A / B) | 0.0707 / 0.0933 | `epoch_metrics.csv` |
| Cycle-reconstruction L1 on all evaluation images (A2B / B2A) | 0.0807 / 0.1018 | `full_metrics_report.csv` |
| LPIPS input → reconstruction (A2B / B2A) | 0.2669 / 0.2307 | `full_metrics_report.csv` |
| Precision / recall A2B | 0.5433 / 0.3558 | `full_metrics_report.csv` |
| Precision / recall B2A | 0.3605 / 0.5367 | `full_metrics_report.csv` |
| Coverage A2B / B2A | 0.0604 / 0.9467 | `full_metrics_report.csv` |
| Near-copy check | 0 byte copies; 1 of 7,038 B2A outputs in the WARN band (source MAD 4.88) | `exp05_control/EXP-05_CHECKPOINT_SWEEP.md` |
| Same configuration and seed, different GPU / CUDA build | EXP-00 52.5733 vs EXP-05 51.7375 combined (difference 0.8357) | `experiments.csv`, `exp05_control/EXP-05_RESULT.md` |
| Checkpoint sweep, epochs 5 → 60 | combined 93.59 → 51.74; epoch 55 is 0.0155 above epoch 60 | `checkpoint_sweep_exp05.csv` |
| Training plots | 8 plots of losses, discriminator outputs, gradient norms and learning rate | `outputs/plots/<run_id>/` |
| Sample grid and prediction panels | notebook section 19: the epoch-60 sample grid and the first six inputs per direction with their predictions | `src/Task3_Parth_CycleGAN_Final.ipynb` |

## Convergence and stability

The generator's total loss falls steadily: 9.274 in epoch 1, 5.871 in epoch 10, 4.973 in epoch 20, 4.609 in epoch 30, 4.064 in epoch 40, 3.606 in epoch 50 and 3.370 in epoch 60; the training cycle loss for domain A falls from 0.277 to 0.0707. The decline continues after the learning rate starts its linear decay at epoch 31, and the checkpoint sweep agrees: epoch 60 scores best on the course evaluator, with epoch 55 only 0.0155 higher. Over 60,000 steps there were 0 non-finite steps, 38 loss-spike steps (24 in epochs 1 to 30 and 14 in epochs 31 to 60, spread over 31 epochs) and 6 discriminator-saturated steps (in epochs 12, 21, 33, 42, 47 and 50); the largest generator gradient norm, 283.95, came in epoch 1.

These diagnostics show that training stayed numerically stable and that neither network overwhelmed the other for long: the spikes and saturated steps are isolated, and at epoch 60 the discriminator losses (D_A 0.142, D_B 0.169) are far from 0. GAN losses do not measure image quality, so a falling generator loss is not proof of good images; it only shows that the logged quantities have no obvious collapse signature. For contrast, EXP-01 had 319 saturated steps and 189 spikes (`exp05_control/EXP-05_RESULT.md`).

## Cycle consistency

Cycle consistency holds in the logged and evaluated numbers. The training cycle L1 is 0.0707 (A) and 0.0933 (B) at epoch 60, the reconstruction L1 over all evaluation images is 0.0807 (A2B) and 0.1018 (B2A), and LPIPS between input and reconstruction (0.2669 and 0.2307) is lower than between input and translation (0.2987 and 0.3801), so the round trip returns closer to the input than the one-way translation is. The notebook's description of the sample grid (section 19) reports reconstructions close to the inputs in every row. A low cycle error does not by itself guarantee a faithful translation, because a generator can hide the information needed for reconstruction in faint patterns; that was not checked.

## Shortcomings and failure cases

1. **Periodic grid texture in flat regions.** The notebook's grid description (section 19, row 7) reports a periodic grid texture across the sky of a sunset photo translated to Monet. It is the checkerboard pattern associated with transposed-convolution upsampling, and it is still present in EXP-05; no metric in the packaged files measures it directly (the spectral measurements cited in notebook section 18 come from files that are not packaged). The resize-convolution fix removed some of the texture but made the model worse overall (EXP-01, 57.2902 combined).
2. **Weak Monet → photo realism.** The grid description says Monet → photo outputs mostly change colour and contrast and keep painterly texture. The metrics point the same way: A2B changes its input less than B2A (LPIPS input → output 0.2987 against 0.3801, content cosine 0.8147 against 0.7525) and covers little of the real-photo distribution (recall 0.3558, coverage 0.0604). One possible cause, not tested, is that a painting lacks the fine detail a photo needs, and the cycle and identity losses reward staying close to the input.
3. **Palette failure on an unusual input.** The same sunset photo comes out with an inverted yellow/blue palette (section 19). This is one image out of eight in the grid, so it shows a failure mode, not a failure rate.
4. **Photo → Monet outputs outside the real Monet distribution.** B2A precision is 0.3605, so most generated paintings fall outside the neighbourhood of the 300 real Monet features, even though they cover those features well (coverage 0.9467).

There is no sign of mode collapse in the packaged evidence: B2A coverage is 0.9467 and recall 0.5367, the copy check found no byte copies and no near-blank outputs, and only 1 of 7,038 B2A outputs fell in the near-copy warning band. Every comparison between runs is also limited by run-to-run variation: the same configuration and seed scored 52.5733 and 51.7375 on two machines.

## What I would try next

Each item is a proposed experiment; none has been run.

1. **Checkerboard-free upsampling without unbalancing the GAN.** Repeat the resize-convolution generator with a slower or spectrally normalized discriminator, and compare the course combined score, the spike and saturation counts, and a frequency-domain measure of the grid texture against EXP-05.
2. **Multi-seed replicates.** Train the baseline configuration with three or more seeds on one machine and report the mean and spread of the course combined score, so that later differences can be judged against measured noise instead of a single 0.84-point gap.
3. **Re-run EXP-06 with full artifacts.** Keep its checkpoint, config, logs and predictions, score it with the course evaluator, and apply the pre-registered rule in notebook section 22.
