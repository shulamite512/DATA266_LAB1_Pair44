# Task 3: CycleGAN, Monet ↔ Photo (Parth)

Final result: **Experiment 05 (EXP-05)**, run `task3_parth_exp05_5090_baseline_control_20260930-145628`. Every number here is copied from a recorded file of that run, and each table names its source.

| Artifact | Path (under `task3_gan/parth/` unless stated) |
|---|---|
| Final notebook (executed, with outputs) | `src/Task3_Parth_CycleGAN_Final.ipynb` |
| Experiment config / base config | `configs/exp05_5090_baseline_control.yaml`, `configs/baseline.yaml` |
| Resolved config, data report | `outputs/runs/<run_id>/config_resolved.yaml`, `outputs/runs/<run_id>/data_report.json` |
| Checkpoint (G_A2B + G_B2A, epoch 60) | `checkpoints/<run_id>/final_generators.pt` (see section 8) |
| Step / epoch metrics, training plots | `outputs/runs/<run_id>/step_metrics.csv`, `epoch_metrics.csv`, `outputs/plots/<run_id>/*.png` |
| Raw training log, raw evaluation log | `reproducibility/raw_logs/<run_id>.log`, `reproducibility/raw_logs/<run_id>_eval_final_generators_exp05_20260930-162436.log` |
| Run manifest, pip freeze | `reproducibility/manifests/<run_id>.json`, `reproducibility/manifests/<run_id>_pip_freeze.txt` |
| Course-evaluator submission | `submission.csv` (byte-identical to `submission_exp05.csv`) |
| Course-evaluator values for `final_generators.pt` | `outputs/checkpoint_sweep/<run_id>/final_generators_epoch_060/course_metrics.json`, `course_submission.csv` |
| Full metrics (EXP-05 rows) | `full_metrics_report.csv`, `metrics_report.csv`; internal evaluator output `outputs/eval/<run_id>/final_generators_exp05/metrics.json` |
| Checkpoint sweep | `checkpoint_sweep_exp05.csv`, `exp05_control/EXP-05_CHECKPOINT_SWEEP.md`, `outputs/checkpoint_sweep/<run_id>/` |
| Run report, watcher and evaluation logs | `exp05_control/` |
| All EXP-00 to EXP-05 runs (one row each) | `experiments.csv` |
| Provenance (repo file → archive file, SHA-256) | `EXP05_PROVENANCE.csv` |

## 1. Data

Source: run manifest (`data` block), `outputs/runs/<run_id>/data_report.json`.

| Item | Value |
|---|---|
| Domain A | 300 Monet paintings, `task3_gan/data/monet_jpg/`, all 256×256 RGB |
| Domain B | 7,038 photos, `task3_gan/data/photo_jpg/`, all 256×256 RGB |
| Duplicates | 9 byte-identical groups within the photos, none within Monet, none across domains |
| Training preprocessing | Bicubic resize to 286, random 256 crop, random horizontal flip, scaled to [-1, 1] |
| Pairing | None: two independent seeded permutations per epoch |
| Steps per epoch | 1,000 (batch size 1) |

## 2. Architecture

Source: `outputs/runs/<run_id>/config_resolved.yaml`, run manifest (`parameters`), notebook sections 6–9.

| Component | Setting |
|---|---|
| Generators `G_A2B`, `G_B2A` | ResNet generator: c7s1-64, d128, d256, 9 residual blocks, u128, u64, c7s1-3, tanh; instance norm, reflection padding, transposed-convolution upsampling; 11,378,179 parameters each |
| Discriminators `D_A`, `D_B` | 70×70 PatchGAN (C64 without norm, C128, C256, C512 stride 1, 1-channel output), LeakyReLU 0.2; 2,764,737 parameters each |
| Total parameters | 28,285,832 (generators 22,756,358, discriminators 5,529,474) |
| Initialization | N(0, 0.02), trained from scratch; no pretrained weights in G or D |

### Design choices

**ResNet-9 generators.** Monet-photo translation should keep a scene's layout while changing colour, texture and brushwork. The generator encodes the image to a 64×64 feature map (c7s1-64, d128, d256), applies 9 residual blocks there and decodes back to 256×256; each residual block computes x + F(x), so its default is to pass features through unchanged and learn only the change, which fits a task where most of the structure should survive. The two generators hold 22.76M of the 28.29M parameters.

**70×70 PatchGAN discriminators.** Each discriminator outputs a grid of real/fake scores, one per overlapping 70×70 patch, instead of one score for the whole image. Style in this task is largely local (brush strokes, colour texture), so judging patches pushes the generator toward realistic local texture with far fewer parameters (2.76M each) than a whole-image classifier would need; global content is constrained by the cycle-consistency loss instead.

**Instance normalization and reflection padding.** Instance normalization normalizes each feature map per image, removing image-specific contrast and colour statistics; it is the common choice for style transfer and image translation, and no alternative was compared here. Reflection padding fills borders with mirrored pixels instead of zeros, to avoid the dark or artificial edges that zero padding can leave in generated images.

**Transposed-convolution upsampling.** The two decoder stages use stride-2 transposed convolutions, as in the original CycleGAN. Transposed convolutions can produce checkerboard patterns where kernel overlaps are uneven, and notebook section 19 reports a periodic grid texture in the sky of one EXP-05 sample. The resize-convolution alternative (EXP-01) scored worse on the course evaluator (57.2902 against 52.5733 for EXP-00), so the transposed convolutions were kept.

## 3. Training configuration

Source: `outputs/runs/<run_id>/config_resolved.yaml`.

| Setting | Value |
|---|---|
| Adversarial loss | LSGAN (least squares) |
| Cycle-consistency loss | L1, λ_cycle = 10.0 |
| Identity loss | L1, λ_identity = 5.0 |
| Optimizer | Adam, lr 2e-4 for G and D, β = (0.5, 0.999) |
| Schedule | 30 epochs constant + 30 epochs linear decay (60 epochs, 60,000 steps) |
| Replay pool | 50 past fakes per discriminator |
| Precision | FP32 (`amp: none`), TF32 allowed, `cudnn.benchmark` on |
| Seed | 8503 |
| Config hash | `dcfb462b504c1704` |
| Command (from the run report) | `python task3_gan/parth/src/train.py --config task3_gan/parth/configs/exp05_5090_baseline_control.yaml --run-id task3_parth_exp05_5090_baseline_control_20260930-145628` |

EXP-05 uses exactly the settings of `baseline.yaml` (EXP-00); the experiment config changes only `run_name`. It was trained from scratch on a different GPU and CUDA build from EXP-00 (`exp05_control/EXP-05_RESULT.md`).

### Why these settings

**Losses.**

- **LSGAN adversarial loss.** Each discriminator is trained with a squared-error loss to output 1 on real images and 0 on generated ones, and each generator is trained so that the discriminator outputs 1 on its fakes. CycleGAN uses this least-squares form in place of the log loss because it gives more stable training.
- **Cycle consistency, λ_cycle = 10.** With unpaired data many mappings could fool a discriminator, including ones that ignore the input. The L1 loss on the A → B → A and B → A → B reconstructions requires each translation to keep enough information to be undone, which ties the output to the input's content.
- **Identity loss, λ_identity = 5.** A real photo fed to the Monet → photo generator, or a real painting fed to the photo → Monet generator, should change very little; the L1 penalty discourages unnecessary colour and tint changes.

These weights are the paper's recipe, with identity at half the cycle weight. EXP-03 (identity 2.5) and EXP-04 (cycle 7.5 with a slower discriminator) did not beat it on the course evaluator, which is evidence against those two changes, not proof that 10 and 5 are optimal.

**Optimization.**

| Setting | Value | Purpose |
|---|---|---|
| Optimizer | Adam, lr 2e-4 for G and D, betas (0.5, 0.999) | The standard CycleGAN setting; a beta1 of 0.5 gives less momentum than the default 0.9, which is common in GAN training where the target keeps moving |
| Schedule | 30 epochs constant, then linear decay to 0 over 30 epochs (60,000 steps) | Large steps while the mapping forms, then progressively smaller ones; the learning rate is 1.94e-4 at epoch 31 and 6.45e-6 at epoch 60 |
| Batch size | 1 | The CycleGAN default for 256×256 images; instance normalization does not depend on batch statistics |
| Replay pool | 50 past fakes per discriminator | Each discriminator update sees a mix of the newest and older generated images instead of only the latest generator's output, which is intended to reduce oscillation between the two networks |
| Precision | FP32 | Full precision for the adversarial updates (no mixed precision); TF32 matmuls allowed |
| Initialization | N(0, 0.02), from scratch | The standard CycleGAN initialization; no pretrained weights |
| Seed | 8503 | Seeds the data permutations and the initialization; `cudnn.benchmark` is on, so GPU kernels stay nondeterministic and the same seed does not reproduce a run exactly |

**Why EXP-05 is the final run.** On the course evaluator (lower is better) the six recorded runs score EXP-00 52.5733, EXP-01 57.2902, EXP-02 53.2846, EXP-03 52.9316, EXP-04 53.4947 and EXP-05 51.7375 (`experiments.csv`), so EXP-05 has the best combined score, and the checkpoint sweep picks its epoch-60 weights (`final_generators.pt`). EXP-05 is an exact replicate of the baseline configuration and seed, trained on a different GPU and CUDA build, so its 0.8357-point improvement over EXP-00 is not the result of any architecture or loss change. Run-to-run and hardware/software variation is the plausible explanation, but one replicate cannot separate those sources (`exp05_control/EXP-05_RESULT.md`).

## 4. Hardware and cost

Source: run manifest (`environment`, `totals`).

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32,607 MB, sm_120, driver 610.60), 1 GPU |
| Software | Windows 11, Python 3.12.10, torch 2.11.0+cu130 (CUDA 13.0), torchvision 0.26.0+cu130, cuDNN 91900 |
| Training loop time | 4,245.34 s (wall clock over the session 5,227.34 s) |
| Training throughput | 28.266 images/s (120,000 images seen) |
| Peak GPU memory | 9,028.8 MB allocated, 9,784.0 MB reserved |
| Inference throughput (internal evaluator) | 321.50 images/s A2B, 518.95 images/s B2A (`full_metrics_report.csv`) |

## 5. Training stability

Source: run manifest (`totals`) and `outputs/runs/<run_id>/epoch_metrics.csv` (epoch 60 means). Plots: `outputs/plots/<run_id>/` (generator total, adversarial, cycle, identity and discriminator losses, discriminator outputs, gradient norms, learning rate).

| Item | Value |
|---|---|
| Non-finite (NaN/Inf) steps | 0 |
| Loss-spike steps (loss > 3 × EMA after 500 warm-up steps) | 38 of 60,000 |
| Discriminator-saturated steps | 6 of 60,000 |
| Epoch 60: G total / adv A2B / adv B2A | 3.3701 / 0.4400 / 0.5183 |
| Epoch 60: cycle A / cycle B | 0.0707 / 0.0933 |
| Epoch 60: identity A / identity B | 0.0689 / 0.0854 |
| Epoch 60: D_A / D_B | 0.1424 / 0.1688 |
| Epoch 60: mean gradient norm G / D_A / D_B | 22.69 / 7.24 / 6.98 |

### Interpretation

The generator's total loss falls steadily: 9.274 in epoch 1, 5.871 in epoch 10, 4.973 in epoch 20, 4.609 in epoch 30, 4.064 in epoch 40, 3.606 in epoch 50 and 3.370 in epoch 60; the training cycle loss for domain A falls from 0.277 to 0.0707. The decline continues after the learning rate starts its linear decay at epoch 31, and the checkpoint sweep agrees: epoch 60 scores best on the course evaluator, with epoch 55 only 0.0155 higher. Over 60,000 steps there were 0 non-finite steps, 38 loss-spike steps (24 in epochs 1 to 30 and 14 in epochs 31 to 60, spread over 31 epochs) and 6 discriminator-saturated steps (in epochs 12, 21, 33, 42, 47 and 50); the largest generator gradient norm, 283.95, came in epoch 1.

These diagnostics show that training stayed numerically stable and that neither network overwhelmed the other for long: the spikes and saturated steps are isolated, and at epoch 60 the discriminator losses (D_A 0.142, D_B 0.169) are far from 0. GAN losses do not measure image quality, so a falling generator loss is not proof of good images; it only shows that the logged quantities have no obvious collapse signature. For contrast, EXP-01 had 319 saturated steps and 189 spikes (`exp05_control/EXP-05_RESULT.md`).

## 6. Results

### Course evaluator (instructor notebook; lower is better)

Source: `outputs/checkpoint_sweep/<run_id>/final_generators_epoch_060/course_metrics.json`, `submission.csv`, `exp05_control/EXP-05_RESULT.md`. The course evaluator reads the first 300 sorted images per direction. Combined = (submission FID + submission MiFID) / 2.

| Metric | Value |
|---|---|
| FID A2B (Monet → photo) | 103.0926 |
| FID B2A (photo → Monet) | 103.0202 |
| MiFID A2B / B2A (course definition) | 0.4229 / 0.4145 |
| Submission FID / MiFID | 103.0564 / 0.4187 |
| Combined | **51.7375** (51.7375179073123) |

A later re-evaluation of `final_generators.pt` with the same evaluator reproduced all seven values exactly, and its `course_submission.csv` is byte-identical to `submission.csv` (`exp05_control/EXP-05_CHECKPOINT_SWEEP.md`).

### Full metrics (internal evaluator, all images)

Source: `full_metrics_report.csv` (from `outputs/eval/<run_id>/final_generators_exp05/metrics.json`). These numbers are on a different scale from the course evaluator and are not comparable with it.

| Metric | A2B (Monet → photo) | B2A (photo → Monet) |
|---|---|---|
| Inputs / reference images | 300 / 7,038 | 7,038 / 300 |
| FID | 81.0700 | 85.0933 |
| KID (mean ± std) | 0.0176 ± 0.0008 | 0.0126 ± 0.0009 |
| Precision / recall | 0.5433 / 0.3558 | 0.3605 / 0.5367 |
| Density / coverage | 0.5778 / 0.0604 | 0.3268 / 0.9467 |
| Cycle-reconstruction L1 | 0.0807 | 0.1018 |
| LPIPS input → output | 0.2987 | 0.3801 |
| LPIPS input → reconstruction | 0.2669 | 0.2307 |
| Content-preservation cosine (mean ± std) | 0.8147 ± 0.0675 | 0.7525 ± 0.1058 |
| NaN count | 0 | 0 |

### Checkpoint sweep

Source: `checkpoint_sweep_exp05.csv`. Epoch 60 (`final_generators.pt`) has the lowest combined score; epoch 55 is 0.0155 higher, epoch 50 0.2262 higher. Every checkpoint is scored on the same 300 + 300 images.

### Copy / memorization checks

Source: `exp05_control/EXP-05_CHECKPOINT_SWEEP.md`. No byte copies and no near-blank outputs. One B2A output out of 7,038 has a source mean absolute difference of 4.88 (WARN band 2–5); it is not among the 300 images the course evaluator reads.

### Interpretation

**Course evaluator.** EXP-05 scores FID 103.0926 (Monet → photo) and 103.0202 (photo → Monet), submission FID 103.0564, submission MiFID 0.41867 and combined 51.7375, the lowest combined score of the six recorded runs; the two directions are almost equal on this evaluator.

**What the course MiFID is.** The executed course notebook takes the first 300 images of each folder in sorted order, extracts Inception-v3 pool features (2,048 values per image), computes FID from their means and covariances, and defines "MiFID" as the mean cosine distance between real and generated feature vectors paired by index. The pairs are simply the i-th files of two unrelated folders, so this term measures the average feature dissimilarity of unrelated images; it is not Kaggle's MiFID, which penalizes memorized training images. The combined score is (FID + MiFID) / 2, so with FID near 103 and MiFID near 0.42 it is about 99.6% FID: MiFID adds about 0.21 to the 51.74.

**Internal evaluator.** The local evaluator scores all images (300 Monet inputs against 7,038 real photos for A2B; 7,038 photo inputs against 300 real paintings for B2A) with its own reference statistics, so its FID (81.07 A2B, 85.09 B2A) is on a different scale from the course FID and must not be compared with it. Its source (`evaluate_local.py` and its metric helpers) and reference statistics (`real_stats.npz`) are not packaged here; only its outputs are.

**Per direction.** Monet → photo has higher precision than recall (0.5433 / 0.3558) and very low coverage (0.0604): the generated photos sit in a narrow part of the real-photo distribution, which is partly expected when 300 outputs are compared with 7,038 references. Photo → Monet is the reverse (precision 0.3605, recall 0.5367, coverage 0.9467): the 7,038 outputs cover the 300 real paintings well, but many of them fall outside the region of real Monet features. Photo → Monet also changes its input more (LPIPS input → output 0.3801 against 0.2987) and keeps less of its content (content cosine 0.7525 against 0.8147), which fits a stronger style change in that direction.

**Visual results.** The notebook's description of the epoch-60 sample grid (section 19) reports that photo → Monet keeps the scene layout while shifting the palette and adding brush-like texture, that Monet → photo mostly changes colour and contrast and still looks partly painterly, that the reconstructions are close to the inputs, and that one sunset photo fails with an inverted palette and a periodic grid texture in the sky. The prediction panels in section 19 are embedded in the notebook output; the prediction folders they were read from are not packaged here.

## 7. Items not yet recorded

| Requirement | Recorded value |
|---|---|
| Blinded human audit (30 fixed samples, 2 raters, agreement) | `PENDING` in `full_metrics_report.csv`; the notebook reports no completed audit |
| Kaggle public / private score and rank | `PENDING` / `NA` in `full_metrics_report.csv`; `EXP-05_RESULT.md` records that nothing was uploaded to Kaggle |
| Generated predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` | Not in this repository (300 + 7,038 images, kept in the original Task 3 project folder) |
| `evaluate_local.py` and the training source `src/*.py` | Not in this repository |
| Local-evaluator reference statistics `real_stats.npz` | Not in this repository |
| Earlier failure-analysis file (F1 to F7), `mini_search/`, `checkpoint_sweep_exp02.csv` | Not in this repository; see `failure_analysis.md` |
| Later campaign runs EXP-06 to EXP-11 | Internal-evaluator rows only, in `evaluation/campaign_reports/`; no configs, checkpoints, logs or course scores are packaged, and none of them is part of the reported results (notebook section 22) |

## 8. Checkpoint to result mapping

| Checkpoint | Run | Epoch / step | SHA-256 | Used for |
|---|---|---|---|---|
| `checkpoints/task3_parth_exp05_5090_baseline_control_20260930-145628/final_generators.pt` (91,065,867 bytes) | EXP-05 | 60 / 60,000 | `e979e90002d17e9c9cd3e80c49a10bad4dfa82e5da20399a73d35b54b7a05b92` | Every number in section 6 |

`final_generators.pt` (91,065,867 bytes, both generators at epoch 60) is the checkpoint intended for the GitHub repository for grading and the live demo. It is below GitHub's 100 MB per-file limit but above its 50 MB warning size, so whether it goes into ordinary Git or Git LFS is decided at final packaging.
