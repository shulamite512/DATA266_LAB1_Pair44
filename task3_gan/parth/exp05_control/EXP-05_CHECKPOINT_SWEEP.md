# EXP-05 checkpoint sweep (read-only)

Run `task3_parth_exp05_5090_baseline_control_20260930-145628`, config `configs/exp05_5090_baseline_control.yaml` (hash `dcfb462b504c1704`, seed 8503), RTX 5090, torch 2.11.0+cu130, cuDNN 91900. Generated 2026-09-30 with `evaluation/checkpoint_sweep.py` (unchanged), which uses the unchanged instructor course evaluator through `evaluation/run_course_evaluation.py`: N_EVAL 300, BATCH_SIZE 32, torchvision Inception-v3, the same preprocessing, sqrtm FID and paired-by-index MiFID. No training, no resume; checkpoints were only read.

- Full report: `task3_gan/parth/checkpoint_sweep_exp05.csv`.
- Per-candidate folders (300 A2B + 300 B2A images, executed notebook, `course_metrics.json`): `outputs/checkpoint_sweep/task3_parth_exp05_5090_baseline_control_20260930-145628/`.
- Inventory and raw results: `sweep_summary.json` in the same folder.

## Control

Two checkpoints were evaluated first: the epoch-60 generator snapshot and `final_generators.pt`.

- **Recorded result:** both reproduce the recorded EXP-05 result exactly, with |diff| = 0 on all seven values. The recorded values come from `evaluation/executed/course_evaluation_exp05_executed.ipynb`, and `submission_exp05.csv` matches them: combined 51.7375179073123.
- **Images:** the epoch-60 images are byte-identical to the evaluated prediction folders `final_generators_exp05`.
- **Weights:** `final_generators.pt`, `latest.pt` and `full_epoch_060.pt` have weights identical to `generators_epoch_060.pt`.

All 20 checkpoint files carry config hash `dcfb462b504c1704`, and each one's epoch and step agree with its file name.

**CONTROL PASSED.**

## Results (lower is better)

Because the control matched exactly, the delta vs 51.7375179073123 equals the delta vs epoch 60.

| Rank | Epoch | Checkpoint | FID_A2B | FID_B2A | MiFID_A2B | MiFID_B2A | Sub FID | Sub MiFID | Combined | vs epoch 60 (= vs 51.7375) | vs EXP-00 52.5733 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 60 | final_generators.pt / generators_epoch_060.pt | 103.0926 | 103.0202 | 0.42286 | 0.41449 | 103.0564 | 0.41867 | **51.7375** | +0.0000 (0.000%) | −0.8357 (−1.590%) |
| 2 | 55 | generators_epoch_055.pt | 102.8009 | 103.3744 | 0.42165 | 0.41507 | 103.0877 | 0.41836 | 51.7530 | +0.0155 (+0.030%) | −0.8202 (−1.560%) |
| 3 | 50 | generators_epoch_050.pt | 102.5950 | 104.4192 | 0.42428 | 0.41654 | 103.5071 | 0.42041 | 51.9637 | +0.2262 (+0.437%) | −0.6095 (−1.159%) |
| 4 | 45 | generators_epoch_045.pt | 105.4018 | 103.1298 | 0.42779 | 0.41272 | 104.2658 | 0.42025 | 52.3430 | +0.6055 (+1.170%) | −0.2302 (−0.438%) |
| 5 | 35 | generators_epoch_035.pt | 109.6135 | 107.3388 | 0.42804 | 0.41166 | 108.4762 | 0.41985 | 54.4480 | +2.7105 (+5.239%) | +1.8748 (+3.566%) |
| 6 | 40 | generators_epoch_040.pt | 110.4864 | 108.0529 | 0.43036 | 0.41626 | 109.2696 | 0.42331 | 54.8465 | +3.1090 (+6.009%) | +2.2732 (+4.324%) |
| 7 | 30 | generators_epoch_030.pt | 107.7015 | 114.3192 | 0.42651 | 0.41117 | 111.0104 | 0.41884 | 55.7146 | +3.9771 (+7.687%) | +3.1413 (+5.975%) |
| 8 | 25 | generators_epoch_025.pt | 121.0994 | 109.3880 | 0.43006 | 0.41141 | 115.2437 | 0.42073 | 57.8322 | +6.0947 (+11.780%) | +5.2589 (+10.003%) |
| 9 | 20 | generators_epoch_020.pt | 114.5378 | 128.7331 | 0.43539 | 0.41839 | 121.6355 | 0.42689 | 61.0312 | +9.2937 (+17.963%) | +8.4579 (+16.088%) |
| 10 | 15 | generators_epoch_015.pt | 132.1606 | 127.9463 | 0.43967 | 0.42009 | 130.0535 | 0.42988 | 65.2417 | +13.5042 (+26.101%) | +12.6684 (+24.097%) |
| 11 | 10 | generators_epoch_010.pt | 153.8787 | 149.6078 | 0.44155 | 0.42728 | 151.7432 | 0.43442 | 76.0888 | +24.3513 (+47.067%) | +23.5156 (+44.729%) |
| 12 | 5 | generators_epoch_005.pt | 218.2598 | 155.1967 | 0.49716 | 0.42027 | 186.7282 | 0.45872 | 93.5935 | +41.8560 (+80.901%) | +41.0202 (+78.025%) |

Full-precision values and checkpoint SHA-256s are in the CSV.

## Findings

- **Best checkpoint: epoch 60** (`final_generators.pt`, sha256 `e979e900…7b05b92`; identical weights to `generators_epoch_060.pt`). Combined 51.7375179073123, which is the already-recorded EXP-05 result.
- **Runner-up: epoch 55** (`generators_epoch_055.pt`, sha256 `4bd88c62…25c68f`). Combined 51.753015054999686, worse than epoch 60 by +0.0155 (+0.030%). It has slightly better FID_A2B (102.80 vs 103.09) but worse FID_B2A (103.37 vs 103.02).
- **Improvement vs epoch 60: none.** No earlier checkpoint beats the final one.
- **Improvement vs the original EXP-00 baseline (52.5732635808473):** epoch 60 is lower by 0.8357 (−1.590%), and epochs 45–55 are also below EXP-00.
- **Training curve:** the score improves almost steadily through the linear-decay phase (epochs 30 → 60), with one bump at epoch 40 vs 35, and flattens over the last 10 epochs (55 → 60: −0.0155). This matches the EXP-00 sweep, where epoch 60 was also best and epoch 55 second (52.5733 vs 52.8602). It differs from EXP-02, whose longer decay peaked before the end (epoch 65, 52.8423).

## Copy / memorization status

The checks follow the validate_submission.py logic: byte copies, a near-copy check against each source (64×64 MAD: FAIL < 2, WARN < 5), a nearest-neighbour near-copy check against every real image of the target domain (all 300 Monet / all 7,038 photos), near-blank images, and JPEG/RGB/256 integrity.

| Images checked | Status | Byte copies | Min source MAD (A2B / B2A) | Min target-NN MAD (A2B / B2A) | Near-blank |
|---|---|---|---|---|---|
| Epoch 60, 300 + 300 course images (this sweep) | PASS | 0 | 7.30 / 5.78 | 18.41 / 18.88 | 0 |
| Epoch 55, 300 + 300 course images (this sweep) | PASS | 0 | 6.32 / 5.47 | 18.87 / 19.82 | 0 |
| Epoch 60, full prediction set 300 A2B + 7,038 B2A (EXP-05 evaluation) | WARN | 0 | 7.30 / 4.88 | 18.41 / 16.69 | 0 |

The WARN is one B2A output out of 7,038 with source MAD 4.88 (between 2 and 5). It is not in the 300 images the course evaluator reads, and it is not a failure.

## Selection-bias warning

Every checkpoint is scored on the same 300 + 300 course-evaluation images that produce the reported score, so choosing the minimum over 12 checkpoints is optimistically biased. Here the selection changes nothing, because the final checkpoint is the minimum. More importantly, EXP-05 is an identical-configuration replicate of EXP-00. Its 0.84-point advantage over EXP-00 is a measure of run-to-run / environment variation, not an improvement from any change. Submitting the better of two identical-configuration runs is itself a best-of-2 selection on the evaluation set, and it should be described that way rather than as a method improvement.

## Is an earlier checkpoint worth considering as a submission candidate?

**No.** No earlier checkpoint beats epoch 60. The runner-up (epoch 55) is 0.0155 worse, which is well within the adjacent-checkpoint noise (0.2–1.2 points in earlier sweeps). The epoch-60 generators are already the ones behind `submission_exp05.csv`, so the sweep gives no reason to create a new submission.

No submission was created or changed; `experiments.csv` was not modified; nothing was uploaded to Kaggle.
