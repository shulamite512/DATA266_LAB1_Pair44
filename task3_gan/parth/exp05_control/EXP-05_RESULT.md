# EXP-05 result: RTX 5090 baseline control

Generated 2026-09-30T16:31:02-07:00 by `evaluation/run_full_experiment.py` (unattended).

- Run ID: `task3_parth_exp05_5090_baseline_control_20260930-145628`; config `task3_gan/parth/configs/exp05_5090_baseline_control.yaml` (hash `dcfb462b504c1704`); trained from scratch.
- Environment: {'torch': '2.11.0+cu130', 'torch_cuda_build': '13.0', 'cudnn': 91900, 'gpu': 'NVIDIA GeForce RTX 5090'}.
- final_generators.pt sha256 `e979e90002d17e9c9cd3e80c49a10bad4dfa82e5da20399a73d35b54b7a05b92`; verification: all 13 checks passed.
- Training totals: {"epochs_completed": 60, "global_step": 60000, "train_loop_seconds": 4245.34, "images_seen": 120000, "images_per_sec": 28.266, "nonfinite_steps": 0, "spike_steps": 38, "d_saturated_steps": 6, "wall_seconds_all_sessions": 5227.34, "peak_memory": {"gpu_peak_allocated_mb": 9028.8, "gpu_peak_reserved_mb": 9784.0, "cpu_peak_rss_mb": null}}
- Output/copy validation: WARN ['B2A: 1 outputs with source MAD < 5']; reference check MEASURED; LPIPS ['OK', 'OK'].
- Course CSV: `task3_gan/parth/submission_exp05.csv`; executed notebook `task3_gan/parth/evaluation/executed/course_evaluation_exp05_executed.ipynb`.

## Instructor course evaluator (lower is better)

| Run | Change | GPU / CUDA | FID_A2B | FID_B2A | MiFID_A2B | MiFID_B2A | Submission FID | Submission MiFID | Combined | vs EXP-00 | % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-00 baseline (RTX 4090, CUDA 12.8) | baseline.yaml | RTX 4090 / 12.8 | 104.2016 | 105.2573 | 0.42273 | 0.41146 | 104.7294 | 0.41710 | 52.5733 | +0.0000 | +0.000% |
| EXP-01 resize-conv | upsample resize_conv | RTX 5090 / 13.0 | 121.7612 | 106.5696 | 0.42612 | 0.40375 | 114.1654 | 0.41494 | 57.2902 | +4.7169 | +8.972% |
| EXP-02 longer decay | 30+45 epochs | RTX 5090 / 13.0 | 105.1428 | 107.1578 | 0.42300 | 0.41459 | 106.1503 | 0.41879 | 53.2846 | +0.7113 | +1.353% |
| EXP-03 identity 2.5 | lambda_identity 2.5 | RTX 5090 / 13.0 | 106.7335 | 104.1539 | 0.42695 | 0.41201 | 105.4437 | 0.41948 | 52.9316 | +0.3583 | +0.682% |
| EXP-04 cycle 7.5 + lr_D 1e-4 | lambda_cycle 7.5, lr_D 1e-4 | RTX 5090 / 13.0 | 105.5490 | 107.5870 | 0.42909 | 0.41380 | 106.5680 | 0.42144 | 53.4947 | +0.9215 | +1.753% |
| EXP-05 (this run) | none: exact replicate of baseline.yaml (run_name only); RTX 5090 / CUDA 13.0; trained from scratch | RTX 5090 / 13.0 | 103.0926 | 103.0202 | 0.42286 | 0.41449 | 103.0564 | 0.41867 | 51.7375 | -0.8357 | -1.590% |

Full precision (EXP-05): FID_A2B=103.09256329850902, FID_B2A=103.02015942255423, MiFID_A2B=0.4228598177433014, MiFID_B2A=0.41448909044265747, submission_FID=103.05636136053162, submission_MiFID=0.41867445409297943, combined_score=51.7375179073123

**EXP-05 vs EXP-00: -0.835746 (-1.590%).**

## Interpretation (control run)

This is an exact replicate of baseline.yaml (same resolved training configuration and seed 8503, same data, same training code path; the only train.py change since EXP-00, the optional lr_D, defaults to lr) trained from scratch on the NVIDIA GeForce RTX 5090 / CUDA 13.0 machine. EXP-00 was trained on the NVIDIA GeForce RTX 4090 / CUDA 12.8 machine. The difference above is evidence of combined run-to-run and environment variation: non-deterministic GPU kernels (cudnn.benchmark is on, so even the same seed does not reproduce a run), different GPUs, and different CUDA / PyTorch / cuDNN builds. One replicate cannot separate these sources, so the difference must NOT be attributed to hardware alone.

Same-hardware comparison of the RTX 5090 runs against this control:

| Run | Combined | vs this control | % |
|---|---|---|---|
| EXP-01 resize-conv | 57.2902 | +5.5527 | +10.732% |
| EXP-02 longer decay | 53.2846 | +1.5470 | +2.990% |
| EXP-03 identity 2.5 | 52.9316 | +1.1941 | +2.308% |
| EXP-04 cycle 7.5 + lr_D 1e-4 | 53.4947 | +1.7572 | +3.396% |

The EXP-00 vs control gap (0.8357 combined points) is one estimate of the variation between two runs of the same configuration. Differences between configurations that are smaller than this should not be attributed to the configuration change.

## Internal metrics (our evaluator; not comparable with the course numbers)

| Run | Internal FID A2B | Internal FID B2A | Official-ref B2A FID (local approx) | Spikes | D-saturated |
|---|---|---|---|---|---|
| EXP-00 baseline (RTX 4090, CUDA 12.8) | 82.5429 | 89.4100 | 86.0591 | 44 | 12 |
| EXP-01 resize-conv | 95.3980 | 90.3885 | 85.6363 | 189 | 319 |
| EXP-02 longer decay | 83.3787 | 87.9526 | 86.8961 | 62 | 28 |
| EXP-03 identity 2.5 | 84.2072 | 86.3597 | 83.0848 | 40 | 29 |
| EXP-04 cycle 7.5 + lr_D 1e-4 | 85.3515 | 91.2862 | 87.0166 | 24 | 3 |
| EXP-05 (this run) | 81.0700 | 85.0933 | 83.7819 | 38 | 6 |

Internal metrics for this run: {"fid_A2B": 81.07001850908875, "kid_mean_A2B": 0.017564932804830753, "precision_A2B": 0.5433333516120911, "recall_A2B": 0.3557828962802887, "cycle_l1_A2B": 0.0806734320273002, "content_cosine_mean_A2B": 0.81468337032289, "fid_B2A": 85.09325794138476, "kid_mean_B2A": 0.012560902460025508, "precision_B2A": 0.3604717254638672, "recall_B2A": 0.5366666913032532, "cycle_l1_B2A": 0.10182560819832905, "content_cosine_mean_B2A": 0.7524819223480117}

Nothing was uploaded to Kaggle. No further experiment was started.

Protected historical artifacts (539 items from `task3_gan/parth/exp05_control/protected_hashes_pre_exp05.json`): PASS
