# Task 3 data: Monet paintings and photos

The images are not committed. Download the data from the Kaggle class competition linked in the lab handout and place the two JPEG folders here:

```text
task3_gan/data/
  monet_jpg/   300 Monet paintings, 256×256 RGB (domain A)
  photo_jpg/   7,038 photos, 256×256 RGB (domain B)
```

The EXP-05 run found exactly these counts, with all images valid (run manifest `reproducibility/manifests/task3_parth_exp05_5090_baseline_control_20260930-145628.json`, `data` block).

This folder must exist for `task3_gan/parth/src/Task3_Parth_CycleGAN_Final.ipynb` to locate the repository root.
