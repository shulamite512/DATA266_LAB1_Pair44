# Task 3 data: Monet paintings and photos

The images are not committed. The team's shared [datasets.zip on Google Drive](https://drive.google.com/file/d/1Elz-CSlEQwb5e6X7R8mPWsrb1negSjpQ/view?usp=drive_link) (read access, see the root `README.md`) unpacks the two JPEG folders here. They can also be downloaded from the Kaggle class competition linked in the lab handout.

```text
task3_gan/data/
  monet_jpg/   300 Monet paintings, 256×256 RGB (domain A)
  photo_jpg/   7,038 photos, 256×256 RGB (domain B)
```

Parth's EXP-05 run found exactly these counts, with all images valid (run manifest `reproducibility/manifests/task3_parth_exp05_5090_baseline_control_20260930-145628.json`, `data` block).

This folder must exist for `task3_gan/parth/src/Task3_Parth_CycleGAN_Final.ipynb` to locate the repository root.
