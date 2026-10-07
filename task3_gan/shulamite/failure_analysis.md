# Task 3: CycleGAN failure analysis — final selected run2

Checkpoint reviewed: `run2/checkpoints/cyclegan_epoch_95.pt`. The cases below use the
selected epoch-95 translations and the run2 evaluation metrics. They describe observed
failure modes rather than claiming that the entire run failed.

## Case 1 — Photo-to-Monet color and texture artifacts

- Output: `run2/outputs/photo_to_monet_epoch_95.jpg`
- Failure type: excessive texture, color streaking, and softened boundaries.
- Observation: The output preserves the broad water, shoreline, and boat-like layout,
  but the image is dominated by blue-toned painterly streaks. Fine structures are
  difficult to distinguish and some regions appear washed out or blotchy. The style
  transfer is recognizable, but the texture transformation is stronger than the
  content-preservation objective should ideally allow. This is consistent with the
  photo-to-Monet content cosine of 0.750 and LPIPS of 0.394.

## Case 2 — Monet-to-photo output is not fully photographic

- Output: `run2/outputs/monet_to_photo_epoch_95.jpg`
- Failure type: oversmoothing and incomplete realism.
- Observation: The output retains the sky, horizon, trees, and foreground arrangement,
  but small objects and edges remain soft. Some parts look like a stylized painting
  with photographic colors rather than a naturally captured photograph. The direction
  has the lower FID (95.87) and higher precision (0.730), but its recall is only 0.367,
  indicating that it may produce convincing images from a narrower portion of the photo
  distribution instead of representing the full target-domain variety.

## Case 3 — Directional asymmetry and incomplete cycle preservation

- Failure type: asymmetric translation quality and residual reconstruction error.
- Observation: Run2 performs differently in the two directions. The Monet-to-photo
  cycle L1 is lower at 0.0628, while photo-to-Monet-to-photo is 0.0677. At the same
  time, photo-to-Monet has higher recall (0.630) but lower precision (0.443), whereas
  Monet-to-photo has higher precision but lower recall. This means the model can cover
  more Monet-like variation in one direction but produces a narrower, more conservative
  photo-like distribution in the other. The nonzero cycle errors and the visible soft
  edges show that cycle consistency constrains the mapping but does not guarantee exact
  object geometry or artifact-free reconstruction.

## Proposed testable fixes

1. Use EMA generator weights at inference and compare them with the epoch-95 checkpoint
   on the same fixed 30 samples; test whether EMA reduces streaking and edge noise.
2. Apply DiffAugment to discriminator inputs to reduce discriminator overfitting to the
   300-image Monet domain; compare FID, KID, precision/recall, and human artifact scores.
3. Increase or rebalance the Monet-domain training data if possible, then evaluate both
   directions separately instead of selecting one checkpoint only by mean FID.
