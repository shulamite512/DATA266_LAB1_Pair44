# Task 3: CycleGAN qualitative analysis — FINAL RUN2

Final selected run: `run2`. Training completed 100 epochs, and the selected
checkpoint was `run2/checkpoints/cyclegan_epoch_95.pt`, chosen by class-notebook FID.
The final outputs are `run2/outputs/monet_to_photo_epoch_95.jpg` and
`run2/outputs/photo_to_monet_epoch_95.jpg`; the run metrics are in
`run2/full_metrics_report.csv` and `run2/outputs/metrics.json`.

## Configuration

Run2 used a constant learning rate for the first 50 epochs followed by linear decay,
resize-286/random-crop-256 augmentation, random horizontal flips, bf16 autocast, and
four data-loader workers. The model contains 28,285,832 parameters and trained on an
NVIDIA RTX 4090 for 5.76 hours. Throughput was 33.96 image pairs/sec and peak GPU
memory was 9,011 MB.

## Architecture and hyperparameter justification

The implementation uses two independent ResNet-style generators, one for
Monet-to-photo and one for photo-to-Monet, plus two 70-by-70 PatchGAN-style
discriminators. Each generator starts with a 7-by-7 reflection-padded convolution,
uses two stride-2 convolutions to encode the image, applies nine residual blocks at
256 channels, and then upsamples with two transposed convolutions before a tanh output.
The residual blocks help preserve scene layout while allowing the generator to change
texture and color. Instance normalization is appropriate for unpaired style transfer
because it reduces instance-specific contrast statistics without requiring paired
examples. The patch discriminator evaluates local realism, which is important for
brush texture, edges, and small photographic details.

The adversarial objective uses least-squares MSE loss for smoother generator feedback.
Cycle-consistency uses L1 loss with weight 10.0 so the translation remains reversible
and does not discard the input scene; identity loss uses weight 5.0 to discourage
unnecessary color and structure changes when an image is already near the target domain.
The Adam optimizers use learning rate 2e-4 and betas (0.5, 0.999), a common stable
choice for GAN updates. Batch size is 8, the replay buffer stores 50 generated images
for discriminator updates, and seed 266 fixes the experiment split and random setup.
Run2 keeps the learning rate constant through epoch 50, then linearly decays it to
zero, giving the generators time to learn the mapping before fine-tuning it.

## Visual quality and content preservation

The epoch-95 Monet-to-photo output preserves the broad landscape arrangement, including
the sky, horizon, trees, and foreground, while producing more photographic structure.
The photo-to-Monet output retains the water, shoreline, and boat-like forms but changes
the image toward blue-toned painterly strokes and softened boundaries. This shows that
both generators learned a domain-style transformation rather than replacing the scene
with an unrelated image.

The outputs are not fully realistic. Fine objects remain soft, some regions are
blotchy or over-smoothed, and the photo-to-Monet result has strong texture and color
streaks. The measured content cosine similarities are 0.750 for photo-to-Monet and
0.762 for Monet-to-photo, so global structure is retained but exact content is not
perfectly preserved.

## Quantitative interpretation

The final submission-style FID/MiFID are 96.01 and 0.4085. In the two translation
directions, class-notebook FID is 96.15 for photo-to-Monet and 95.87 for Monet-to-photo;
the KID means are 0.0102 and 0.0163. Run2 therefore improves the reported FID over the
earlier run1 values of 101.81 and 106.48.

Generative precision/recall is asymmetric. Photo-to-Monet scores 0.443/0.630, while
Monet-to-photo scores 0.730/0.367. Monet-to-photo has higher precision but lower recall,
suggesting that its outputs are closer to a narrower portion of the photo distribution.
Monet-to-photo also has the slightly lower class-notebook FID (95.87 versus 96.15), so
it is the better direction under FID for run2. This is not a universal win: its KID is
higher (0.0163 versus 0.0102) and its recall is lower, so the direction comparison
depends on which quality property is prioritized.
Cycle L1 distances are 0.0677 for photo-to-Monet-to-photo and 0.0628 for
Monet-to-photo-to-Monet. These values indicate improved cycle reconstruction relative
to run1, while the visual outputs still show residual artifacts.

The final Kaggle leaderboard rank was **17**, which earns **9 points** under the lab
rubric's 11–20 rank bracket. This rank is reported separately from the local FID and
MiFID measurements because the leaderboard result reflects the competition submission.

The 30-sample human audit was completed by two raters across style, content
preservation, and artifacts, giving 90 ratings per rater. The overall mean audit score
was **4.10/5**. Exact agreement was **3.3%**, and the overall Cohen's kappa was
**-0.194**. The low kappa indicates substantial disagreement between the raters, so
the mean score should be interpreted with caution. Dimension means were 4.27/5 for
style, 3.52/5 for content preservation, and 4.52/5 for artifacts.

## Training behaviour and stability

Across the full run2 training history, generator loss decreases from 7.398 at epoch 1
to 2.666 at epoch 100. Cycle loss decreases from 0.4531 to 0.1095, identity loss
decreases from 0.4148 to 0.0673, and discriminator loss ends at 0.2275 after starting
at 0.4273. The selected epoch-95 checkpoint itself has generator/cycle/identity/
discriminator losses of 2.6836/0.1117/0.0693/0.2329. The largest recorded gradient
norm is 45.91 and the NaN count is zero, so the mixed-precision run remained
numerically stable. The learning-rate decay after epoch 50 coincides with continued
reductions in cycle and identity loss.

## Strengths and limitations

The main strength of run2 is the improved distributional score and lower cycle error
after longer training, while both directions preserve scene-level structure. The
limitations are the small Monet domain of 300 images, the much larger photo domain,
soft fine detail, and visible painterly or blotchy artifacts. FID and KID are
distributional metrics and do not prove that every individual translation is good.
The human audit is now recorded, but its low inter-rater agreement is a limitation.
The raters should be described as independent reviewers, and the negative kappa should
not be presented as evidence of strong evaluator consistency.

## Proposed next experiment

Because run2 is the final selected run for this report, the next experiment should test
stability and visual diversity rather than simply repeat its training schedule. Run3's
EMA variant would test whether averaging generator weights produces cleaner and more
consistent translations at inference. A separate run4-style DiffAugment experiment
would test whether stronger discriminator-side augmentation reduces overfitting to the
small 300-image Monet domain. Both variants should use the same fixed evaluation
images and compare FID, KID, precision/recall, cycle L1, content cosine similarity,
and a two-rater human audit. The best checkpoint should be selected before inspecting
test outputs, and any Kaggle submission should be generated directly from that
checkpoint.
