# Task 2: Failure / Error Analysis (Parth)

Source files:

| File | Contents |
|---|---|
| `outputs/misclassified_examples.csv` | every error from every evaluated model |
| `outputs/misclassified_examples_<model>.csv` | errors for one model, most confident first |
| `outputs/error_review_candidates_<model>.csv` | the 20 errors sampled for manual review |
| `outputs/slice_metrics_<model>.csv` | macro-F1 and error rate per slice |
| `outputs/disagreements_logreg_vs_<model>.csv` | reviews where exactly one of the two models is right |
| `outputs/mcnemar_results.csv` | paired significance test against the baseline |

## Manual review of 20 errors (lab requirement 2.2.4)

Model reviewed: `neural_cnn` (Experimental 2, TextCNN), checkpoint `checkpoints/best_neural_cnn.pt` (SHA-256 `7f34ff61f82c63b78d93412792a6b14e22c70f55ffc5f57af0ec650d02f61814`), evaluation log `logs/eval_neural_cnn_20260925_232228.log`.

The 20 rows are copied verbatim from `outputs/error_review_candidates_neural_cnn.csv`, which `evaluate.py` selected with seed 8503: 5 confident false positives, 5 confident false negatives, 5 near-threshold errors (P(positive) closest to 0.5), and 5 errors from `has_contrast`, the slice with the highest error rate for this model (0.0688). Threshold 0.5; label 0 = negative, 1 = positive. Review text is the raw review after undoing CSV escapes (`\n` -> space), which is what `evaluate.py` stores.

Each case below has a manual error type and one proposed testable fix, framed as an experiment rather than a claimed result. The category counts are in the summary after case 20.

| # | Bucket | Review ID | True | Pred | P(positive) | Confidence |
|---|---|---|---|---|---|---|
| 1 | confident FP | `test_29330` | 0 | 1 | 0.9999753 | 0.9999753 |
| 2 | confident FP | `test_5752` | 0 | 1 | 0.99915254 | 0.99915254 |
| 3 | confident FP | `test_18122` | 0 | 1 | 0.9990633 | 0.9990633 |
| 4 | confident FP | `test_37771` | 0 | 1 | 0.99887043 | 0.99887043 |
| 5 | confident FP | `test_12480` | 0 | 1 | 0.9980787 | 0.9980787 |
| 6 | confident FN | `test_30793` | 1 | 0 | 0.00015882253 | 0.99984115 |
| 7 | confident FN | `test_22807` | 1 | 0 | 0.00033526638 | 0.9996647 |
| 8 | confident FN | `test_33573` | 1 | 0 | 0.00039222906 | 0.99960774 |
| 9 | confident FN | `test_9438` | 1 | 0 | 0.00048468274 | 0.9995153 |
| 10 | confident FN | `test_11398` | 1 | 0 | 0.0006829828 | 0.999317 |
| 11 | near threshold | `test_34487` | 0 | 1 | 0.50012237 | 0.50012237 |
| 12 | near threshold | `test_32335` | 0 | 1 | 0.50020707 | 0.50020707 |
| 13 | near threshold | `test_1372` | 0 | 1 | 0.5003647 | 0.5003647 |
| 14 | near threshold | `test_32304` | 1 | 0 | 0.49954125 | 0.5004587 |
| 15 | near threshold | `test_23084` | 0 | 1 | 0.50064075 | 0.50064075 |
| 16 | slice: has_contrast | `test_10621` | 1 | 0 | 0.32625556 | 0.67374444 |
| 17 | slice: has_contrast | `test_8361` | 1 | 0 | 0.4235403 | 0.5764597 |
| 18 | slice: has_contrast | `test_20227` | 0 | 1 | 0.7999777 | 0.7999777 |
| 19 | slice: has_contrast | `test_8938` | 0 | 1 | 0.84852064 | 0.84852064 |
| 20 | slice: has_contrast | `test_34683` | 0 | 1 | 0.9741432 | 0.9741432 |

---

### Case 1: confident FP

| Field | Value |
|---|---|
| Review ID | `test_29330` |
| Review bucket | `confident_false_positive` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.9999753 |
| Confidence | 0.9999753 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 27 / 18 |
| Length bucket | short |
| has_negation / has_contrast | False / False |

```text
Wow love the place and everything is very clean and new!  Great place to come and relax worth a try!  Cheers,  Eric Van Nguyen Visited April 2012
```

**Manual error type:** Possible rating-text mismatch. The text is positive throughout ("love the place", "Great place to come and relax"), and all three models predict positive with P > 0.999.

**Proposed testable fix:** Manually audit the confident errors on which all three models agree, flag suspected rating-text mismatches, and report test accuracy with and without the flagged reviews.

---

### Case 2: confident FP

| Field | Value |
|---|---|
| Review ID | `test_5752` |
| Review bucket | `confident_false_positive` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.99915254 |
| Confidence | 0.99915254 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 417 / 216 |
| Length bucket | long |
| has_negation / has_contrast | True / True |

```text
NOTE:  This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes.  See update below. Complaining I can't find a good meatball sub in Phoenix, I was referred to Santisi Brothers.  I was told that the meatballs are still made by the brothers' mother, so I was intrigued.  Santisi Brothers did not disappoint.  That sub was so freaking good!!!  I got the 1/2 sandwich, which included two meatballs smothered under mozzarella and marinara on a toasted baguette-type roll.  Amazing.  The bread was soft inside with just the right crust, the meatballs were sizable (that's what she said!), and they did not scrimp on the mozzarella.  Since this is their only location in the valley, we will be sure to be back.  (It's right off I-17, close to the 101, so that's not hard.) If you like sports, go!  This place has each wall covered in TVs - huge to 13"", they don't spare a bit of viewing area.  The only reason I gave 4 stars instead of 5 is because the seating leaves a bit to be desired.  We sat at a hightop table with stools.  Next are a row of tables with nicer chairs (with backs), and the row closest to the TVs - who wants to sit there?  Music - we could hear each other, but agreed a table of four might have to shout.  Our server, Ariel, was very enjoyable, nice, and attentive. Big beer and bar selection.  Husband's Kiltlifter was in a huge beer mug and he was happy to see a few IPAs.  I had a John Daly that was verrry tasty.  It's one of those sneaky super-yummy drinks...  Sweet iced tea, lemonade, and vodka - I'll be stealing that recipe this summer! Husband had the turkey club, which was large enough to be funny to watch him eat, but he said it was unremarkable.  Side of fries were also really, really good.  Perfect amount of potato, crispiness, and salt.  We also had garlic knots, which are served with marinara and pretty tasty, but left me wishing there was a little ball of mozzarella in the middle.  House salads were... house salads.  Nothing special there. To review: great for sports (I'm sure they can spare a TV for your game), may want to go early for good seats, good bar, and some great food.  EAT/DRINK:  Meatball sub, french fries, and a John Daly.  All wonderful. PS - Kitchen is open till 11pm Sun-Thu, midnight Fri-Sat.
```

**Manual error type:** Temporal sentiment shift. A former 4-star review keeps its positive body; the negative verdict is only the opening note ("food quality and ESPECIALLY customer service have gone down the tubes"), which the label follows.

**Proposed testable fix:** Add an update-marker feature (reviews starting with NOTE, EDIT or UPDATE) or score the first sentence separately, and compare error rates on reviews containing these markers.

---

### Case 3: confident FP

| Field | Value |
|---|---|
| Review ID | `test_18122` |
| Review bucket | `confident_false_positive` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.9990633 |
| Confidence | 0.9990633 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 73 / 38 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
20 years for me and sad to see them go.  Sadder still to see a great traditional Japanese sushi restaurant end, and not carry on the tradition.  We would go here for consistently great fresh fish and great service.  They have also trained many young sushi chefs through the years here in Phoenix.  Great Japanese food isn't about fancy rolls and combinations, but about great fresh ingredients.  END OF AN ERA.  I agree.
```

**Manual error type:** Contextual sentiment with an ambiguous label. The writer praises a sushi restaurant ("consistently great fresh fish and great service") and is sad that it is closing; the negative label reflects the closure, not the food.

**Proposed testable fix:** Filter test reviews that mention a closure ("closed", "sad to see them go", "end of an era"), audit their labels, and measure each model's error rate on that subset.

---

### Case 4: confident FP

| Field | Value |
|---|---|
| Review ID | `test_37771` |
| Review bucket | `confident_false_positive` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.99887043 |
| Confidence | 0.99887043 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 78 / 31 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
I have to say I love cafe rio! Their food is simple yet delicious, the service is great as well as consistent, and my appetite and tummy are always satisfied. However I've been to this location 3 different times and have not experienced any of these.. Even though it may be convenient for many of us here in henderson, please listen to the other reviews and DO NOT EAT AT THIS LOCATION ! New management is highly needed
```

**Manual error type:** Mixed sentiment with a target shift. Love for the chain ("I love cafe rio!") comes first and the verdict on this location ("DO NOT EAT AT THIS LOCATION") after "However", a word the stopword list removes.

**Proposed testable fix:** Retrain with contrast words kept (remove "but", "however", "though" and "yet" from the stopword list) and compare error rates on the `has_contrast` slice.

---

### Case 5: confident FP

| Field | Value |
|---|---|
| Review ID | `test_12480` |
| Review bucket | `confident_false_positive` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.9980787 |
| Confidence | 0.9980787 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 28 / 11 |
| Length bucket | short |
| has_negation / has_contrast | False / True |

```text
my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!
```

**Manual error type:** Possible rating-text mismatch. One mild complaint ("a little on the small side for $10") against "good", "great" and "awesome"; all three models predict positive.

**Proposed testable fix:** Include in the label audit of case 1 and report accuracy with and without flagged reviews.

---

### Case 6: confident FN

| Field | Value |
|---|---|
| Review ID | `test_30793` |
| Review bucket | `confident_false_negative` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.00015882253 |
| Confidence | 0.99984115 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 101 / 38 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
This place is so much better since they changed owners.  My wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they ""made more"".  It was horrible.  Now its much better.  The staff are very friendly, they treat their customers very well and I have nothing but positive things to now say about this place.  Its much better with the new owners.
```

**Manual error type:** Temporal sentiment shift. Most of the text describes the old owners ("terrible", "horrible", "waited forever"); the positive verdict is "Now its much better" under the new owners.

**Proposed testable fix:** Compare against an order-aware model (for example a BiLSTM) on a subset of reviews with temporal markers ("now", "since", "used to", "new owners") and measure the error rate there.

---

### Case 7: confident FN

| Field | Value |
|---|---|
| Review ID | `test_22807` |
| Review bucket | `confident_false_negative` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.00033526638 |
| Confidence | 0.9996647 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 99 / 47 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
EDIT: They really did change the service up since I last posted this.  Horrible service.  Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'  Sorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.
```

**Manual error type:** Possible rating-text mismatch after an edit. The text is negative ("Horrible service", an altercation with a server) and all three models predict negative; the positive label is not supported by the text as written.

**Proposed testable fix:** Audit the labels of reviews that start with EDIT or UPDATE and report accuracy with and without suspected mismatches.

---

### Case 8: confident FN

| Field | Value |
|---|---|
| Review ID | `test_33573` |
| Review bucket | `confident_false_negative` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.00039222906 |
| Confidence | 0.99960774 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 23 / 12 |
| Length bucket | short |
| has_negation / has_contrast | False / False |

```text
For being a DUMP, should expect much more.  Flys, stink, garbage, dirt, and everything that comes with.  Salt River... Keepin it real dumpy!
```

**Manual error type:** Irony or unusual phrasing. "For being a DUMP" and "Keepin it real dumpy!" read as affectionate irony about a dive, while the words are strongly negative ("stink", "garbage", "dirt"). It could also be label noise.

**Proposed testable fix:** Hand-label the 1,355 reviews all three models get wrong as irony, label noise or genuine error, and report the share of each to decide which fix to pursue.

---

### Case 9: confident FN

| Field | Value |
|---|---|
| Review ID | `test_9438` |
| Review bucket | `confident_false_negative` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.00048468274 |
| Confidence | 0.9995153 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 31 / 14 |
| Length bucket | short |
| has_negation / has_contrast | True / False |

```text
All they need to do is add a Forever 21, and then I will never have to go to ghetto Fiesta Mall or drive through god-awful traffic to the Chandler Mall!
```

**Manual error type:** Sentiment target shift. The negative words describe other places ("ghetto Fiesta Mall", "god-awful traffic"); the positive view of this mall is implied by wanting to shop only there.

**Proposed testable fix:** Build a small hand-annotated subset of reviews whose negative words refer to other businesses, and compare the three models with an order-aware model on it.

---

### Case 10: confident FN

| Field | Value |
|---|---|
| Review ID | `test_11398` |
| Review bucket | `confident_false_negative` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.0006829828 |
| Confidence | 0.999317 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 36 / 19 |
| Length bucket | short |
| has_negation / has_contrast | False / True |

```text
Drinks & appetizers, that's my advice. I like their beet salad..but the green beans are raw (or always undercooked?) and they make the salad annoying to chew/eat every time, otherwise that salad is one to try.
```

**Manual error type:** Mixed sentiment. A mild recommendation ("I like their beet salad", "one to try") is outweighed by complaint words ("raw", "undercooked", "annoying"), and the "but" between them is removed by preprocessing.

**Proposed testable fix:** Retrain with sentence-level pooling (score each sentence, then average) and compare the error rate on short reviews in the `has_contrast` slice.

---

### Case 11: near threshold

| Field | Value |
|---|---|
| Review ID | `test_34487` |
| Review bucket | `near_threshold` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.50012237 |
| Confidence | 0.50012237 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 106 / 55 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
After reading the other yelp reviews I was looking forward to trying this place.  I generally like asian food. Unfortunately it didn't deliver  I had the chicken teriyaki bowl on brown rice.  The flavors didn't mix well.  The teriyaki seemed more like straight soy sauce and the carrots were raw.  I also ordered the lettuce wraps (I was hungry) which were slightly better.  The sauce was flavorful but the chicken filling seemed like they put it in the blender for too long.  There aren't a lot of great food options in the Charlotte airport but unless you are craving Asian stir fry, I would look elsewhere
```

**Manual error type:** Negation-heavy negative review, TextCNN-only error. The review is clearly negative ("didn't deliver", "didn't mix well", "I would look elsewhere"); the logistic regression (0.327) and MeanPoolMLP (0.208) are right, and the TextCNN sits at 0.5001.

**Proposed testable fix:** Mark negation scope in preprocessing (prefix tokens after "not" or "didnt" with NOT_ until the next punctuation mark) and compare error rates on the `has_negation` slice.

---

### Case 12: near threshold

| Field | Value |
|---|---|
| Review ID | `test_32335` |
| Review bucket | `near_threshold` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.50020707 |
| Confidence | 0.50020707 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 121 / 45 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
Even though the restaurant was empty, they hostess let us know that the only tables available were the outdoor tables by the waterfall.  They had a minimum of $150 pp, so we figured we might as well splurge on a meal and get the tasting menu.  The service was great, as were the drinks.  The food definitely did not meet my expectations.  Perhaps the fish in Houston is fresher, but the food at Mizumi just didn't do it for me.    They asked if we had any food allergies, and I told them I was allergic to peanuts.  Since I couldn't have the dessert that was part of the tasting menu, they brought out sorbet for me (and charged me for it!!).
```

**Manual error type:** Mixed sentiment, TextCNN-only error. Praise for the service and drinks against "The food definitely did not meet my expectations" and an unexpected charge; the other two models are confident and right (0.128 and 0.189).

**Proposed testable fix:** Average the logistic regression and TextCNN probabilities (weights fixed on the validation set) and measure accuracy among reviews where the TextCNN's P(positive) is between 0.4 and 0.6.

---

### Case 13: near threshold

| Field | Value |
|---|---|
| Review ID | `test_1372` |
| Review bucket | `near_threshold` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.5003647 |
| Confidence | 0.5003647 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 37 / 19 |
| Length bucket | short |
| has_negation / has_contrast | False / False |

```text
I work in the industry; needed to get nose pad replacements for expensive wire frames I wear everyday; they charged me 24 dollars with tax;  I could get them for free at most optical shops;  enough said.
```

**Manual error type:** Implicit sentiment. There are no explicit sentiment words; the complaint is a price comparison ("they charged me 24 dollars with tax; I could get them for free at most optical shops"). The logistic regression is right (0.348) and both neural models sit just above 0.5.

**Proposed testable fix:** Add the baseline's TF-IDF score as an extra input to the TextCNN and compare error rates on a hand-labelled subset of reviews that contain no sentiment-lexicon words.

---

### Case 14: near threshold

| Field | Value |
|---|---|
| Review ID | `test_32304` |
| Review bucket | `near_threshold` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.49954125 |
| Confidence | 0.5004587 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 145 / 66 |
| Length bucket | medium |
| has_negation / has_contrast | True / False |

```text
Wife and I were able to stop by, and try a few things. We had their chicken and waffles, they were OK! I kept on saying to my wife, man I have had this before. chicken reminded me of another time!! Finally hour in a half later, it struck me!!! Chicken tasted just like Swanson frozen tv dinner!!! I grew up on that! So you can take that any way you want! You need to eat the pretzels with no salt. Comes with way to much! The pretzels rolls were probably the best dinner rolls I have ever had. Since I was driving, it was a no beer night! Would like to come back and try some brats and beer. Outdoor patio is the best, on cool nights they have a great fire pit! Plenty of room, Until next time and a second visit!  Later,
```

**Manual error type:** Mixed sentiment with weak overall polarity. "they were OK!" and a comparison to a "Swanson frozen tv dinner", then "the best dinner rolls" and "Would like to come back"; the logistic regression (0.835) and MeanPoolMLP (0.589) are right.

**Proposed testable fix:** Tune the decision threshold on the validation set instead of fixing it at 0.5, and compare test accuracy among reviews with P(positive) between 0.4 and 0.6.

---

### Case 15: near threshold

| Field | Value |
|---|---|
| Review ID | `test_23084` |
| Review bucket | `near_threshold` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.50064075 |
| Confidence | 0.50064075 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 84 / 38 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
Last week my neighbors and my husband and I enjoyed a few quick beers at Jacks???, then went to have a bite at Yo Rita;s, Service was a little off, thought we were upsetting the waitress? The Taco's were really good, but when all the chips and salsa were gone I really do not think a Taco is quite enough, I know, we all say less food but if we were informed to order a salad or app we sure would of. Just sayin.
```

**Manual error type:** Mixed sentiment with an understated complaint. "Service was a little off" and "I really do not think a Taco is quite enough" against "The Taco's were really good"; the other two models are right (0.379 and 0.143).

**Proposed testable fix:** Retrain with contrast words kept, as in case 4, and measure the error rate among `has_contrast` reviews whose P(positive) is between 0.4 and 0.6.

---

### Case 16: slice: has_contrast

| Field | Value |
|---|---|
| Review ID | `test_10621` |
| Review bucket | `slice:has_contrast` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.32625556 |
| Confidence | 0.67374444 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 58 / 29 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
I really enjoyed myself here! Went with my girlfriends and they waived the cover but I'm sure they do that for groups of girls anyway. We danced the night away and the crowd was drama-free! Drinks were expensive but we didn't exactly pay for any so I can honestly say my low-cost night at Tao was just lovely.
```

**Manual error type:** Mixed sentiment in concessive clauses. Negative words appear in clauses the writer dismisses ("Drinks were expensive but we didn't exactly pay for any") inside a clearly positive review ("I really enjoyed myself here!", "just lovely"); the logistic regression is right (0.789).

**Proposed testable fix:** Add explicit contrast-clause features: split each review at contrast words before stopword removal, pool the clauses separately, and compare `has_contrast` error rates.

---

### Case 17: slice: has_contrast

| Field | Value |
|---|---|
| Review ID | `test_8361` |
| Review bucket | `slice:has_contrast` |
| True label | 1 (positive) |
| Predicted label | 0 (negative) |
| P(positive) | 0.4235403 |
| Confidence | 0.5764597 |
| Automatic error category | `false_negative` |
| Raw words / processed tokens | 121 / 57 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
How disappointed I was to hear that the best restaurant in Vegas closed its doors forever.  When I ask myself, what is the best thing I've ever eaten?  I immediately answer...everything from Rosemary's.  I've purchased the Rosemary's cookbook ""Food of Love"" and tried to duplicate some of the recipes.  A feew successfully, others, not so.  But knowing how much time, energy, and detail went into each and every one of their dishes, I completely understand why it was so amazing.  What I cannot understand is why they closed.   Until Michael and Wendy grace Las Vegas with their talents once more, I will begrudgingly patronize one of the overpriced and played out celebrity chef locations when I visit Vegas.  C'est la vie.
```

**Manual error type:** Contextual sentiment with an ambiguous label. "How disappointed I was" refers to the closure of a restaurant the writer calls the best in Vegas; this review is labeled positive, while case 3, also praise for a closing restaurant, is labeled negative.

**Proposed testable fix:** Use the closure subset from case 3, label each review twice (sentiment toward the restaurant, sentiment about the event), and measure how each model scores against both labelings.

---

### Case 18: slice: has_contrast

| Field | Value |
|---|---|
| Review ID | `test_20227` |
| Review bucket | `slice:has_contrast` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.7999777 |
| Confidence | 0.7999777 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 303 / 138 |
| Length bucket | long |
| has_negation / has_contrast | True / True |

```text
I offer up this haiku review...  Drinks on Friday Night Only non-employee there Two Ales and I left.   Let me 'splain...most places have happy hour and there is a crowd. I have been to others and it begins filling up at 4pm. The sign on the door said Arizona's longest happy hour - beginning at 2pm. That's nice, but apparently if you build it, they won't come. Friday night and I was there at 4. Everyone else there was an employee. Really? How can you get happy with yourself? Another guy came in, but he was evening help. Mike, who was very personable told the guy it has been a good day. The other guy said really? How so? He said he had a good lunch, a nice pop...and Mark came in...normally I do draw a crowd, but I must have been off tonight. I went with the recommended Whale Ale. The glass was big and heavy. Mike explained that all of the beers are served cold and the extra thick glass ensures they stay that way. Get there at 11 AM, he said, and it is excellent. The menu looked good-  Mike recommended the fish & chips, which are a favorite of mine, but I was leaning toward the crab reuben. Just sounded different - crab salad, ham and swiss - interesting. I had a second Wale Ale while thinking it over. Instead, nearing 5 pm, the place was still just me - so I requested the check and left.   They have events such as the Saturday night disco night and the Wednesday happy hour. Maybe one of those nights would be nice. The place was cool and the beer was good. I would like to try the food, but I don't want to go it alone. Anyone want to tag along?
```

**Manual error type:** Weakly negative long review. The writer likes the staff, the beer and the place ("The place was cool and the beer was good"); the complaint is that nobody else came, and the verdict is implied by leaving early.

**Proposed testable fix:** Match test reviews to their original star ratings in the 5-class Yelp reviews data and measure error by star level, to test whether errors concentrate in mid-range ratings.

---

### Case 19: slice: has_contrast

| Field | Value |
|---|---|
| Review ID | `test_8938` |
| Review bucket | `slice:has_contrast` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.84852064 |
| Confidence | 0.84852064 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 347 / 151 |
| Length bucket | long |
| has_negation / has_contrast | True / True |

```text
I will say that I do go here, though very sporadically. I gave it two stars, but it's because of my tastes, which I will explain in detail.  When I do go, it's always early in the evening. I don't like going to this bar when it's crowded, as it's excessively crowded. It's worth a trip if you're into the scene or are visiting Phoenix and want to experience the scene. This is definitely one of the Valley's landmark bars. If you're going alone and don't mind spending the evening by yourself (i.e. people-watching) it really isn't that bad. If you want to mingle with people, this bar isn't a great atmosphere for that. People generally go in cliques and you'll notice that the bar is full of clique-clusters. It definitely is not one of the friendlier bars in the Valley. It also seems to have an ""instant snob"" effect on people.   Now, for pluses... Amsterdam does have a beautiful outdoor area, which is suitable for smokers. You can not only take your drinks out there, but there is an outdoor bar at which you can get drinks. It's a nice area to have drinks in the spring, fall and early/late summer. Most of the summer it can still top 100 degrees even late at night and winter nights do get chilly. Amsterdam does have a nice interior as well. They also have an extensive, unique drink menu.  Amsterdam does have a different crowd earlier in the evening than it does late nights. If I were to focus solely on the evening atmosphere and if that were my only experience, I could honestly give Amsterdam more stars. The dense crowds aside, the ""pretentious"" environment and ""go to be seen"" atmosphere make it harder to enjoy. I've largely gone to bars based on the general environment and comfort, which is why it's harder for me to make Amsterdam a focus-point on a night out. If I'm in the area and feel like a drink, and it's early in the evening, I might be more inclined to stop in.
```

**Manual error type:** Balanced review whose explicit rating is lost in preprocessing. "I gave it two stars" states the verdict, but the stopword list removes "two" and "but", leaving `gave star`; the rest weighs pros and cons.

**Proposed testable fix:** Keep number words and map "N star(s)" to a rating token, then compare error rates on reviews that state a star count.

---

### Case 20: slice: has_contrast

| Field | Value |
|---|---|
| Review ID | `test_34683` |
| Review bucket | `slice:has_contrast` |
| True label | 0 (negative) |
| Predicted label | 1 (positive) |
| P(positive) | 0.9741432 |
| Confidence | 0.9741432 |
| Automatic error category | `false_positive` |
| Raw words / processed tokens | 106 / 56 |
| Length bucket | medium |
| has_negation / has_contrast | True / True |

```text
Everyday Noodles has a lot going for it:  Fast, friendly service, attractive decorations, impeccably clean just renovated interior, reasonable prices and convenient location.  They only have one issue, bland, plain food.  The textures and temperatures were right, but between us, my dinner party tried about a dozen appetizers and entrees and the flavors were consistently very simple and tame.    Plenty of people don't like intense flavors, and this would be perfect for them.  When they go, they'll only spot me at Everyday Noodles if I'm walking by it, on my way to How Lee or the Raman Bar across the street for more flavorful, adventurous fair.
```

**Manual error type:** Strong positive words inside a negative review. Five praised aspects ("Fast, friendly service", "impeccably clean") against one decisive complaint ("bland, plain food") and a closing line saying the writer eats elsewhere.

**Proposed testable fix:** Add a separate representation of the final sentence (the verdict) to the TextCNN and compare errors on reviews whose last sentence has the opposite polarity to the rest.

---

## Summary of the manual review

| Error category (primary) | Cases | Count |
|---|---|---|
| Mixed sentiment / contrast | 4, 10, 12, 14, 15, 16, 20 | 7 |
| Possible rating-text mismatch | 1, 5, 7 | 3 |
| Temporal sentiment shift | 2, 6 | 2 |
| Contextual sentiment, ambiguous label (closed restaurant) | 3, 17 | 2 |
| Weak or balanced long review | 18, 19 | 2 |
| Irony / unusual phrasing | 8 | 1 |
| Sentiment target shift | 9 | 1 |
| Implicit sentiment | 13 | 1 |
| Negation-heavy, TextCNN-only | 11 | 1 |
| **Total** | | **20** |

Mixed sentiment is the most common category, but the sample is not random: 5 of the 20 cases were drawn from the `has_contrast` slice by design. Five cases (1, 3, 5, 7 and 17) point to labels the text does not clearly support, and all five except case 17 are errors shared by all three models.

## 1. False positives

Twelve of the 20 reviewed errors are false positives. The five confident ones (cases 1 to 5) all have P(positive) above 0.998, and all three models make the same mistake. Cases 1 and 5 read as positive throughout, so they look like rating-text mismatches rather than model errors; cases 2, 3 and 4 contain long stretches of genuine praise (an old 4-star review, praise for a closing restaurant, love for the chain) with the negative verdict confined to one or two sentences.

Across the whole test set the TextCNN makes more false positives than false negatives (1,297 against 1,063), and its extra errors relative to the baseline lean the same way: 59.7% of the 834 reviews the baseline gets right and the TextCNN gets wrong are negative reviews predicted positive.

## 2. False negatives

The five confident false negatives (cases 6 to 10) are positive reviews predicted negative with P(positive) below 0.001, again with all three models agreeing. They are full of negative words that are not the writer's verdict: complaints about the previous owners (case 6), about other malls and traffic (case 9), an ironic "DUMP" (case 8) and a side complaint about green beans (case 10). Case 7 reads as clearly negative, so its positive label is a likely rating-text mismatch.

## 3. Short reviews

Slice `length_short` (<= 50 raw words): n = 9,287, macro-F1 0.9298, error rate 0.0672 for `neural_cnn` (from `outputs/slice_metrics_neural_cnn.csv`).

Short reviews have a slightly higher TextCNN error rate (0.0672) than medium (0.0610) and long (0.0561) reviews, although short reviews are also 59.9% positive, so part of the difference is class mix. Six of the 20 cases are short (1, 5, 8, 9, 10 and 13). With few words, one strongly worded phrase can decide the prediction ("DUMP" in case 8), and an implicit complaint with no sentiment words (case 13) leaves little for any model to use.

## 4. Long reviews

Slice `length_long` (>= 300 raw words): n = 3,336, macro-F1 0.9372, error rate 0.0561 for `neural_cnn`. The neural models only see the first 256 processed tokens; 1.42% of test reviews are longer than that (`outputs/eda_summary.json`).

Long reviews have the lowest error rate of the three length buckets for every model, so length alone is not the main problem. Truncation is a separate question: on the 540 test reviews longer than 256 processed tokens, the TextCNN's error rate is 8.89% against 6.17% on the rest, the MeanPoolMLP's 7.59% against 6.86%, and the logistic regression, which sees the whole review, 5.19% against 5.37%. That pattern makes truncation a plausible cost for the neural models, but 540 reviews is a small sample and no run tested a longer limit. None of the 20 reviewed cases was truncated (the longest has 216 tokens); the three long cases (2, 18 and 19) are all false positives that mix a lot of positive detail with a negative verdict.

## 5. Sarcasm / ambiguous reviews

Case 8 is the only clear candidate for irony ("Keepin it real dumpy!" in a positive review). Cases 1, 5 and 7 look like rating-text mismatches, because the text supports the models' prediction. Cases 3 and 17 are genuinely ambiguous: both praise a restaurant that is closing, but one is labeled negative and the other positive, so the label depends on something the text does not settle. Cases 18 and 19 are balanced long reviews whose text is at most weakly negative; case 19 even states "two stars", which preprocessing reduces to `gave star`.

## 6. Reviews with mixed sentiment

Slice `has_contrast` is a keyword proxy (but, however, although, though, except, yet, unfortunately): n = 22,770, macro-F1 0.9306, error rate 0.0688 for `neural_cnn`, versus 0.0521 on `no_contrast`.

Mixed sentiment is the most common category in the manual review (7 of 20), and 15 of the 20 cases contain a contrast word, partly because 5 cases were selected from the `has_contrast` slice. The typical pattern is a review with real praise and a decisive complaint, or the reverse, where the verdict is in one clause (cases 4, 16 and 20). The stopword list removes "but", "however", "though", "yet" and "except" before any model sees the text, so none of the models can use the contrast word to find the decisive clause; that is consistent with `has_contrast` being the worst slice for all three models, but it was not tested.

## 7-9. Baseline vs neural models

From `outputs/mcnemar_results.csv` (same 38,000 test reviews):

| Comparison | logreg right, model wrong | logreg wrong, model right | p-value |
|---|---|---|---|
| logreg vs neural | 947 | 376 | 2.39e-55 |
| logreg vs neural_cnn | 834 | 513 | 2.81e-18 |

The individual reviews are in `outputs/disagreements_logreg_vs_neural.csv` and `outputs/disagreements_logreg_vs_neural_cnn.csv`.

Counted from `outputs/predictions_<model>.csv` (base rates on the full test set: 59.9% contrast, 75.1% negation, 1.4% truncated, median 98 words):

| Reviews | n | Negative reviews (FP for the wrong model) | Contrast | Negation | Truncated | Median words |
|---|---|---|---|---|---|---|
| logreg right, neural_cnn wrong | 834 | 59.7% | 63.7% | 78.5% | 3.2% | 89 |
| logreg wrong, neural_cnn right | 513 | 46.2% | 70.0% | 78.4% | 1.4% | 107 |
| logreg right, neural wrong | 947 | 48.3% | 66.4% | 82.9% | 1.8% | 95 |
| logreg wrong, neural right | 376 | 52.7% | 68.4% | 76.6% | 1.1% | 93 |

The reviews the TextCNN loses to the baseline lean toward false positives and are more than twice as likely to be truncated as an average review, while the reviews it wins are longer and contain contrast words more often. The MeanPoolMLP's losses to the baseline are concentrated in reviews with negation (82.9%), which fits a model that averages away negation scope. Reading a small sample of short rows from both files also turned up non-English reviews (French and Spanish) and reviews that state a star count, which neither kind of model handles reliably; that observation comes from a handful of rows, not a count.

## 10. Agreement across all three models

Counted by joining `outputs/predictions_logreg.csv`, `predictions_neural.csv`, and `predictions_neural_cnn.csv` on `id` (38,000 test reviews). The pairwise totals match `mcnemar_results.csv`.

| Pattern | Count |
|---|---|
| All three models right | 34,512 |
| All three models wrong | 1,355 |
| Only logreg right | 332 |
| Only neural right | 171 |
| Only neural_cnn right | 308 |
| logreg and neural right, neural_cnn wrong | 502 |
| logreg and neural_cnn right, neural wrong | 615 |
| neural and neural_cnn right, logreg wrong | 205 |

The 1,355 reviews that all three models get wrong are 3.6% of the test set. They are roughly balanced between false positives and false negatives (48.3% are positive reviews), they contain contrast words more often than average (67.2% against 59.9%), and they are no more likely to be truncated (1.4%). The models are often confident on them (mean TextCNN confidence 0.807), which is what would be expected if many of these reviews are genuinely mixed or carry labels the text does not support. In the manual review, 13 of the 20 cases are all-model errors, and 4 of those 13 (cases 1, 3, 5 and 7) point to label issues rather than a model failure.

## Proposed fixes and how I would test them

Each fix below is an experiment tied to the error types above; none has been run.

| Fix | Error type it targets | Experiment and metric |
|---|---|---|
| Keep contrast words and intensifiers in preprocessing | Mixed sentiment / contrast (7 cases) | Retrain all three models without "but", "however", "though", "yet", "except", "only" and "very" in the stopword list; compare `has_contrast` error rates |
| Sentence-level or order-aware model | Mixed sentiment, temporal shift, target shift | Add a BiLSTM or attention pooling over sentence vectors; compare error on `has_contrast` and on reviews with temporal markers ("now", "used to", "EDIT", "update") |
| Longer input for the neural models | Truncation | Retrain the TextCNN with `max_length` 512; compare error on the 540 reviews truncated at 256 |
| Label audit | Rating-text mismatch, ambiguous labels | Hand-label a random sample of the 1,355 all-model errors as label noise, irony or genuine error; report accuracy with and without suspected label noise |
| Probability averaging or a tuned threshold | Near-threshold errors (cases 11 to 15) | Average logreg and TextCNN probabilities, or tune the threshold, on the validation set; compare accuracy for P(positive) between 0.4 and 0.6 |
| Rating-statement token | Explicit rating lost in preprocessing (case 19) | Keep number words and map "N star(s)" to a rating token; compare error on reviews that state a star count |
