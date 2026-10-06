# Task 2: Error review (20 errors)

Model reviewed: `experimental_gru` (checkpoint: checkpoints/experimental_gru.pt). Full texts: outputs/error_review_candidates.json

| # | test idx | category | label | pred | p(pos) | error type (manual) |
|---|---|---|---|---|---|---|
| 1 | 1196 | confident_false_positive | 0 | 1 | 0.997 | sarcasm / negative overall stance |
| 2 | 3611 | confident_false_positive | 0 | 1 | 0.994 | mixed sentiment; faint praise |
| 3 | 3773 | confident_false_positive | 0 | 1 | 0.994 | value complaint despite positive aspects |
| 4 | 2463 | confident_false_positive | 0 | 1 | 0.994 | very short / insufficient context |
| 5 | 2888 | confident_false_positive | 0 | 1 | 0.992 | contrastive sarcasm |
| 6 | 3626 | confident_false_negative | 1 | 0 | 0.001 | sarcasm and contradictory cues |
| 7 | 9647 | confident_false_negative | 1 | 0 | 0.001 | positive narrative without simple keywords |
| 8 | 5130 | confident_false_negative | 1 | 0 | 0.003 | long mixed review / contextual polarity |
| 9 | 1534 | confident_false_negative | 1 | 0 | 0.003 | positive-to-negative contrast |
| 10 | 4490 | confident_false_negative | 1 | 0 | 0.004 | aspect-level mixed sentiment |
| 11 | 3065 | near_threshold | 0 | 1 | 0.500 | mixed and qualified opinion |
| 12 | 8273 | near_threshold | 0 | 1 | 0.501 | positive opening followed by criticism |
| 13 | 713 | near_threshold | 1 | 0 | 0.498 | niche preference / subjective wording |
| 14 | 8939 | near_threshold | 1 | 0 | 0.497 | mild negatives outweighed by praise |
| 15 | 5846 | near_threshold | 1 | 0 | 0.497 | multiple aspects with different polarity |
| 16 | 4835 | slice_specific (length_long) | 1 | 0 | 0.497 | long review; evidence dispersed |
| 17 | 7099 | slice_specific (length_long) | 0 | 1 | 0.503 | long review; positive style cues dominate |
| 18 | 9345 | slice_specific (length_long) | 1 | 0 | 0.497 | long review; praise and criticism mixed |
| 19 | 4155 | slice_specific (length_long) | 1 | 0 | 0.496 | long review; local positive cues dominate |
| 20 | 8285 | slice_specific (length_long) | 0 | 1 | 0.506 | long review; delayed complaint structure |

## Error-pattern reasoning

The confident false positives are mostly reviews with positive local words but a
negative overall judgment: sarcasm, faint praise, price/value complaints, or very short
context. The confident false negatives show the reverse problem: positive opinions are
expressed through detailed narrative, while negative opinions are mixed with praise or
contradictory statements. Near-threshold examples are genuinely ambiguous because
different aspects of the same business receive different sentiment. The five long-review
errors show that truncation and dispersed evidence are important: the final judgment may
occur far from the words describing the experience. These patterns are consistent with
the slice results: the GRU is strongest without negation (macro-F1 0.922) and weakest on
long reviews (macro-F1 0.914) and reviews containing negation (macro-F1 0.916).

## Notes per error

### 1. confident_false_positive (idx 1196)
> Note: The Company that owned Scottsdale Auto Salon (Scottsdale Auto Salon LLC) Declared Chapter 11 Bankruptcy in early March 2009.  The facility is being operated by the bank according to employees that now staff it.  "Auto Salon" and "Auto Spa" are terms that only get thrown around in Scottsdale apparently.  In the rest of the civilized world, these places are simply known as car washes.  Sadly, ...

- Error type:
- Why the model got it wrong:

### 2. confident_false_positive (idx 3611)
> Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago.   Copper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly, which is rare. Usually indian food, good indian good, takes at least 30-45 minutes. We got our order in li...

- Error type:
- Why the model got it wrong:

### 3. confident_false_positive (idx 3773)
> Very small portions, but good food. Had a perfectly cooked fillet minion, but $60 and they could have served with a toothpick, it was so small.   Same for rest of our party. Excellent salmon, but maybe 5 bites again $60.  Everything is a la carte so sides extra. Thai papaya salad good, and key lime pie great.

- Error type:
- Why the model got it wrong:

### 4. confident_false_positive (idx 2463)
> Food is always good.

- Error type:
- Why the model got it wrong:

### 5. confident_false_positive (idx 2888)
> What I love about Rubios' is that they always have beer. Always.  That is all I love though...

- Error type:
- Why the model got it wrong:

### 6. confident_false_negative (idx 3626)
> TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++  The food here is good but isn't that spectacular, the beer selection is somewhat disappointing if you like a good craft beer. Ask for a craft beer and get an insulting comment from the waitress.   What makes this place awesome is the atmosphere, live bands, sports on the TVs, the very obnoxious staff and the hat...

- Error type:
- Why the model got it wrong:

### 7. confident_false_negative (idx 9647)
> i wasn't in an objective frame of mind, given that i was a visitor from another state and was willing to try just about anything local. that said, this place left a smile on my face.  maybe it was the 1950s-era big neon sign out front. maybe it was the fiberglass horse on the roof. maybe it was the kitchy old-school cowboy-themed decor inside. maybe it was the fact that our waitress wore a holster...

- Error type:
- Why the model got it wrong:

### 8. confident_false_negative (idx 5130)
> Last night several parents came in with over 15 children to celebrate their 9 year olds 4th grade graduation at 9:15 pm. The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue if anything were to happen to them on their premise. The children were running in and out of the bar while the parents continued to drink upstairs claim...

- Error type:
- Why the model got it wrong:

### 9. confident_false_negative (idx 1534)
> I really enjoyed the cupcakes from Mad Hatter since they opened. Unfortunately they decided to double the prices and as much as I like the cupcakes I rather spend my money on something else. They now charge $26 for a dozen mini-cupcakes and the previous price was $15. They lost a loyal (Chubby) customer.

- Error type:
- Why the model got it wrong:

### 10. confident_false_negative (idx 4490)
> Low points first:  - weird, office-like swivle chairs.  Uncomfortable & they kill the ambiance of an otherwise decent restaurant. -  beef filets, even when prepared medium rare, were dry. -  when your food arrives there are a ton of waitstaff, but drinks were refilled too rarely & some in my party didn't get new silverware after the appetizer utensils were cleared. -  bernaise sauce was kind of we...

- Error type:
- Why the model got it wrong:

### 11. near_threshold (idx 3065)
> If you are looking for cheap and accessible. This is it. Don't expect a whole lot. It's a hotel for cheapsters, so you will get what you expect. It isn't terrible, but it isn't wonderful either. The pool was descent and we went 2 times. The location is great- right on the strip, monorail station inside.

- Error type:
- Why the model got it wrong:

### 12. near_threshold (idx 8273)
> Yes, pizza from Brooklyn, is brand/location advertising that is attractive to a pizza lover.  Even the interior decor of b/w framed poster is unmistakably New York New York; it is homey and warm without the crowd and the lines of a neighborhood pizzeria.  A "Hello" when I walked-in was very welcoming and they had ESPN on so the sporty feel was nice too.  And then the order...  My spaghetti sauce i...

- Error type:
- Why the model got it wrong:

### 13. near_threshold (idx 713)
> THIS is an old-school pizza shop!  I'm sure that many of the foodie-types who frequent Yelp are going to dismiss this place, but having heard so much about it from my local pizza-loving friends about the pie at Fiori's, I finally decided to stop and order a couple of pizzas to go while I was in the neighborhood.  The sauce definitely isn't for everyone - it's a departure from the typical corporate...

- Error type:
- Why the model got it wrong:

### 14. near_threshold (idx 8939)
> I'm giving this place 5 stars. The Boyfriend & I decided to randomly try a new place (his idea) and I know he loves Brazilian steakhouses... So I steered us toward this place. At first we weren't sure what to think.. The place looks empty when you walk in, but actually almost every table had people. It took a little bit to get a table, but obviously they were busy. She sat us at a booth, which was...

- Error type:
- Why the model got it wrong:

### 15. near_threshold (idx 5846)
> We went here back in October and I forgot to come back and write my review. I'm giving it 4 stars based on my feelings that it was 4.5 or 5 but my hubby thought more like a 3. He wasn't impressed. I actually really liked my cotton candy martini drink - it was very very sweet and pretty original I thought. Try it. The bartender really wasn't very friendly at all though. Our waitress on the other ha...

- Error type:
- Why the model got it wrong:

### 16. slice_specific (length_long) (idx 4835)
> I was initially a bit skeptical about coming here because 1) there were bad reviews and 2)  they were doing a groupon sale.  I know my second reason may sound weird to some of you but businesses promote their business on groupon normally when they're struggling, which may be a result of poor service.  So I contacted them to ask them if their groupon deal was applicable to any day since it was a "n...

- Error type:
- Why the model got it wrong:

### 17. slice_specific (length_long) (idx 7099)
> A shi-shi sports bar with an appetizing menu executed in a very average manner.    The current tenants of this space (previously Daddy Macs, and Blue Wave before that) have stuck with the existing deep red on black decor and subdued lighting. Very Victorian. It harkens back to the supper clubs of the olden days. I half expect to find gentlemen in top hats when I entered the door. Instead I only fo...

- Error type:
- Why the model got it wrong:

### 18. slice_specific (length_long) (idx 9345)
> I've stayed here so many times, I've actually lost count. Most of the stays have been comped, too. I appreciate the slot host, although he doesnt seem to like to be bothered all that much. I absolutely love the atmosphere. It's great to go a few times a year and stay there but I'm getting worn out with the crowd. It never changes...its always the good looking people. I've been around them a long t...

- Error type:
- Why the model got it wrong:

### 19. slice_specific (length_long) (idx 4155)
> took little lucy here and she did great.  :) she just gets the brush and bath but leaves so clean a her coat is so soft  afterwards.  their pricing is spot on, in my opinion, and I actually asked to have her teeth cleaned in stead of having her nails trimmed and it wasn't a problem at all and no extra charge.  thought she might not like it since it was her first grooming experience but she left ha...

- Error type:
- Why the model got it wrong:

### 20. slice_specific (length_long) (idx 8285)
> Came in after some highly touted recommendations.  Stood at the front for approx. 4-5 mins before anyone acknowledged us (other than the pianist who was  a little too loud).  We were seated then another 10 mins passed at which point I waved down a waiter so we didn't need to continue our wait.   We promptly ordered, 2 salads, 3 entrees. Salads came out in about 10-15 mins.  I started with a spinac...

- Error type:
- Why the model got it wrong:

## Per-error reasoning summary

1. Bankruptcy and dismissive wording are negative despite neutral business terms.
2. “Fine enough” and comparison with a preferred restaurant weaken the positive food cues.
3. Positive food adjectives are outweighed by tiny portions and excessive price.
4. The isolated short sentence gives almost no context beyond a positive-looking phrase.
5. “That is all I love” reverses the earlier beer praise through sarcasm.
6. Extreme criticism is contradicted by “would eat here again” and “A++++”.
7. Positive feeling is expressed through a detailed story and atmosphere rather than simple sentiment words.
8. The review’s polarity depends on the bar/children situation and long-range event context.
9. Early praise is reversed by the price increase and loss of loyalty.
10. Several service and food complaints are mixed with the statement that the restaurant is otherwise decent.
11. “Cheap and accessible” is positive, while “cheapsters,” “not terrible,” and “not wonderful” are qualified and mixed.
12. Welcoming atmosphere is followed by a negative account of the order, creating a discourse-level reversal.
13. The reviewer anticipates disagreement and describes a niche preference, making lexical polarity unreliable.
14. Small initial doubts are outweighed by the final five-star judgment.
15. Different reviewers and different aspects disagree, so the overall label is inherently mixed.
16. In the long review, evidence about possible poor service is distributed across the text and may be truncated.
17. Positive decorative and menu language competes with “very average” execution.
18. Appreciation of the atmosphere is mixed with fatigue about the crowd, requiring aspect aggregation.
19. Many positive local details can dominate even when the selected long-review label is negative.
20. The review builds slowly from waiting-time complaints to food evaluation, so early and late cues conflict.

## Proposed testable fix

- Fix: Preserve negation and discourse structure, increase the maximum sequence length or use hierarchical pooling, and add hard-negative examples containing sarcasm, contrast, and mixed aspect sentiment. Keep very short reviews as a separate slice rather than treating them as ordinary long reviews.
- How to test it: retrain the GRU with the same seed and split, compare the original 20-error categories and the existing length/negation slices, and require improvements in long-review and negation macro-F1 without reducing overall macro-F1. Separately compare a truncation length above 120 tokens and check whether the five long-review errors become less confident.
