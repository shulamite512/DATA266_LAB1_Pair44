# Task 1: Sequence model failure analysis

Checkpoint reviewed: `checkpoints/best.pt`. The examples below were selected from
`outputs/failure_candidates.md`; they are generated continuations, not edited text.

## Case 1 — Greedy repetition loop

- Decoding / prompt: greedy / `There was a big red ball`
- Snippet: `and said, "I want to the bird and said, "I want to the bird and said, "I want to the ball and said, "I want to the ball ...`
- Failure type: repetition and broken grammar.
- Observation: The model enters a high-probability loop around the phrase “and said, I want to the ball.” The repeated 4-gram rate is 0.836, the highest among the candidates. Greedy decoding always selects the locally most likely next character, so once the context falls into this loop it has no sampling noise to escape it. The result preserves common TinyStories surface patterns but loses story progress and grammatical structure.

## Case 2 — Greedy repetition after a plausible opening

- Decoding / prompt: greedy / `The dog was very happy because`
- Snippet: `he was so happy to the ball and said, "I want to the ball and said, "I want to the bear and said, "I want to the ball ...`
- Failure type: loss of coherence with repetition.
- Observation: The first continuation is locally plausible (“he was so happy”), but the model then changes the subject and repeatedly produces the same dialogue template. The repeated 4-gram rate is 0.765. This shows that a reasonable short prefix does not guarantee long-range state tracking: the character-level model can imitate frequent phrase transitions without maintaining the dog, the cause of happiness, or a consistent event sequence.

## Case 3 — Temperature sampling reduces repetition but increases noise

- Decoding / prompt: temperature 1.0 with top-20 sampling / `One day, a little girl named Lily`
- Snippet: `So then see burned her wing again. They saw the bull hat a fun. They told his car card. Tim was a limted to his grandma boom.`
- Failure type: broken grammar, spelling-like character errors, and loss of coherence.
- Observation: Sampling avoids the deterministic loop: its repeated 4-gram rate is 0.000. However, the continuation contains malformed words, abrupt subject changes, and weak causal links. The model therefore has learned useful local character statistics but not reliable sentence-level semantics. Increasing randomness improves diversity at the cost of selecting lower-probability, less grammatical character sequences.

## Overall reasoning

The aggregate generation metrics show the same trade-off. Greedy decoding has low diversity (Distinct-1/2/3 = 0.088/0.118/0.137) and a repeated 4-gram rate of 0.703, whereas temperature 0.8 and top-20 temperature 1.0 both have zero repeated 4-grams and much higher diversity. The latter still need better coherence. A testable next step is constrained decoding or a repetition penalty, evaluated alongside a larger context window and more training data; the comparison should report repetition rate, distinct-n, and a fixed human coherence rubric together.
