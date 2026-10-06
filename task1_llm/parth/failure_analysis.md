# Task 1: Failure Analysis (Parth)

The lab asks for at least 3 failure cases, each with a generated snippet, a failure type (for example repetition, broken grammar, loss of coherence, hallucination), and an observation.

Below are all 7 samples in `outputs/generated_samples.txt`, copied verbatim (prompt + 500 generated characters). Every sample comes from the same checkpoint:

| Field | Value |
|---|---|
| Run ID | `parth_gpt_char_v1_20260925T214701Z` |
| Checkpoint | `task1_llm/parth/checkpoints/best.pt` (epoch 10, val loss 0.739443) |
| Script | `src/generate.py` |
| New characters per sample | 500 |
| Context window | last 128 characters (no KV cache) |

Three cases are analysed, chosen to show three different failure types. The other four candidates stay below, verbatim, as evidence, each with a one-line note of what it shows.

| Failure case | Candidate | Decoding | Failure type |
|---|---|---|---|
| 1 | 1 | temperature 0.8, top-k 50, seed 8503, prompt "Once upon a time" | Entity and name inconsistency (loss of coherence) |
| 2 | 2 | temperature 0.8, top-k 50, seed 8504, prompt "Once upon a time" | Semantic incoherence and broken grammar |
| 3 | 7 | greedy decoding, seed 8503, prompt "Once upon a time" | Repetition under greedy decoding |

---

## Candidate 1

Decoding: temperature 0.8, top-k 50 | Seed: 8503 | Prompt: `Once upon a time`

```text
Once upon a time, there was a little girl named Mia. She had a big box with a ball and a big, red ball. One day, she saw a big box of wood far on the swings. Mia was very sad.
Lily told Tom to stop and play with him. They were sad because they were not stuck in the pond. They did not know that they were sad and didn't move.
Then he heard a voice and then something unexpected happened. A big cat came and took the park. The cat looked at the zoo and saw the spear was the best of friends and playing together.
The 
```

**Selected as failure case 1.**

- **Quoted span:** `there was a little girl named Mia. [...] Mia was very sad. Lily told Tom to stop and play with him.`
- **Failure type:** Entity and name inconsistency (loss of coherence)
- **Observation:** The sample introduces Mia as the main character and mentions her again, then continues with Lily and Tom as if they had already appeared. Mia's second mention starts at character 158 and "Lily" appears at character 176, so Mia was still inside the 128-character window and the switch is not explained by the context limit alone. Each sentence is locally grammatical, but the story does not keep track of who it is about.

---

## Candidate 2

Decoding: temperature 0.8, top-k 50 | Seed: 8504 | Prompt: `Once upon a time`

```text
Once upon a time, there was a little boy named Tim. Tim had a big purple said, "Hi Tim! You are a nice museum." Tim was very happy, and Tim remembered his mum would have a raven banana.
Tim would be fun to play with his friends and play with his new toy all day. One day, Tim saw a big box of window. It was a big, round tail.
Tim asked, "Why do you like to share and run around with me?" Tim said, "I want to play just with you, but I am also scared." Tim was sad. He cried and cried, "The dog is not mine."
Tim and
```

**Selected as failure case 2.**

- **Quoted span:** `Tim had a big purple said, "Hi Tim! You are a nice museum." [...] his mum would have a raven banana. [...] Tim saw a big box of window. It was a big, round tail.`
- **Failure type:** Semantic incoherence and broken grammar
- **Observation:** The words are spelled correctly and are common in TinyStories, but they are combined without meaning. "A big purple said" drops the noun the adjectives need, and "a nice museum", "a raven banana" and "a big box of window" join words that do not fit the sentence or each other. The model has learned word forms and short-range word order, but not which words make sense together.

---

## Candidate 3

Decoding: temperature 0.8, top-k 50 | Seed: 8505 | Prompt: `Once upon a time`

```text
Once upon a time, there was a modest car. The car was very nosy and liked it. He liked to play with his hook, so he was very smart. One day, Tom went to for a walk with his mom. Tom saw a big piece of the pilot and thought it was a friend. Tom wanted to help the bird find it too. Tom asked his mom, "Can you help me find my friend?"
Tom and Lily nodded. They wore happy or a fun song together. They were not so locky that being foolish. They learned that sharing is good to have always help each other.

Once upon a
```

**Not selected.** Shows invented or misused words ("locky", "a big piece of the pilot"), Lily appearing without introduction, and a restart of "Once upon a" at the end of the sample.

---

## Candidate 4

Decoding: temperature 0.8, top-k 50 | Seed: 9503 | Prompt: `Tom had a red ball`

```text
Tom had a red ball. He asked his friend, Sam, "Can I help you?" Sally said, "Yes, I want to play with your ball and let me help me find you?"
Sally thought the ball was a good boy. She had an idea. She ran to her friend, Sally. It was the balloon to sleep. Sally was so happy and played with the ball. She had a big ball, a plan, and some other animals. They laughed and had lots of fun.

Once upon a time, there was a little duck named Sam. Sam was walking in his yard. He was a girl called Lily. She had a great day.
```

**Not selected.** Shows name inconsistency (Tom asks his friend Sam, then Sally answers; "a little duck named Sam [...] He was a girl called Lily") and a restart of "Once upon a time" inside the continuation.

---

## Candidate 5

Decoding: temperature 0.8, top-k 50 | Seed: 9504 | Prompt: `Tom had a red ball`

```text
Tom had a red ball that she loved to play with his friends. One day, Tom said to his friend, Tom, "Wait for you. I can teach you something us can still be fun."
Just then, a big dog came and saw the dog had a pretty flower. The dog jumped out of the sky. The cat told Tim and the dog were safe and happy. They had a great time playing with the big dog.

Once upon a time, there was a hungry girl named Lily. She liked to play with her toys and she would run away. One day, she found a bug in the bush. She was scared a
```

**Not selected.** Shows pronoun and coreference errors ("Tom had a red ball that she loved", "Tom said to his friend, Tom") and a restart of "Once upon a time" inside the continuation.

---

## Candidate 6

Decoding: temperature 0.8, top-k 50 | Seed: 9505 | Prompt: `Tom had a red ball`

```text
Tom had a red ball. He loved to see what was inside the bag, and he would find it.
One day, Tom found a funny box. The box was very tired. It did not want to go on the box. Tom wanted to help too. He tried to put the box. Tom did not know what to do. He went up to the box and started to take the box up.
Tom showed his friends the box to Tom. They both asked their mom, "Why is the box?" Lily said, "Tom, you saw the box in the river and be friends."
Tom and Sam said, "Yes, you can play with their toys." They hugged
```

**Not selected.** Shows lexical repetition ("box" appears 9 times in 500 characters) and a self-reference error ("Tom showed his friends the box to Tom").

---

## Candidate 7

Decoding: greedy (argmax) | Seed: 8503 | Prompt: `Once upon a time`

```text
Once upon a time, there was a little boy named Tim. Tim loved to play with his ball. One day, Tim saw a big box of cars. He wanted to play with it. Tim thought it would be fun to play with it.
Tim went to the box and saw a big box. He wanted to play with it. He took the box and started to play with it. Tim was very happy to have a new friend.
The boy said, "I want to play with you, but I want to play with you." Tim was sad because he was still scared. He wanted to help the bird. He tried to catch the bird, but 
```

**Selected as failure case 3.**

- **Quoted span:** `He wanted to play with it. Tim thought it would be fun to play with it. [...] He wanted to play with it. He took the box and started to play with it. [...] "I want to play with you, but I want to play with you."`
- **Failure type:** Repetition under greedy decoding
- **Observation:** "to play with it" appears four times in 500 characters, "He wanted to play with it." repeats as a whole sentence, and one line of dialogue repeats its own first clause. The sample's repeated 4-gram rate is 0.073, against 0.000 for the six temperature samples. Greedy decoding always takes the single most likely next character, so a likely phrase tends to recur; temperature sampling with top-k breaks these loops but produces more incoherent combinations, as in candidate 2.

---

## Related metrics

From `metrics_report.csv` (word-level, continuation only):

| Decoding | Samples | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate |
|---|---|---|---|---|---|
| temperature 0.8, top-k 50 | 6 | 0.405910 | 0.811617 | 0.933439 | 0.000000 |
| greedy | 1 | 0.526786 | 0.756757 | 0.845455 | 0.073394 |

## Summary across the selected cases

Three failure types recur across the samples: entity drift (candidates 1, 4 and 5), meaningless word combinations (candidates 2 and 3), and repetition (candidate 7, and the word "box" in candidate 6). Three of the six temperature samples also restart "Once upon a time" inside the continuation (candidates 3, 4 and 5). Spelling is mostly correct, so the errors are in meaning, entity tracking and repetition rather than in characters, which fits metrics (top-1 accuracy 0.766, perplexity 2.09) that reward local next-character prediction.

Possible causes, none of them tested here, are the 128-character context (about 28 words), the small model (3.2M parameters, 10 epochs) and the decoding settings. Candidate 1 shows that the context limit is not the whole explanation, because the name changed while the original name was still in the window. The restarts are consistent with how the training windows are cut: each split's stories are joined with a blank line into one stream, so some training windows contain the end of one story followed by "Once upon a time".

**How to test fixes** (proposed, not run):

1. Generate many samples per prompt with fixed seeds and compare greedy, temperature-only and temperature with top-k on the repeated 4-gram rate, to separate decoding effects from model effects.
2. Train a model with a longer context (for example 256 characters) and count name switches per sample on the same prompts and seeds.
3. Train on windows that never cross a story boundary and count how often "Once upon a time" appears inside continuations.
