# Routing: confidence score and thresholds (Day 5)

After searching, the agent has to decide: *do I trust these past tickets enough to write a reply?* It turns that into one number, the **confidence**, and a router picks the next step.

## The formula

```
confidence = 0.8 * (best similarity score) + 0.2 * (category agreement)
```

- **Best similarity score** (top-1): how close the single closest past ticket is, from 0 to 1.
- **Category agreement**: of the 5 similar tickets, the share whose category is the same as the one the agent just classified. For example 4 of 5 gives 0.8.

The weights and thresholds live in one block at the top of `src/agent/nodes.py`.

Why these weights: the similarity score is by far the stronger signal, so it gets 0.8. Agreement is a useful second opinion, but the dataset labels are noisy (see `DATA_NOTES.md`), so even good in-domain tickets have a median agreement of only 0.4. It gets the smaller 0.2.

## Where the numbers came from

`scripts/calibrate.py` ran all 100 eval tickets and 10 off-topic tickets (cooking, a rant, gibberish, a Tesla question, the single word "help", and so on) through classify and retrieve. Raw results are in `docs/calibration.csv`.

| Measure | Group | min | 25% | median | 75% | max |
|---|---|---|---|---|---|---|
| top-1 score | in-domain (100) | 0.63 | 0.90 | 0.96 | 0.98 | 1.00 |
| top-1 score | off-topic (10) | 0.14 | 0.21 | 0.25 | 0.28 | 0.45 |
| average of top-5 | in-domain | 0.59 | 0.82 | 0.88 | 0.91 | 0.97 |
| average of top-5 | off-topic | 0.13 | 0.18 | 0.22 | 0.27 | 0.32 |
| category agreement | in-domain | 0.0 | 0.0 | 0.4 | 0.8 | 1.0 |
| category agreement | off-topic | 0.0 | 0.0 | 0.0 | 0.2 | 0.8 |
| **confidence** (formula above) | in-domain | 0.54 | 0.77 | 0.82 | 0.90 | 0.99 |
| **confidence** | off-topic | 0.11 | 0.21 | 0.24 | 0.27 | 0.36 |

Things this shows:
- The best similarity score separates the two groups very well. The worst in-domain ticket (0.63) is far above the best off-topic ticket (0.45). The "very high scores" were real, but so is the gap.
- Category agreement alone is a poor signal. One off-topic ticket (a poem request) got 0.8 because the classifier is forced to pick a category and its neighbours happened to be Customer Service. This is why agreement only gets a small weight.
- Confidence for off-topic tickets tops out at **0.36**. Confidence for in-domain tickets bottoms out at **0.54**. There is a clear gap of about 0.18 between them.

## The two thresholds

| Threshold | Value | Meaning |
|---|---|---|
| **LOW** | **0.45** | Below this: escalate to a human |
| **HIGH** | **0.75** | At or above this: write a reply |

- **LOW = 0.45** sits in the middle of the gap between the highest off-topic confidence (0.36) and the lowest in-domain confidence (0.54). Any value from about 0.37 to 0.54 gives the same result on this data, so the middle is the safest pick.
- **HIGH = 0.75** is about the 15th percentile of in-domain confidence. I chose it so that most tickets get a reply and a meaningful minority are sent for a retry. It is a judgement call, not a cliff: 0.70 would retry 8%, 0.80 would retry 39%.

## What that does on the calibration set

| Route | In-domain tickets (100) | Off-topic tickets (10) |
|---|---|---|
| draft (confidence >= 0.75) | **83** | 0 |
| rewrite_query (0.45 to 0.75) | **17** | 0 |
| escalate (< 0.45) | 0 | **10** |

How the threshold choice changes things (in-domain draft / rewrite / escalate, off-topic escalated):

| HIGH / LOW | In-domain draft | rewrite | escalate | Off-topic escalated |
|---|---|---|---|---|
| 0.70 / 0.45 | 92 | 8 | 0 | 10 of 10 |
| **0.75 / 0.45 (chosen)** | **83** | **17** | **0** | **10 of 10** |
| 0.80 / 0.45 | 61 | 39 | 0 | 10 of 10 |
| 0.75 / 0.55 | 83 | 16 | 1 | 10 of 10 |

## Honest limits

- **Only 10 off-topic tickets.** All of them were escalated with a good margin, but that is a small sample. The hardest one was the Tesla autopilot question, a tech problem for a product the dataset never mentions. It scored 0.27 confidence, still well below LOW. A real out-of-scope ticket about, say, a Windows or router problem could score higher, because the dataset is full of IT problems.
- **The eval tickets are unusually easy to match.** The knowledge base contains many near-copies of the same synthetic ticket. Real customer tickets would probably score lower, so these thresholds should be checked again on real data.
- **The retry band is driven by label noise.** Most of the 17 "rewrite" tickets have a high similarity score (often 0.85 to 0.94) but category agreement 0.0, meaning the similar tickets carry different labels than the agent's classification. Those tickets are borderline mostly because the dataset labels disagree, not because the search failed.
- **The classifier has no "Other" option.** An off-topic ticket still gets a category (the cooking question became "Customer Service"). The confidence score catches it, not the classifier.
