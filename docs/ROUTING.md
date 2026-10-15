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

---

# Day 6: the middle band gets smart

Day 5 sent every middle-band ticket (confidence 0.45 to 0.75) to a rewrite. The calibration data showed that is the wrong fix for about half of them, so the router now looks at *why* confidence is low.

## The rules

| Situation | Route |
|---|---|
| confidence >= 0.75 | draft |
| confidence < 0.45 | escalate |
| middle band, 2 retry steps already used | escalate (never loop forever) |
| middle band, top score >= 0.85, agreement < 0.5, not reconciled yet | **reconcile_category** |
| any other middle-band ticket | **rewrite_query** |

All settings (`TOP1_STRONG = 0.85`, `AGREEMENT_LOW = 0.5`, `MAX_ATTEMPTS = 2`, `OFF_TOPIC_TOP1 = 0.55`) are at the top of `src/agent/nodes.py`.

## Where 0.85 came from

The 17 middle-band in-domain tickets in `docs/calibration.csv`, sorted by top-1 score:

| Top-1 score | Tickets | Category agreement | Diagnosis |
|---|---|---|---|
| below 0.85 (0.63 to 0.84) | 8 | 0.0 to 0.6, mixed | weak matches -> rewrite the query |
| 0.85 or more (0.86 to 0.94) | 9 | **0.0 for all 9** | search worked, labels disagree -> reconcile |

The split sits in a natural gap (0.843 and 0.858 are the neighbours), so 0.85 is not a fine-tuned number.

`OFF_TOPIC_TOP1 = 0.55`: off-topic tickets scored at most 0.45 and real tickets never went below 0.63. Below 0.55 the escalation summary uses a fixed template and no LLM call.

## What reconcile does (checked on the 9 conflict tickets)

I ran reconcile on all 9 conflict tickets. After the prompt was changed to tell the LLM to follow how similar tickets are filed, **all 9 reached confidence 0.77 or higher** (0.77 to 0.91) and would be drafted.

The first version of the prompt told the LLM to "use your own judgement", and on the first test ticket (T28372) it simply kept its old category, which sent the ticket on to a rewrite and then an escalation. Telling it to follow the neighbours' majority fixed that.

Caveats:
- The new category matched the dataset's own label in only 2 of the 9 tickets (before reconcile, 1 of 9). The dataset labels are noisy (see `DATA_NOTES.md`), so "matches the neighbours" and "matches the dataset label" are different things. For the reply this matters little, because the reply is built from the neighbours' answers.
- Confidence rises partly by construction: moving the category toward the neighbours raises agreement. A "drafted after reconcile" ticket is therefore less certain than a "drafted directly" one.
- These 9 tickets were also the ones used to pick the 0.85 cutoff, so this check is not independent.

## Four deliberate test cases (`scripts/try_cases.py`)

| Case | Path | Result |
|---|---|---|
| a) easy ticket (T15090) | classify, retrieve, score, draft | drafted directly, confidence 0.96, 2 LLM calls |
| b) label conflict (T28372) | classify, retrieve, score, reconcile_category, draft | drafted, confidence 0.69 -> 0.77, 3 LLM calls |
| c) vague: "it's broken again since yesterday, same as last time, please fix asap" | classify, retrieve, score, rewrite, retrieve, score, rewrite, retrieve, score, escalate | escalated, confidence 0.55 -> 0.57, 4 LLM calls (LLM summary) |
| d) nonsense: "qwerty zxcvb lorem asdf 12345 !!!" | classify, retrieve, score, escalate | escalated, confidence 0.20, 1 LLM call (template summary) |

Case c showed that the second rewrite can return the same query as the first. The node now detects that and uses up the remaining retries instead of searching again (only the first loop really changes the search).

## TODO: full outcome counts (blocked by the Groq daily token limit)

The 110-ticket run (100 eval + 10 off-topic) is still to be done. The free Groq limit for `openai/gpt-oss-120b` (200,000 tokens/day) ran out at ticket 41 of 110 during development. `scripts/evaluate_routing.py` is now resumable and skips the draft LLM call (a draft is always exactly 1 call in the logs and never changes the route), so it needs about 80,000 tokens. Once the quota resets:

1. `python scripts/evaluate_routing.py` (writes `docs/routing_day6.csv`; run it again if it stops at the limit).
2. Fill in this table and the average LLM calls per ticket:

| Outcome | In-domain (100) | Off-topic (10) |
|---|---|---|
| drafted directly | TODO | TODO |
| drafted after reconcile | TODO | TODO |
| drafted after rewrite | TODO | TODO |
| drafted after reconcile + rewrite | TODO | TODO |
| escalated | TODO | TODO |

Average LLM calls per ticket: TODO (in-domain), TODO (off-topic).

---

# Day 7: the self-check (catching made-up replies)

Everything above decides *whether to try* answering. The self-check decides whether the answer that was written is *safe to send*. A reply can be well-written and still contain a promise or a fact that none of the past tickets support.

## What the draft step does now

The draft prompt tells the LLM to cite its sources: after every sentence that uses a past ticket it adds the id, like `[T01234]`, and it returns the list of cited ids (`used_ticket_ids`). The code no longer silently drops ids that are not in the search results (Day 4 did). It keeps them, so the check can catch them. Cited ids are the JSON list plus any `[T....]` tags found inside the reply text.

Citations stay in `draft_reply`. A later step could strip them from the customer-facing version.

## What the check does (two layers)

**Layer 1: rule check (no LLM, runs first).** Three plain rules:
1. There is a reply to check.
2. The reply cites at least 1 ticket.
3. Every cited id is one of the tickets retrieved for *this* ticket.

If a rule fails, the verdict is `not_supported` right away and the LLM is **not called**.

**Layer 2: LLM check.** Only if the rules pass. The LLM sees the reply and only the cited tickets (their text and their past resolution), and answers `{"verdict": "supported" | "not_supported", "reason": "..."}`. It is told to be strict about concrete claims (facts, steps, promises, product names, numbers, contact details, deadlines, policies) and to ignore greetings, apologies, citation tags and polite requests for more information. If the checker's own answer is unusable twice, the verdict is `not_supported` (fail safe).

One deliberate addition to "show ONLY the cited tickets": the checker also sees the customer's own ticket text, labelled as the customer's words. Without it, a reply that simply restates the customer's problem ("we understand access logs are being examined") would be flagged as unsupported, because that fact comes from the customer and not from a past ticket.

## Why the rule check runs before the LLM

- **It is free.** No tokens, no waiting. Given the Groq daily limit, that matters.
- **It catches the worst failure for sure.** A cited id that does not exist is a fabrication that no LLM judgement is needed to detect, and an LLM can be fooled where a set lookup cannot.
- **It cannot give a wrong answer about ids.** The LLM check is only used for the judgement that needs judgement: do these tickets actually say that?

A side effect: a polite "I need more details" reply that cites nothing fails the rule check. That is intended, since a reply with no source is not supported. The prompt asks the LLM to cite the closest ticket even then.

## What happens after the check

| Check result | What happens | Final status |
|---|---|---|
| supported | done | `replied` |
| not supported, no redraft yet | `redraft` once. The check's reason and the rejected reply go into the redraft prompt. Then check again. | (continues) |
| not supported again | `escalate`. The reason appears in the "WHY ESCALATED" part of `escalation_summary`, with the last rejected draft marked "do NOT send". | `escalated` |

**Redraft limit: exactly 1** (`MAX_REDRAFTS = 1` in `nodes.py`, counted by `redrafts` in the state). The worst case for one ticket is 1 draft, 1 redraft, 2 checks, then the escalation summary.

When a ticket is escalated after a failed check, `draft_reply` is cleared so a rejected reply can never be shown to a customer by mistake.

## Tests

`tests/test_check.py` (11 tests, run with `python -m pytest tests`) uses a fake LLM and a fake search, so it never touches Groq or the database. It covers: a good reply is `replied`; an invented id fails the rule check with **zero** LLM calls; a failed check followed by a good redraft ends `replied` with 1 redraft; two failed checks escalate with the reason in the summary; and there is never more than 1 redraft.

## TODO: live self-check test pending quota

Live self-check test pending quota: run a ticket with no good match and confirm it ends escalated, not answered.

Also still to check live: how often the real LLM check says `not_supported` on good replies (false alarms), and how often it lets a bad reply through. The mocked tests prove the wiring, not the quality of the LLM's judgement.

---

# Day 8: human review and learning

Escalating used to mean "stop and write a summary". Now it means "pause, wait for a person, and then continue".

## How it works

1. **Every run is saved.** With a checkpointer (LangGraph's Postgres one, in the same Docker database), each run is saved under `thread_id = ticket id` after every node. Closing the program loses nothing.
2. **Escalation pauses.** `escalate` builds the summary as before. The next node, `human_review`, calls `interrupt()` and the run stops there, waiting.
3. **A human decides** (`scripts/review_queue.py`, or `resume_ticket()` in code):

| Decision | Needs | Result |
|---|---|---|
| approve | a reply (the human writes it, or accepts the rejected draft in the CLI) | status `replied_by_human` |
| edit | the corrected reply | status `replied_by_human` |
| reject | nothing (optional note) | status `closed`, no reply |

   Every decision must say who made it (`decided_by`). The state stores the action, who, when (UTC) and the note in `human_decision`. A decision that is incomplete, unsigned or has an unknown action is refused **before** anything is resumed, and the ticket keeps waiting.
4. **Learning.** After a reply was *sent* (the self-check passed, status `replied`, or a human replied, status `replied_by_human`), the `learn` node embeds the ticket text and final reply and inserts them in the knowledge base as `L-<ticket id>` with `source = 'learned'`. Search results now include `source`, so learned tickets can be told apart from the original data. Rejected and closed tickets are never learned, and neither are tickets whose draft was rejected and not replaced.

## Rules that protect the data

- **Learning is off by default.** `run_ticket(..., learn=True)` switches it on for that run. Tests, calibration and evaluation runs therefore cannot change the knowledge base.
- **Eval-set tickets are never learned**, even with `learn=True`. This keeps the Day 2 rule that eval tickets never enter the database.
- **Citation tags (`[T01234]`) are stripped** from a reply before it is learned. They point at tickets that mean nothing outside that one run.
- **Learning is idempotent.** Running it twice (for example after a restart) adds the row once.
- **Without a checkpointer** (scripts, evaluation) `escalate` goes straight to the end, as before. A run can only pause if its state is being saved.

## Why `human_review` is a separate node from `escalate`

When a paused run continues, LangGraph runs the paused node again from its first line. If the pause were inside `escalate`, resuming would repeat the LLM call that writes the summary. In its own node, the repeated part is just the pause itself. (Test (b) checks this: after a restart and a resume, the LLM was called exactly once in total, for classify.)

## Limits and risks

- **Learning from the agent's own replies can feed on itself.** A `replied` ticket learned today can be retrieved tomorrow as a "past ticket" for a similar one. The self-check keeps the quality bar up, but a reply that slipped through would be copied. Human replies are the safer thing to learn from. If this becomes a problem, learn only from `replied_by_human`, or add a delay and a spot-check.
- **Near-duplicates pile up.** Nothing stops many learned copies of the same kind of ticket. The synthetic dataset already has this problem.
- **The learned ticket keeps the agent's category.** If the category was wrong, the learned row is labelled wrong too.
- **Nobody is notified.** `review_queue.py list` is a pull model: someone has to run it. A real system needs an alert or a web page (the Streamlit day).

## Tests

`tests/test_hitl.py` uses a mocked LLM and a mocked embedding, but a real Postgres. It runs against a **separate database called `tickets_test`** that it creates itself, and it refuses to run if `DATABASE_URL` does not point there, so the real knowledge base cannot be touched. It covers: (a) an escalated ticket pauses and is saved; (b) a restart (new graph, new connection) and a resume work, with no repeated LLM calls; (c) approve replies and learns; edit is recorded as `edit`; (d) reject closes and learns nothing; (e) after learning, a near-identical new ticket finds the learned one first (`source = 'learned'`). Extra tests cover learning from a self-checked reply, learning being off by default, the eval guard, refusing bad decisions, and refusing a second decision on a finished ticket.

## TODO: live check pending quota

Everything today used mocked LLM calls. Still to do once Groq quota allows: run a few real tickets with `learn=True` and a checkpointer against the real database, review them with `review_queue.py`, and check that learned rows look sensible. After that, remove any test learned rows from the real `tickets` table (`DELETE FROM tickets WHERE source = 'learned'`), unless they are meant to stay.
