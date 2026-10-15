# LangGraph Basics (Day 3 study notes)

## The three ideas: state, nodes, edges

**Everyday analogy: a hospital patient chart.**
A patient walks in with a clipboard. They go from the front desk, to the nurse, to the doctor, to the pharmacy. Each person reads the chart, does their one job, writes their result on the chart, and passes it on.

| LangGraph word | Hospital analogy | In this project |
|---|---|---|
| **State** | The chart on the clipboard. Everyone reads and writes it. | `TicketState` in `src/agent/state.py`: ticket text, category, priority, draft reply, and so on |
| **Node** | One person or station with one job | A Python function, like `classify(state)` in `src/agent/nodes.py` |
| **Edge** | The arrow telling the chart where to go next | `graph.add_edge("classify", END)` |
| **Conditional edge** | A fork: "if the nurse writes *emergency*, go to the ER" | Coming later: "if confidence is low, try again" |
| **START / END** | The front door and the exit | Built-in markers |

Important rules:
- A node **receives the whole state** and **returns only the fields it wants to change**. LangGraph merges that into the state for you.
- A node does one job. That makes each step easy to test on its own.
- The edges decide the order, not the nodes.

## Our graph on Day 3

```
START  ->  classify  ->  END
```

Code (`src/agent/graph.py`): create `StateGraph(TicketState)`, add the node, add the two edges, then `.compile()`. `run_ticket(text)` creates a fresh chart and runs it through.

## Dry run: one real ticket

Ticket **T12786** from the eval set. Its real label is Technical & IT Support, priority high.

> Recurring Timeouts in Project Dashboard
>
> Facing frequent timeouts on the project dashboard likely due to server overload and inefficient queries.

`run_ticket` fills in the starting chart. The classify node then asks the LLM (through `call_llm`) and writes its answer onto the chart.

| State field | Before classify | After classify |
|---|---|---|
| `ticket_id` | `"T12786"` | `"T12786"` (unchanged) |
| `text` | the ticket text above | unchanged |
| `category` | *(not set)* | `"Technical & IT Support"` |
| `priority` | *(not set)* | `"high"` |
| `classify_reason` | *(not set)* | `"Frequent timeouts on the project dashboard"` |
| `similar_tickets` | *(not set)* | *(not set, comes on Day 4)* |
| `draft_reply` | *(not set)* | *(not set)* |
| `analysis` | *(not set)* | *(not set)* |
| `confidence` | *(not set)* | *(not set)* |
| `attempts` | `0` | `0` |
| `search_query` | *(not set)* | *(not set)* |
| `check_result` | *(not set)* | *(not set)* |
| `status` | `"new"` | `"classified"` |
| `steps_log` | `[]` | `["classify: Technical & IT Support / high - Frequent timeouts on the project dashboard"]` |

What happened inside the classify node:
1. Build a prompt with the 8 allowed categories, the 3 priorities and the ticket text.
2. `call_llm(..., json_mode=True)` returns something like `{"category": "...", "priority": "high", "reason": "..."}`.
3. **Validate.** Is it real JSON? Is the category on the list? Is the priority low, medium or high? If not, try once more with a stricter prompt. If that fails too, use `Unknown` and `medium`.
4. Return the changed fields. LangGraph merges them into the chart, and the graph reaches END.

The LLM call took about 1 second and about 370 tokens in and 150 tokens out. Every call is written to `logs/llm_calls.jsonl`.

## Day 4: the graph now has three nodes

```
START -> classify -> retrieve -> draft -> END
```

- **classify** (Day 3): picks category and priority.
- **retrieve**: searches the database for the 5 most similar past tickets. No LLM involved, just the embedding search.
- **draft**: gives the LLM the new ticket plus those 5 past tickets *with their answers*, and asks for a reply and an analysis card, as JSON. The LLM is told to use only the past answers and not to invent anything.

### Call order (who calls whom)

```
run_ticket(text)                      graph.py   builds the starting state, runs the graph
 |
 +-> classify(state)                  nodes.py
 |     +-> call_llm(...)              llm.py     the ONLY place that talks to Groq
 |
 +-> retrieve(state)                  nodes.py
 |     +-> search_similar(query, 5)   search.py  embeds the text, asks Postgres/pgvector
 |
 +-> draft(state)                     nodes.py
       +-> call_llm(...)              llm.py     (a 2nd call only if the first answer was invalid)
```

A normal ticket costs **2 LLM calls** (one in classify, one in draft) and **1 database search**.

### Dry run: ticket T15090 through all three nodes

> Concerns About Hospital Security Systems
>
> Suspected data breach, currently examining access logs

Real label: Technical & IT Support, priority medium.

**Start** (made by `run_ticket`): `ticket_id`, `text`, `attempts = 0`, `status = "new"`, `steps_log = []`. Every other field is empty.

**After classify** - changed fields:

| Field | Value |
|---|---|
| `category` | `"Technical & IT Support"` (new) |
| `priority` | `"high"` (new) |
| `classify_reason` | `"Suspected data breach requires urgent technical investigation"` (new) |
| `status` | `"new"` -> `"classified"` |
| `steps_log` | + `"classify: Technical & IT Support / high - ..."` |

**After retrieve** - changed fields:

| Field | Value |
|---|---|
| `similar_tickets` | 5 past tickets (each with ticket_id, text, answer, category, priority, similarity). Ids and scores: T11305 0.99, T23267 0.97, T13969 0.86, T21192 0.85, T10969 0.84 (new) |
| `search_query` | the ticket text, since it was empty before (new) |
| `steps_log` | + `"retrieve: found 5 similar tickets, top score 0.99"` |

`status` stays `"classified"`, and the category, priority and reason are untouched.

**After draft** - changed fields:

| Field | Value |
|---|---|
| `draft_reply` | "We take this potential data breach very seriously and are ready to assist. Please provide the information from the access logs to better assist with this matter. We request a call at your earliest convenience to discuss the situation further." (new) |
| `used_ticket_ids` | `["T11305", "T23267", "T13969"]` (new) |
| `analysis` | `summary`: suspected data breach in the hospital's security systems, logs being examined. `root_cause`: potential unauthorized access, exact cause not yet known. `suggested_fix`: provide the access log details and schedule a call to investigate. `severity`: `"high"` (new) |
| `attempts` | `0` -> `1` |
| `status` | `"classified"` -> `"drafted"` |
| `steps_log` | + `"draft: reply written, used 3 past tickets, severity high"` |

Still empty after Day 4: `confidence` and `check_result`. They are for the next days.

What the draft node checks before it accepts the LLM's answer:
1. Is it valid JSON, with a non-empty `reply` and all four analysis fields?
2. Is `severity` low, medium or high?
3. Is every id in `used_ticket_ids` really one of the 5 tickets we found? Made-up ids are dropped and noted in `steps_log`. This stops the agent from claiming it used a ticket that does not exist.
4. Does the reply contain a raw tag like `<name>`? If so it is cleaned.

If the answer is invalid, it tries once more with a stricter prompt. If that fails too, `status` becomes `"draft_failed"` and there is no reply.

## Day 5: conditional edges (the graph starts to choose)

Until now every arrow went to one fixed place. A **conditional edge** is an arrow with a fork. Instead of saying "after this node, go to X", it says "after this node, ask a small function which way to go".

**Analogy: the triage nurse.** After checking your vitals, the nurse looks at one number and says "routine, go to the doctor", "unclear, let's re-check you", or "this is not something we handle, here is a referral". The nurse does not treat you, they only point.

In our graph:

```
START -> classify -> retrieve -> score_confidence -> router -+-> draft ---------+-> END
                                                              +-> rewrite_query -+
                                                              +-> escalate ------+
```

- **score_confidence** is a normal node. It computes the confidence number and writes it into the state.
- **router** is *not* a node. It is a small function (`router(state)` in `nodes.py`) that reads `confidence` and returns a name: `"draft"`, `"rewrite_query"` or `"escalate"`. It never changes the state.
- In `graph.py`, `add_conditional_edges("score_confidence", router, {...})` connects that returned name to the node with the same name.
- Today `rewrite_query` and `escalate` are placeholders. They only set `status` and write a log line. Day 6 makes them real.

The rule (numbers explained in `ROUTING.md`):

| Confidence | Route | Status afterwards |
|---|---|---|
| 0.75 or more | draft | `drafted` |
| 0.45 up to 0.75 | rewrite_query | `needs_rewrite` |
| below 0.45 | escalate | `escalated` |

### Dry run A: a ticket that goes to draft (T15090)

> Concerns About Hospital Security Systems
>
> Suspected data breach, currently examining access logs

State **at the router** (right after score_confidence):

| Field | Value |
|---|---|
| `category` | Technical & IT Support |
| `priority` | high |
| `similar_tickets` | T11305 0.99, T23267 0.97, T13969 0.86, T21192 0.85, T10969 0.84 |
| `top_score` | 0.9942 (new in this step) |
| `agreement` | 0.8 (4 of the 5 are Technical & IT Support) (new) |
| `confidence` | 0.8 x 0.9942 + 0.2 x 0.8 = **0.9554** (new) |
| `status` | `classified` (unchanged) |
| `steps_log` (new last line) | `score_confidence: confidence 0.96 (top score 0.99, category agreement 0.8) -> route: draft` |

0.9554 is at least 0.75, so the router returns `"draft"`. The draft node then writes the reply, `status` becomes `drafted` and `attempts` becomes 1.

### Dry run B: a ticket that goes to escalate (OFF01)

> How do I make a good carbonara? I always end up with scrambled eggs.

State **at the router**:

| Field | Value |
|---|---|
| `category` | Customer Service (the classifier must pick something, so it picked the nearest-sounding one) |
| `priority` | low |
| `classify_reason` | "User asks a general cooking question unrelated to support topics" |
| `similar_tickets` | T16421 0.27, T16955 0.22, T17960 0.20, T11460 0.20, T28090 0.19 |
| `top_score` | 0.2707 (new) |
| `agreement` | 0.2 (only 1 of 5 is Customer Service) (new) |
| `confidence` | 0.8 x 0.2707 + 0.2 x 0.2 = **0.2565** (new) |
| `status` | `classified` (unchanged) |
| `steps_log` (new last line) | `score_confidence: confidence 0.26 (top score 0.27, category agreement 0.2) -> route: escalate` |

0.2565 is below 0.45, so the router returns `"escalate"`. The escalate placeholder sets `status` to `escalated` and logs that the ticket was handed to a human. **No `draft_reply` is written**, and the LLM is called only once (for classify). That is the point of the router: a bad match costs less and never produces a made-up reply.

Interesting detail: the classifier did not notice the ticket was off-topic. The *search scores* did. The scores are a more honest signal than asking the LLM.

## Day 6: loops (the graph can go back)

So far every ticket moved forward only. A **loop** is an edge that points *backwards* to a node that already ran, so the agent can try again with new information.

**Analogy: a doctor who is not sure.** Two different doubts need two different actions:
- "The test results are clear, but two of my colleagues label this condition differently." -> don't repeat the test, just **decide which label to use** (`reconcile_category`).
- "The test result is blurry." -> **repeat the test with a better sample** (`rewrite_query`, then search again).
- "I have tried twice and it is still unclear." -> **stop and refer the patient** (`escalate`).

Why the middle band is low tells us which fix to use. The router now has four exits:

| Situation | Route |
|---|---|
| confidence 0.75 or more | `draft` |
| confidence below 0.45 | `escalate` |
| middle band, best match score 0.85 or more, but few similar tickets share our category (the search worked, the labels disagree) | `reconcile_category` |
| middle band, weaker matches | `rewrite_query` |
| middle band, but 2 retry steps already used | `escalate` |

The two loops:
- **reconcile_category -> router.** The LLM sees the ticket and the 5 similar tickets with their categories, and picks the final category. The node recomputes agreement and confidence and goes straight back to the router. It runs at most once per ticket (`reconciled` flag in the state).
- **rewrite_query -> retrieve -> score_confidence -> router.** The LLM turns the ticket into a short search query, and the search runs again with it. If the rewrite comes out identical to the old query, retrying is pointless, so the retries are used up at once.

**Why a loop cannot run forever:** every reconcile or rewrite adds 1 to `attempts`. The router escalates when `attempts` reaches 2 (`MAX_ATTEMPTS` in `nodes.py`). So the longest possible path is two retry steps and then escalate.

**The escalate node is real now.** It sets `status = "escalated"` and writes `escalation_summary` for a human: a problem paragraph, why it was escalated, what was tried (the `steps_log`), and the 3 closest past tickets. When the best match score is below 0.55 (nothing in the history looks like this ticket, for example gibberish) it uses a fixed template instead of asking the LLM, which saves a call.

The full picture is in the docstring of `src/agent/graph.py`, and a generated diagram is in `docs/graph.md`.

### Dry run of a label conflict: ticket T28372

> Hello Customer Support, I hope this message finds you well. I am writing to seek assistance with the inconsistent results we are encountering in our digital strategy. It seems there might be some issues related to software integration and data discrepancies. Despite our efforts to troubleshoot by updating the software and cross-referencing the analytics, we have not yet seen any improvements. [...]

The dataset's own label is Product Support, priority medium. Path taken: `classify -> retrieve -> score_confidence -> reconcile_category -> draft`. Tables show only the fields that changed at each node.

**Start** (made by `run_ticket`): `ticket_id = "T28372"`, `text`, `attempts = 0`, `reconciled = False`, `confidence_history = []`, `status = "new"`, `steps_log = []`.

**1. After classify**

| Field | Value |
|---|---|
| `category` | `"Technical & IT Support"` |
| `priority` | `"high"` |
| `classify_reason` | "Inconsistent results due to software integration and data discrepancies" |
| `status` | `"new"` -> `"classified"` |
| `steps_log` | + `classify: Technical & IT Support / high - ...` |

**2. After retrieve**

| Field | Value |
|---|---|
| `similar_tickets` | T23563 0.86 (Product Support), T19586 0.80 (Returns and Exchanges), T15087 0.79 (Customer Service), T28314 0.79 (Customer Service), T15463 0.79 (Product Support) |
| `search_query` | the ticket text (it was empty) |
| `steps_log` | + `retrieve: found 5 similar tickets, top score 0.86` |

**3. After score_confidence**

| Field | Value |
|---|---|
| `top_score` | 0.8584 |
| `agreement` | 0.0 (none of the 5 are Technical & IT Support) |
| `confidence` | 0.8 x 0.8584 + 0.2 x 0.0 = **0.6867** |
| `confidence_history` | `[0.6867]` |
| `steps_log` | + `score_confidence: confidence 0.69 (top score 0.86, category agreement 0.0) -> route: reconcile_category` |

**Router:** 0.69 is in the middle band (0.45 to 0.75), attempts = 0, the top score 0.86 is at least 0.85 and agreement is below 0.5, and `reconciled` is False. So it returns `"reconcile_category"`.

**4. After reconcile_category**

| Field | Value |
|---|---|
| `category` | `"Technical & IT Support"` -> `"Product Support"` (the LLM: "Matches majority of similar tickets and concerns product functionality") |
| `agreement` | 0.0 -> 0.4 (2 of the 5 are now Product Support) |
| `confidence` | 0.6867 -> **0.7667** (0.8 x 0.8584 + 0.2 x 0.4) |
| `confidence_history` | `[0.6867, 0.7667]` |
| `reconciled` | `False` -> `True` |
| `attempts` | `0` -> `1` |
| `steps_log` | + `reconcile_category: changed 'Technical & IT Support' to 'Product Support' - ... (agreement 0.0 -> 0.4, confidence 0.69 -> 0.77)` |

**Router again** (same function, new numbers): 0.7667 is at least 0.75, so it returns `"draft"`.

**5. After draft**

| Field | Value |
|---|---|
| `draft_reply` | "Hello, thank you for reaching out to us regarding the inconsistent results you are seeing in your digital strategy. [...] could you please provide more details about the software integration and the specific data discrepancies...?" |
| `used_ticket_ids` | 3 of the 5 similar tickets |
| `analysis` | filled in, severity `"high"` |
| `attempts` | `1` -> `2` (draft also counts) |
| `status` | `"classified"` -> `"drafted"` |
| `steps_log` | + `draft: reply written, used 3 past tickets, severity high` |

This ticket cost 3 LLM calls (classify, reconcile, draft) instead of the usual 2.

An honest note: reconcile moved the category to match how the *neighbours* are filed. That is also why the confidence rose, so part of the improvement is by construction. It is a reasonable choice here because the reply is built from those neighbours' answers, but it means "reconciled" tickets are less certain than "drafted directly" ones.
## Day 7: check and redraft (the agent reviews its own work)

After `draft`, the graph now goes to **check**, and from there it can go forward, loop back for one rewrite, or stop at a human.

**Analogy: a proofreader.** The writer hands in a letter with footnotes saying where each fact came from. The proofreader first does a quick mechanical check (do the footnotes point at real documents? are there any footnotes at all?). Only if that passes do they read the letter against the cited documents. If something is not backed up, the letter goes back to the writer *once*, with a note saying what is wrong. If it is still wrong, a human takes over.

```
draft -> check -> [after_check] -+-- supported ---------------------------> END (status "replied")
           ^                      +-- not supported, first time -> redraft -+
           |                      |                                          |
           +----------------------+------------------------------------------+
                                  +-- not supported again -------> escalate -> END
```

- `check` is a normal node. It writes `check_result`, `check_reason` and `check_attempts` into the state.
- `after_check` is a conditional edge (a function, not a node), like `router`. It returns `"done"`, `"redraft"` or `"escalate"`.
- `redraft` is the draft step again, with the reason for the rejection added to the prompt. It adds 1 to `redrafts`. The edge `redraft -> check` sends the new reply back for review.
- The loop cannot run forever, because `after_check` only returns `"redraft"` while `redrafts` is 0.

### Dry run of case (c): rejected once, fixed by the redraft

This run uses the **mocked** LLM from `tests/test_check.py` (no real Groq call), so the ticket, the five past tickets (T00001 to T00005, similarity 0.94 and below) and the replies are made-up test data. It shows exactly how the state moves.

> Concerns about hospital security systems. Suspected data breach, examining access logs.

Tables show only the fields that changed at each node. After `check` fails, `after_check` sends the ticket to `redraft`; after the second `check` passes, it returns `"done"`.

**Start** (`run_ticket`): `ticket_id = "T3"`, `text`, `attempts = 0`, `reconciled = False`, `redrafts = 0`, `check_attempts = 0`, `confidence_history = []`, `status = "new"`, `steps_log = []`.

**1. After classify**

| Field | Value |
|---|---|
| `category` / `priority` | Technical & IT Support / high |
| `classify_reason` | "breach" |
| `status` | `new` -> `classified` |

**2. After retrieve**

| Field | Value |
|---|---|
| `similar_tickets` | 5 past tickets, top score 0.94 |
| `search_query` | the ticket text |

**3. After score_confidence**

| Field | Value |
|---|---|
| `top_score` / `agreement` | 0.94 / 1.0 |
| `confidence` | 0.8 x 0.94 + 0.2 x 1.0 = **0.952** |

The router returns `"draft"` (0.952 is at least 0.75).

**4. After draft**

| Field | Value |
|---|---|
| `draft_reply` | "We will fix it within 24 hours [T00001]. Please send the access logs [T00002]." |
| `used_ticket_ids` | `["T00001", "T00002"]` (the cited ids) |
| `analysis` | filled in (severity high) |
| `attempts` | 0 -> 1 |
| `status` | `classified` -> `drafted` |

**5. After check (first time)**

The rule check passes: there is a reply, it cites 2 tickets, and both are in the search results. So the LLM check runs and rejects the reply, because the 24 hour promise is not in the cited tickets.

| Field | Value |
|---|---|
| `check_result` | `"not_supported"` |
| `check_reason` | "the reply promises a 24 hour fix that the past tickets never mention" |
| `check_attempts` | 0 -> 1 |
| `status` | `drafted` -> `check_failed` |

`after_check`: not supported and `redrafts` is 0, so it returns `"redraft"`.

**6. After redraft**

The redraft prompt now contains the rejected reply and the reason above.

| Field | Value |
|---|---|
| `draft_reply` | "Please send the access logs so we can investigate [T00001] [T00002]." |
| `attempts` | 1 -> 2 |
| `redrafts` | 0 -> **1** |
| `status` | `check_failed` -> `drafted` |

`check_result` still says `not_supported` here. It is only updated by the next check.

**7. After check (second time)**

| Field | Value |
|---|---|
| `check_result` | `"supported"` |
| `check_reason` | "every claim matches the cited tickets" |
| `check_attempts` | 1 -> 2 |
| `status` | `drafted` -> **`replied`** |

`after_check` returns `"done"`, and the graph ends. LLM calls: classify, draft, check, redraft, check = 5. Path: `classify -> retrieve -> score_confidence -> draft -> check -> redraft -> check`.

If the second check had also said `not_supported`, `after_check` would return `"escalate"` (because `redrafts` is already 1), and the human would receive the reason in `escalation_summary`.
## How this will grow

Each new day adds nodes or edges. The chart gets longer, and the graph gets forks and loops.

```
Day 3   START -> classify -> END

Day 4   START -> classify -> retrieve -> draft -> END   (done)

Day 5-6 router, reconcile/rewrite loops, real escalate   (done, see above)
Day 7   check + one redraft before anything is sent   (done, see above)
        retrieve: search the past tickets (src/search.py) and fill similar_tickets
        draft:    write a reply using those tickets as examples

Later   START -> classify -> retrieve -> draft -> check --+--> END (reply is good)
                                  ^                       |
                                  +--- rewrite search_query, attempts + 1 (not good)

        + analysis node: summary, root cause, suggested fix, severity
        + router (conditional edge): easy tickets go straight to a reply,
          risky or unsure ones (low confidence, high severity) go to a human
        + learn step: save good answers back into the database as source = "learned"
```

Why the state already has fields for later days (`similar_tickets`, `draft_reply`, `confidence`, `attempts`, `search_query`, `check_result`): the chart is designed once, and later nodes simply fill in more boxes. A retry loop works because `attempts` counts how many laps were made, so the graph can stop after a limit.

## First results (20 eval tickets, `openai/gpt-oss-120b`)

- **Category accuracy: 70% (14 of 20).**
- **Priority accuracy: 60% (12 of 20).**
- No ticket fell back to `Unknown`, so the JSON and validation steps worked every time.
- Each call took roughly 0.9 seconds on average.

With only 20 tickets, treat these as rough: one ticket is 5 percentage points.

### Where it gets confused, and why

Most misses are not clear model mistakes. They come from noisy or overlapping labels (see `DATA_NOTES.md`):

1. **Nonsense labels.** T16405 is a question about investment data analytics, labelled *Returns and Exchanges*. The model said *Sales and Pre-Sales*, which is a reasonable reading. No model can guess a label that has nothing to do with the text.
2. **Overlapping categories.** T09076 asks about digital marketing strategies and is labelled *Customer Service*, and the model said *Sales and Pre-Sales*. Both are defensible. T24885 ("advice on optimizing GIMP") is labelled *Technical & IT* and the model said *Product Support*. The same overlap shows up as Product Support swapped with Technical & IT and with Customer Service.
3. **A fair model call against the label.** T14703 says "system outage" and "downtime", so the model chose *Service Outages and Maintenance*. The dataset label is *Technical & IT*. A person could side with either.
4. **Priority is the shakiest label.** The model rates anything with a breach, an outage or a failure as `high`. The dataset often says `medium` or even `low`: T23238, a hospital data breach, is labelled `low`. The model said `high`, which most people would agree with. In 7 of the 8 priority misses the model said a higher priority than the label, so it errs on the cautious side.

Conclusion: a ceiling around 70-80% on this dataset is probably down to the labels, not the code. Later days should not chase category accuracy with prompt tweaks. A better use of effort is to make the agent useful even when the category is a bit off, because the reply depends mostly on the similar tickets it finds.
