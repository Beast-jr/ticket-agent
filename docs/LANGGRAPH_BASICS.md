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

## How this will grow

Each new day adds nodes or edges. The chart gets longer, and the graph gets forks and loops.

```
Day 3   START -> classify -> END

Day 4   START -> classify -> retrieve -> draft -> END   (done)
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
