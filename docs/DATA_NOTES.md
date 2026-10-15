# Data Notes (Day 1)

Source: Tobias Bueck's "Customer IT Support Ticket Dataset" (Kaggle), file `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, now at `data/raw/tickets.csv`.

**Important: this data is synthetic.** It was written by an AI model, not by real customers. Tickets sound polished and repetitive, labels are cleaner than real life in some places and oddly wrong in others, and results here will look better than they would on real tickets. Fine for a portfolio project, but I should say so honestly in the README.

## How many rows can I use?

| Step | Rows |
|---|---|
| Whole file (English + German) | 28,587 |
| English only | 16,338 |
| After removing rows with an empty body or answer (3 rows) | **16,335** |
| Duplicate rows found | 0 |

Notes:
- 2,606 English tickets have **no subject**. I kept them, because the body and answer are what matter. The agent should not rely on the subject.
- The `tag_5` to `tag_8` columns are mostly empty. I will ignore tags for now.

## Queues (the category)

Original 10 queues, English, cleaned:

| Queue | Rows |
|---|---|
| Technical Support | 4,736 |
| Product Support | 3,071 |
| Customer Service | 2,410 |
| IT Support | 1,942 |
| Billing and Payments | 1,595 |
| Returns and Exchanges | 820 |
| Service Outages and Maintenance | 664 |
| Sales and Pre-Sales | 513 |
| Human Resources | 348 |
| General Inquiry | 236 |

### Suggested final list: 8 categories

| Final category | Made from | Rows |
|---|---|---|
| Technical & IT Support | Technical Support + IT Support | 6,678 |
| Product Support | Product Support | 3,071 |
| Customer Service | Customer Service + General Inquiry | 2,646 |
| Billing and Payments | Billing and Payments | 1,595 |
| Returns and Exchanges | Returns and Exchanges | 820 |
| Service Outages and Maintenance | Service Outages and Maintenance | 664 |
| Sales and Pre-Sales | Sales and Pre-Sales | 513 |
| Human Resources | Human Resources | 348 |
| **Total** | | **16,335** |

Why merge:
- **Technical Support and IT Support** read almost the same (VPN, login, network, connection problems). Nobody, including me, could reliably tell them apart.
- **General Inquiry** is tiny (236) and by definition vague, so it fits best with Customer Service.
- **Human Resources** (348) is small and its tickets look like ordinary IT problems (see below). If the classifier struggles with it on later days, drop it and use 7 categories.

## Priority counts (cleaned)

| Priority | Rows |
|---|---|
| medium | 6,616 |
| high | 6,345 |
| low | 3,374 |

## Type counts (cleaned)

| Type | Rows |
|---|---|
| Incident | 6,571 |
| Request | 4,665 |
| Problem | 3,394 |
| Change | 1,705 |

## Data problems I noticed

1. **Placeholders.** The dataset hides personal info behind tags like `<name>`, `<tel_num>`, `<acc_num>`, `<email>`, `<br>`.
   - 518 bodies and 8,010 answers (about half of all answers) contain one.
   - Most common: `<tel_num>` (~6,300 times), `<name>` (~3,600), `<acc_num>` (~1,500), `<br>` (~1,300, a leftover line break).
   - Plan: the agent's replies must not copy these tags to the customer. Clean or strip them before using answers as examples.
2. **Questionable queue labels.** Many tickets sit in a queue that does not match their text:
   - A hospital data-security question was labelled *Returns and Exchanges*. Only 97 of 820 tickets in that queue even mention "return", "exchange" or "refund".
   - A question about data analytics tools was labelled *Billing and Payments*.
   - *Human Resources* tickets are about office app access, login problems and peripheral devices, not HR topics.
   - Billing words (invoice, refund, payment) do show up mostly in *Billing and Payments* (928 of 934), so that queue is the most trustworthy one.
   - Expect a ceiling on classifier accuracy because the labels themselves are noisy.
3. **Very short tickets.** 181 bodies have 5 words or fewer (for example "Our system" or "Requesting Assistance"), and 753 have fewer than 10. Not enough information to classify or answer.
4. **Very short answers.** 115 answers have 5 words or fewer (for example "Will call soon", "Express apologies"), and 296 have fewer than 10. These are poor examples for the agent to learn from.
5. **Long tickets.** Bodies reach 168 words and answers 135. Not truly long, so no cutting needed. Only 41 bodies are over 120 words.
6. **Typical size.** Median body is 55 words and median answer is 59 words, so everything is short and fits easily in an LLM prompt.
7. **Some answers are vague.** Many just say "we will look into it" without solving anything. When the agent retrieves past answers, prefer longer, more helpful ones.
8. **Priority looks unreliable.** A ticket with "Urgent assistance is needed" was labelled `low`. Treat priority labels with caution.

## Decisions for next days

- English only, drop 3 rows with empty body or answer, keep subject-less tickets.
- Use the 8-category list above (or 7 without HR).
- Strip or handle placeholders before using answers.
- Consider filtering out bodies under ~10 words and answers under ~10 words when building the answer-search database.
