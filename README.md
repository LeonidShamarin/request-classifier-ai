**English** | [Українська](README.uk.md)

# AI Request Classifier

Classifies an incoming stream of requests from internal teams (Slack, Telegram, email; free
form, in Ukrainian) into structured fields with an LLM.

Input: `data/input_requests.csv` with the fields `id, channel, timestamp, raw_text`.
Output: `output/output.json` with the full structured result and `output/report.md` with
aggregates by category, priority and department.

The artefacts of a real run are in the repository, in [`output/`](output/): all 18 requests
classified, no processing errors, in 3 minutes 36 seconds. Model: `gemini-3.5-flash-lite`
through the `google-genai` SDK. The code is covered by 35 unit tests that need no network.

Of the optional items in the brief, async batch processing, a Docker image and writing to a
Google Sheet are done. The Telegram digest is not: postponed on purpose, left in the plans.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip

cp .env.example .env
# in .env: GEMINI_API_KEY=<key from https://aistudio.google.com/apikey>

.venv/Scripts/python main.py
```

A run on 18 requests takes ~3.5 min. That is not the model's speed but deliberate pacing for
the free tier quota (5 requests/min).

### Options

| Flag | Default | What it does |
|---|---|---|
| `--input` | `data/input_requests.csv` | Input CSV |
| `--output-dir` | `output` | Where to put `output.json` and `report.md` |
| `--concurrency` | `3` | How many LLM requests run in parallel |
| `--rpm` | `5` | Requests per minute limit. `0` means no limit |
| `--model` | `gemini-3.5-flash-lite` | Gemini model |
| `--max-retries` | `1` | How many times to self-repair invalid output |
| `--sheets` | off | Also write the result to a Google Sheet |

### Environment variables

| Variable | Required | Why |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini key ([free](https://aistudio.google.com/apikey)) |
| `GOOGLE_SHEETS_ID` | no | Sheet id from the URL, for `--sheets` |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | no | Path to the service account JSON |

Keys come only from `.env` / the environment and never reach the repository.

### Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

No network needed: the LLM is mocked, no API key required.

### Docker

```bash
docker build -t ai-request-classifier .
docker run --rm --env-file .env -v ${PWD}/output:/app/output ai-request-classifier
```

Checked end to end: the image builds, and the container with default arguments classifies all
18 requests.

---

## Google Sheets (optional)

The `--sheets` flag also exports the result to a sheet: 15 columns, including
`related_request_id` and the reason for the link.

**Checked on a real sheet:**
[open the result](https://docs.google.com/spreadsheets/d/1xJDFTOtPiwvcaLdtuaR7ejtbtBuzchFJ-2wqmi8LodI/edit?usp=sharing)

![Export result in Google Sheets](docs/sheets-export.png)

Setup:

1. Create a Google Cloud project and enable the [Sheets API](https://console.cloud.google.com/apis/library/sheets.googleapis.com).
2. Create a service account → **Keys → Add key → JSON**, save the file **outside** the repository.
3. Share the sheet with the email from the `client_email` field of that JSON, with **Editor** rights.
4. Put `GOOGLE_SHEETS_ID` and `GOOGLE_SERVICE_ACCOUNT_JSON` in `.env`.

Step 3 is the most common point of failure: without it you get a `PermissionError` even with a
valid key, because to Google a service account is an outside user.

Google API errors deliberately do **not** fail the run: `output.json` and `report.md` are
produced regardless of whether Sheets is reachable.

---

## Data schema

The required fields come straight from the brief. Extensions were added only where a specific
row of the given CSV would otherwise be handled wrongly:

| Field | Why |
|---|---|
| `language` (uk/en/mixed) | REQ-006 is English text with a Ukrainian tail "Хто може взятись?" (who can take it?). Cheap to get together with the other fields, useful for routing. |
| `clarification_reason` | If `needs_clarification=true`, a person needs to know *why* right away. Goes straight into `report.md`. |
| `related_request_id` + `related_reason` | REQ-013 literally says "the same report Olya asked for on Monday" (= REQ-001). Without this, two identical tasks look independent. |
| `confidence` | The model's self-assessment, separate from `needs_clarification`. `needs_clarification` is about the *request text* (vague or not), `confidence` is about the model's certainty in *its own* classification, even of clear text. Different things. |

I also added the category **`інше`** (other). The data has REQ-008 ("thanks for yesterday,
everything works great"): not a request at all, and labelling it "out of scope" (meaning "not
our topic") would be inaccurate. `інше` = "not a request at all": thanks, greetings, noise.

### Why `related_request_id` is a separate pass

The first version passed the model the ids of neighbouring requests in each request's prompt.
**That did not work**: from a bare `REQ-001` there is no way to guess it is about Google Ads, and
REQ-013 was not linked. Showing the full inbox in each of N prompts is O(N²) tokens.

So linking moved into a separate pass (`GeminiClient.link_requests`): after classification, one
call gets the list of `id + category + short_summary` and looks for duplicates. O(N) tokens,
one extra request for the whole batch. The result is validated: ids invented by the model and
self-links are dropped.

It is also more honest architecturally: a link between requests is a property of the batch, not
of a single row. So the field lives in `ProcessedRequest`, not in `LLMClassification`.

---

## Validation and handling invalid LLM output

1. The request goes with `response_mime_type="application/json"` and `response_schema` (the
   Pydantic model is passed to the SDK directly), which already greatly reduces the chance of
   getting non-JSON.
2. Regardless, every response goes through Pydantic. I do not trust the model's structured
   output mode blindly.
3. If parsing or validation fails, **one retry**: the model is shown its own error and asked to
   fix it ("self-repair").
4. If the second attempt also fails, the row is **not lost**. It goes into `output.json` as
   `processing_status="fallback_error"`, `needs_clarification=true`, `category="інше"`, and
   `clarification_reason` holds the technical reason. Such records are listed separately in
   `report.md` so a person sees right away what to sort out by hand.

Aggregates in the report are computed **only over successful records**; otherwise the fallback
placeholders (`інше`/`medium`) would quietly inflate the statistics.

### Four levels of resilience

Split by class of problem, not piled into one retry:

| Level | What it catches | Reaction |
|---|---|---|
| Rate limiter | the 5 requests/min quota | spreading calls over time |
| Backoff | per-minute 429, 5xx, network | the pause the server itself named |
| Circuit breaker | the **daily** quota | fail immediately, no pointless calls |
| Self-repair | invalid JSON / Pydantic | show the model its error |

---

## Where it breaks / limits

Below is what I actually ran into while getting this to work, not a hypothetical list.

- **A daily quota is not a per-minute quota.** The most expensive lesson. Gemini returns
  `retryDelay` (e.g. `25s`) in *every* 429, even when the daily quota is exhausted. The first
  version obeyed that advice blindly and hung for 14 minutes, hopelessly retrying. Now the daily
  quota is recognised by `quotaId` (`PerDay`), trips a circuit breaker, and `main.py` warns
  explicitly that the result is incomplete. **Not closed**: partial results are not finished
  automatically; that has to be re-run by hand.
- **Choosing a model is stricter than it looks.** `gemini-2.5-flash` returns 404 "no longer
  available to new users". `thinking_budget=0` on Gemini 3.x is 400 INVALID_ARGUMENT; only
  `thinking_level="low"` works. A config copied from an old example fails on every request.
- **TLS inspection.** On networks with a corporate proxy or antivirus, `certifi` lacks the local
  root and every call fails with `CERTIFICATE_VERIFY_FAILED`. Closed with `truststore`
  (verification against the OS store). Certificate verification was **not** turned off: that
  would be the same as working without TLS.
- **Invalid LLM output**: closed (retry + fallback record). Not closed: if the model's answer
  parses *validly* but is wrong in substance (the wrong category), nothing catches it
  automatically. That needs an eval set with golden examples.
- **Large volume**: the pipeline is async with a semaphore and a rate limiter, so 18 requests or
  1800 is the same code. But there is no persistence: if the process dies on request 1500 of
  2000, everything starts from zero. For real volume, write intermediate results to SQLite and
  retry only the unfinished ids.
- **Non-determinism**: `temperature=0.0` reduces variation but does not guarantee identical
  calls. Measured: two independent runs on the same CSV differed in **1 field out of 90**
  (98.9% match): in REQ-010 `target_department` was `null` once and `маркетинг` (marketing) the
  other time. Categories and priorities matched completely. So it is exactly the borderline
  cases, where the department is not named in the text, that drift. For stability: a cache by
  the hash of `raw_text`.
- **Token cost**: on 18 requests, within the free tier. The system prompt dominates the input
  tokens. At scale: (a) a cache by text hash, (b) a shorter system prompt, (c) the batch API
  instead of online calls. `thinking_level="low"` already removes unneeded reasoning tokens:
  classification against a fixed schema does not need them.
- **`related_request_id`** works within one batch. In a real stream (Slack/Telegram in real
  time) it needs a search over the accumulated history with embeddings, not over "neighbours in
  the file".
- **`--sheets` needs a full re-run.** To update the sheet you currently have to classify all
  requests again. Exporting an existing `output.json` would make more sense; not done yet.

---

## What I would do next

1. **An eval set**: 20-30 hand-labelled examples + a script that computes accuracy after every
   prompt change. Right now prompt quality is not measured quantitatively, only checked by hand
   on 18 requests against the control cases.
2. **Persistence**: SQLite instead of "everything in memory", so large volumes are processed
   incrementally and recover after crashes and an exhausted quota.
3. **Deduplication with embeddings** on top of `related_request_id`: clustering by similarity of
   `short_summary`, not only explicit mentions in the text.
4. **A Telegram digest**: `build_report()` already returns ready text; only a sender module is
   left.
5. **CI**: GitHub Actions with `pytest` on every push; the tests do not depend on the network,
   so it is cheap.

---

## Structure

```
main.py              CLI: arguments, env, orchestration, writing artefacts
src/schema.py        Pydantic contract: LLMClassification + ProcessedRequest
src/llm_client.py    Gemini: prompt, structured output, retry, backoff, quotas
src/pipeline.py      Reading the CSV, async processing, fallback, linking pass
src/report.py        Aggregation into a Markdown report
src/sheets.py        Optional export to a Google Sheet
tests/               Unit tests with a fake client, no network
data/                Input CSV from the brief
output/              Artefacts of a real run
```
