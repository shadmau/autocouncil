# autocouncil

Run an OpenClaw agent plan or output past three LLMs and get a fast **“good enough to proceed?”** verdict.

A CLI tool for adding lightweight review to OpenClaw agent workflows.

## How it works

1. Send the same plan or output to 3 models in parallel
2. Each returns `PASS`, `REVISE`, or `BLOCK` plus one key issue
3. autocouncil aggregates that into one JSON verdict

## Install

```bash
pip install litellm
```

Set API keys for whichever providers you use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

LiteLLM handles the provider routing. Any model string it supports works here.

## Quickstart

```bash
cat draft_email.txt | python council.py \
  --mode output_review \
  --purpose "Can this be sent externally to a prospect?"
```

## Usage

### Review a plan

```bash
python council.py \
  --mode plan_review \
  --input-file plan.txt \
  --purpose "Should we proceed with this approach?" \
  --static-context-file context.txt
```

### Review an output

```bash
cat draft_email.txt | python council.py \
  --mode output_review \
  --purpose "Can this be sent externally to a prospect?"
```

### Typical pattern

Use a file for stable background context and inline text for the current situation.

```bash
python council.py \
  --mode output_review \
  --input-file draft_email.txt \
  --static-context-file output_context.txt \
  --context "This is external outreach to a first pilot prospect"
```

### Inline text

```bash
python council.py \
  --mode plan_review \
  --text "Step 1: do X. Step 2: do Y." \
  --purpose "Is this plan reasonable?"
```

### Specify models explicitly

```bash
python council.py \
  --mode output_review \
  --input-file report.txt \
  --models "gpt-5.4,claude-opus-4-6,gemini/gemini-3.1-pro-preview"
```

Exactly 3 models are used. If you pass more than 3, the extras are ignored.

## Options

| Flag                    | Default        | Description                               |
| ----------------------- | -------------- | ----------------------------------------- |
| `--mode`                | required       | `plan_review` or `output_review`          |
| `--input-file`          | —              | File to review                            |
| `--text`                | —              | Inline text to review                     |
| `--purpose`             | `""`           | One sentence on what this is for          |
| `--context`             | —              | Per-run situation as inline text          |
| `--context-file`        | —              | Per-run situation from a file             |
| `--static-context`      | —              | Stable background context as inline text  |
| `--static-context-file` | —              | Stable background context from a file     |
| `--models`              | env or default | Comma-separated list of 3 models          |
| `--temperature`         | `0.2`          | Sampling temperature                      |
| `--thinking`            | `medium`       | Reasoning effort: `low`, `medium`, `high` |

Content priority: `--text` > `--input-file` > stdin.

If both `--context` and `--context-file` are given, they are combined.  
Same for `--static-context` and `--static-context-file`.

Default models (if `--models` and `COUNCIL_MODELS` are both unset):

```bash
gpt-5.4,claude-opus-4-6,gemini/gemini-3.1-pro-preview
```

You can also set the default via env:

```bash
export COUNCIL_MODELS="gpt-5.4,claude-opus-4-6,gemini/gemini-3.1-pro-preview"
```

## Output

autocouncil returns a single JSON object to stdout:

```json
{
  "mode": "plan_review",
  "overall_verdict": "PASS",
  "average_score": 7.3,
  "top_strengths": ["Clear objective", "..."],
  "top_issues": ["Missing timeline", "..."],
  "fix_now": "Add a concrete timeline before proceeding.",
  "reviews": [
    {
      "model": "gpt-5.4",
      "verdict": "PASS",
      "score": 8,
      "main_strength": "...",
      "main_issue": "...",
      "fix_now": "..."
    }
  ]
}
```

### Verdicts

- `PASS` — 2+ models voted `PASS`
- `BLOCK` — 2+ models voted `BLOCK`
- `REVISE` — everything else

## `plan_review` vs `output_review`

### `plan_review`

Judges whether a plan is good enough to act on.

Looks for:

- a clear objective
- concrete next steps
- realistic scope
- awareness of important risks or missing information

Use it before you start something.

### `output_review`

Judges whether a finished output is good enough for its intended use.

Looks for:

- correctness
- usefulness
- clarity
- completeness
- trustworthiness for external use

Use it before you ship, send, or rely on something.

## Agent loop

autocouncil is designed to sit inside an OpenClaw agent loop as a lightweight review step.

Typical pattern:

1. the agent drafts a plan
2. autocouncil reviews the plan
3. if the verdict is `PASS`, the agent executes
4. the agent produces an output
5. autocouncil reviews the output
6. if the verdict is `PASS`, the output is used or sent

This helps an agent improve plans and outputs before moving forward, without building a full multi-agent system.

## Static context

Use static context for stable background that should apply to every review.

Examples:

- team defaults
- what “good enough” means in your environment
- bias toward speed vs caution
- expectations for external-facing outputs

Example:

```txt
Small team, moving fast. Plans are operational, not research proposals.
Bias toward action when risk is low. Flag missing information only if it would actually block progress.
External outputs must be accurate and trustworthy.
```

Keep it short and plain-text.

## When to use this

Good for:

- reviewing AI-generated plans before acting on them
- checking outputs before sending
- getting a fast second opinion when uncertain
- adding a simple review gate to an agent pipeline

## Limitations

- every run makes 3 API calls
- costs are roughly 3x a single-model check
- no streaming
- no retries
- models do not see each other’s responses
