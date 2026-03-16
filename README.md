# autocouncil

AutoCouncil is an LLM council for reviewing OpenClaw agent plans and outputs.

Runs Claude CLI, Codex CLI, or API models as council members. Each member scores the OpenClaw submission independently; the council aggregates their verdicts into a single judgment.

## How it works

1. Send the same plan or output to 1–3 council members in parallel
2. Each member returns `PASS` or `REVISE` plus a severity, a needs_input question (if context is missing), and one key issue
3. The council aggregates the individual verdicts into one JSON judgment
4. The agent can revise and re-submit until the council is satisfied

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
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

### Council members

Use `--members` to run council members through local CLI tools instead of (or alongside) API calls.

Format: `<backend>:<model>:<effort>`, comma-separated. Up to 3 members. Effort: `low`, `medium`, or `high`.

Backends: `litellm` (API via LiteLLM), `claude_cli` (local `claude` binary), `codex_cli` (local `codex` binary).

```bash
# Codex only
python council.py --mode plan_review --input-file plan.txt \
  --members "codex_cli:gpt-5.4:high"

# Claude CLI + Codex CLI
python council.py --mode output_review --input-file report.txt \
  --members "claude_cli:claude-opus-4-6:high,codex_cli:gpt-5.4:high"

# Mixed: API + local CLIs
python council.py --mode plan_review --input-file plan.txt \
  --members "litellm:gemini/gemini-3.1-pro-preview:medium,claude_cli:claude-opus-4-6:high,codex_cli:gpt-5.4:high"

# Default API path (no --members needed, uses hardcoded litellm defaults)
python council.py --mode plan_review --input-file plan.txt
```

See available backends with:

```bash
python council.py --doctor
```

1 to 3 members are supported. If a member fails (e.g. missing binary, expired auth, bad API key), the council continues with the remaining successful verdicts.

## Options

| Flag                    | Default        | Description                                                                |
| ----------------------- | -------------- | -------------------------------------------------------------------------- |
| `--mode`                | required       | `plan_review` or `output_review`                                           |
| `--doctor`              | —              | Show available backends and example `--members` strings                    |
| `--input-file`          | —              | File to review                                                             |
| `--text`                | —              | Inline text to review                                                      |
| `--purpose`             | `""`           | One sentence on what this is for                                           |
| `--context`             | —              | Per-run situation as inline text                                           |
| `--context-file`        | —              | Per-run situation from a file                                              |
| `--static-context`      | —              | Stable background context as inline text                                   |
| `--static-context-file` | —              | Stable background context from a file                                      |
| `--members`             | env or default | Council members: `<backend>:<model>:<effort>,...` (env: `COUNCIL_MEMBERS`) |

Content priority: `--text` > `--input-file` > stdin.

If both `--context` and `--context-file` are given, they are combined.
Same for `--static-context` and `--static-context-file`.

Default members (if `--members` and `COUNCIL_MEMBERS` are both unset): the three default API models via litellm.

You can also set defaults via env:

```bash
export COUNCIL_MEMBERS="claude_cli:claude-opus-4-6:high,codex_cli:gpt-5.4:high,litellm:gemini/gemini-3.1-pro-preview:medium"
```

## Output

autocouncil returns a single JSON object to stdout:

```json
{
  "mode": "plan_review",
  "overall_verdict": "PASS",
  "severity": "low",
  "needs_input": "",
  "average_score": 7.3,
  "top_strengths": ["Clear objective", "..."],
  "top_issues": ["Missing timeline", "..."],
  "fix_now": "Add a concrete timeline before proceeding.",
  "reviews": [
    {
      "model": "gpt-5.4",
      "verdict": "PASS",
      "score": 8,
      "severity": "low",
      "needs_input": "",
      "main_strength": "...",
      "main_issue": "...",
      "fix_now": "..."
    }
  ]
}
```

### Verdicts

- `PASS` — 2+ council members voted `PASS`
- `REVISE` — everything else

### Signals

- `severity` — highest severity across council members: `low`, `medium`, or `high`
- `needs_input` — if any member needed more context to assess confidently, their specific question appears here; empty otherwise

## `plan_review` vs `output_review`

### `plan_review`

The council judges whether a plan is good enough to act on: clear objective, concrete next steps, realistic scope, and awareness of key risks.

### `output_review`

The council judges whether an output is good enough for its intended use: correctness, usefulness, clarity, completeness, and trustworthiness for external use.

## Agent loop

autocouncil is designed to sit inside an OpenClaw agent loop as a self-improvement step.

Typical pattern:

1. the agent drafts a plan or output
2. the council reviews it
3. if the verdict is `REVISE`, the agent improves it and re-submits
4. if `needs_input` is non-empty, the agent surfaces the missing context before retrying
5. if the verdict is `PASS`, the agent proceeds

## Using AutoCouncil with OpenClaw

AutoCouncil is designed to plug into an OpenClaw workspace as a review loop, with the loop defined in `AGENTS.md` and/or `HEARTBEAT.md`.

For a clean setup:

- keep one local installation path
- document the local command usage in `TOOLS.md`
- reuse an existing working installation instead of creating duplicate copies
- keep the integration minimal

## Static context

Use static context for stable background that applies to every council run.

Examples:

- team defaults
- what "good enough" means in your environment
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

- running iterative self-improvement loops on plans and outputs
- deciding whether a draft is good enough to ship or needs another revision
