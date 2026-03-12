#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from statistics import mean

import litellm
from litellm import acompletion

litellm.drop_params = True

PLAN_SYSTEM = """You are one member of a 3-model review council.
Your job is to judge whether a plan is good enough to proceed now.

A good plan, at minimum, has:
- a clear objective
- concrete next steps in a sensible order
- realistic scope for the situation
- awareness of important risks or missing information
- a likely path to a useful result or learning

Use only the provided context. Do not assume missing facts.
Do not reward polished language. Judge practicality.
Be direct and concise.
Return STRICT JSON only with keys:
verdict, score, main_strength, main_issue, fix_now

Rules:
- verdict must be one of: PASS, REVISE, BLOCK
- score must be an integer from 1 to 10
- main_strength, main_issue, fix_now should each be one short sentence
"""

OUTPUT_SYSTEM = """You are one member of a 3-model review council.
Your job is to judge whether an output is good enough for its intended use.

A good output, at minimum, is:
- correct or at least not misleading
- useful for the stated purpose
- clear and easy to act on
- complete enough for its intended use
- trustworthy if it will be used externally

Use only the provided context. Do not assume missing facts.
Do not reward style over substance.
Be direct and concise.
Return STRICT JSON only with keys:
verdict, score, main_strength, main_issue, fix_now

Rules:
- verdict must be one of: PASS, REVISE, BLOCK
- score must be an integer from 1 to 10
- main_strength, main_issue, fix_now should each be one short sentence
"""

VALID_THINKING = {"low", "medium", "high"}
VALID_BACKENDS = {"litellm", "claude_cli", "codex_cli"}
TEMPERATURE = 0.2

def read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_review(raw: str) -> dict:
    cleaned = strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))

    verdict = str(data.get("verdict", "REVISE")).upper().strip()
    if verdict not in {"PASS", "REVISE", "BLOCK"}:
        verdict = "REVISE"

    try:
        score = int(round(float(data.get("score", 5))))
    except Exception:
        score = 5
    score = max(1, min(10, score))

    return {
        "verdict": verdict,
        "score": score,
        "main_strength": str(data.get("main_strength", "")).strip(),
        "main_issue": str(data.get("main_issue", "")).strip(),
        "fix_now": str(data.get("fix_now", "")).strip(),
    }


def build_messages(mode: str, content: str, purpose: str, static_context: str, extra_context: str) -> list[dict]:
    system = PLAN_SYSTEM if mode == "plan_review" else OUTPUT_SYSTEM

    if static_context:
        system += "\n\nStatic context:\n" + static_context.strip()

    user_parts = []
    if purpose:
        user_parts.append("Purpose:\n" + purpose.strip())
    if extra_context:
        user_parts.append("Context:\n" + extra_context.strip())
    user_parts.append("Submission to review:\n" + content.strip())

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ── JSON schema for structured output ─────────────────────────────────────────

def build_review_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "REVISE", "BLOCK"]},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            "main_strength": {"type": "string"},
            "main_issue": {"type": "string"},
            "fix_now": {"type": "string"},
        },
        "required": ["verdict", "score", "main_strength", "main_issue", "fix_now"],
        "additionalProperties": False,
    }


# ── Member parsing ─────────────────────────────────────────────────────────────

def parse_members(raw: str) -> list[dict]:
    members = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) < 2:
            raise SystemExit(f"Invalid member spec: {part!r}. Format: <backend>:<model>:<effort>")
        backend = pieces[0].strip()
        if backend not in VALID_BACKENDS:
            raise SystemExit(f"Unknown backend: {backend!r}. Valid: {', '.join(sorted(VALID_BACKENDS))}")
        # Effort is required as last segment
        if len(pieces) >= 3 and pieces[-1].strip().lower() in VALID_THINKING:
            effort = pieces[-1].strip().lower()
            model = ":".join(pieces[1:-1]).strip()
        else:
            raise SystemExit(
                f"Missing effort in member spec: {part!r}. "
                f"Format: <backend>:<model>:<effort>  (effort: low | medium | high)"
            )
        if not model:
            raise SystemExit(f"Empty model in member spec: {part!r}")
        members.append({"backend": backend, "model": model, "effort": effort})
    if not members:
        raise SystemExit("No members specified.")
    return members[:3]


def get_members(cli_members: str | None) -> list[dict]:
    raw = (cli_members or os.getenv("COUNCIL_MEMBERS", "")).strip()
    if raw:
        return parse_members(raw)
    # Fall back to COUNCIL_MODELS / hardcoded defaults via litellm
    default = os.getenv(
        "COUNCIL_MODELS",
        "openai/responses/gpt-5.4,claude-opus-4-6,gemini/gemini-3.1-pro-preview",
    )
    models = [m.strip() for m in default.split(",") if m.strip()]
    if not models:
        raise SystemExit("No members specified. Set --members or COUNCIL_MEMBERS.")
    return [{"backend": "litellm", "model": m, "effort": "medium"} for m in models[:3]]


# ── Backend detection ──────────────────────────────────────────────────────────

def detect_backends() -> dict:
    return {
        "litellm": True,
        "claude_cli": shutil.which("claude") is not None,
        "codex_cli": shutil.which("codex") is not None,
    }


# ── LiteLLM backend ────────────────────────────────────────────────────────────

def thinking_kwargs(model: str, level: str) -> dict:
    m = model.lower()
    if m.startswith("gemini/"):
        return {"thinking_level": level}
    if "claude" in m:
        return {"reasoning_effort": level}
    if m.startswith("openai/responses/"):
        mapped = "xhigh" if level == "high" else level
        return {"reasoning": {"effort": mapped}}
    return {}


async def review_one_litellm(member: dict, messages: list[dict]) -> dict | None:
    model = member["model"]
    effort = member["effort"]
    extra = thinking_kwargs(model, effort)
    if not extra:
        print(f"[info] {model}: reasoning_effort not supported — running without extended thinking", file=sys.stderr)
    temperature = 1.0 if ("reasoning_effort" in extra or "reasoning" in extra) else TEMPERATURE
    fmt = {} if model.lower().startswith("openai/responses/") else {"response_format": {"type": "json_object"}}
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            **fmt,
            **extra,
        )
        raw = response.choices[0].message.content
        parsed = parse_review(raw)
        parsed["model"] = model.removeprefix("openai/responses/")
        return parsed
    except Exception as e:
        print(f"[warning] litellm:{model} failed: {e}", file=sys.stderr)
        return None


# ── Claude CLI backend ─────────────────────────────────────────────────────────

async def review_one_claude_cli(member: dict, messages: list[dict]) -> dict | None:
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    effort = member["effort"]

    cmd = [
        "claude", "-p", user_msg,
        "--model", member["model"],
        "--output-format", "json",
        "--json-schema", json.dumps(build_review_schema()),
        "--system-prompt", system_msg,
    ]
    env = os.environ.copy()
    env["CLAUDE_CODE_EFFORT_LEVEL"] = effort
    env.pop("CLAUDECODE", None)  # allow nested claude invocation

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("timed out after 120s")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"exited {proc.returncode}: {err[:300]}")

        raw = stdout.decode(errors="replace").strip()
        # claude --output-format json: structured_output holds schema-validated result
        try:
            outer = json.loads(raw)
            if isinstance(outer.get("structured_output"), dict):
                inner_raw = json.dumps(outer["structured_output"])
            else:
                result_field = outer.get("result", raw)
                inner_raw = result_field if isinstance(result_field, str) else json.dumps(result_field)
        except json.JSONDecodeError:
            inner_raw = raw

        parsed = parse_review(inner_raw)
        parsed["model"] = f"claude_cli:{member['model']}"
        return parsed
    except Exception as e:
        print(f"[warning] claude_cli:{member['model']} failed: {e}", file=sys.stderr)
        return None


# ── Codex CLI backend ──────────────────────────────────────────────────────────

async def review_one_codex_cli(member: dict, messages: list[dict]) -> dict | None:
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    effort = member["effort"]

    schema_fd, schema_path = tempfile.mkstemp(suffix=".json")
    output_fd, output_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(schema_fd, "w") as f:
            json.dump(build_review_schema(), f)
        os.close(output_fd)

        prompt = f"{system_msg}\n\n{user_msg}"
        cmd = [
            "codex", "exec",
            "--model", member["model"],
            "--output-schema", schema_path,
            "--output-last-message", output_path,
            "--ephemeral",
            "--skip-git-repo-check",
            "-c", "approval_policy=never",
            "-c", f"model_reasoning_effort={effort}",
            "-s", "read-only",
            prompt,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("timed out after 120s")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"exited {proc.returncode}: {err[:300]}")

        with open(output_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        parsed = parse_review(raw)
        parsed["model"] = f"codex_cli:{member['model']}"
        return parsed
    except Exception as e:
        print(f"[warning] codex_cli:{member['model']} failed: {e}", file=sys.stderr)
        return None
    finally:
        for path in (schema_path, output_path):
            try:
                os.unlink(path)
            except OSError:
                pass

async def review_one_member(member: dict, messages: list[dict]) -> dict | None:
    backend = member["backend"]
    if backend == "litellm":
        return await review_one_litellm(member, messages)
    if backend == "claude_cli":
        return await review_one_claude_cli(member, messages)
    if backend == "codex_cli":
        return await review_one_codex_cli(member, messages)
    print(f"[warning] Unknown backend: {backend!r}", file=sys.stderr)
    return None

def summarize_texts(texts: list[str]) -> list[str]:
    texts = [t.strip() for t in texts if t and t.strip()]
    counts = Counter(texts)
    ordered = sorted(counts.items(), key=lambda x: (-x[1], texts.index(x[0])))
    return [text for text, _count in ordered[:3]]


def aggregate_verdict(reviews: list[dict]) -> str:
    counts = Counter(r["verdict"] for r in reviews)
    n = len(reviews)
    if n == 1:
        return reviews[0]["verdict"]
    threshold = 2
    if counts["PASS"] >= threshold:
        return "PASS"
    if counts["BLOCK"] >= threshold:
        return "BLOCK"
    return "REVISE"



async def run(
    members: list[dict],
    mode: str,
    content: str,
    purpose: str,
    static_context: str,
    extra_context: str,
) -> dict:
    messages = build_messages(mode, content, purpose, static_context, extra_context)
    raw_results = await asyncio.gather(*[review_one_member(m, messages) for m in members])
    reviews = [r for r in raw_results if r is not None]

    if not reviews:
        raise SystemExit("All council members failed. Check your backends and credentials.")

    overall_verdict = aggregate_verdict(reviews)
    return {
        "mode": mode,
        "overall_verdict": overall_verdict,
        "average_score": round(mean(r["score"] for r in reviews), 1),
        "top_strengths": summarize_texts([r["main_strength"] for r in reviews]),
        "top_issues": summarize_texts([r["main_issue"] for r in reviews]),
        "fix_now": (summarize_texts([r["fix_now"] for r in reviews]) or [""])[0],
        "reviews": reviews,
    }

def cmd_doctor() -> None:
    backends = detect_backends()
    print("Available backends:")
    print(f"  litellm   : available (requires API keys)")
    print(f"  claude_cli: {'found on PATH' if backends['claude_cli'] else 'not found on PATH'}")
    print(f"  codex_cli : {'found on PATH' if backends['codex_cli'] else 'not found on PATH'}")
    print()
    print("Example --members strings:")
    print('  --members "codex_cli:gpt-5.4:high"')
    print('  --members "claude_cli:claude-opus-4-6:high,codex_cli:gpt-5.4:high"')
    print('  --members "litellm:gemini/gemini-3.1-pro-preview:medium,claude_cli:claude-opus-4-6:high,codex_cli:gpt-5.4:medium"')

def get_content(input_file: str | None, text: str | None) -> str:
    if text and text.strip():
        return text.strip()
    if input_file:
        return read_text(input_file)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --text, --input-file, or pipe content via stdin")

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple 3-member LLM review council")
    parser.add_argument("--doctor", action="store_true", help="Show available backends and example --members strings")
    parser.add_argument("--mode", choices=["plan_review", "output_review"])
    parser.add_argument("--input-file", help="File containing the plan/output to review")
    parser.add_argument("--text", help="Raw text to review")
    parser.add_argument("--purpose", default="", help="Short sentence on what this is for")
    parser.add_argument("--context", default="", help="Per-run situation as inline text")
    parser.add_argument("--context-file", default="", help="Per-run situation from a file")
    parser.add_argument("--static-context", default="", help="Stable background context as inline text")
    parser.add_argument("--static-context-file", default="", help="Stable background context from a file")
    parser.add_argument("--members", default="",
                        help="Council members: <backend>:<model>:<effort>,... (backends: litellm, claude_cli, codex_cli)")
    args = parser.parse_args()

    if args.doctor:
        cmd_doctor()
        return

    if not args.mode:
        parser.error("--mode is required")

    content = get_content(args.input_file, args.text)
    extra_context = "\n\n".join(filter(None, [args.context.strip(), read_text(args.context_file)]))
    static_context = "\n\n".join(filter(None, [args.static_context.strip(), read_text(args.static_context_file)]))
    members = get_members(args.members or None)

    result = asyncio.run(
        run(
            members=members,
            mode=args.mode,
            content=content,
            purpose=args.purpose,
            static_context=static_context,
            extra_context=extra_context,
        )
    )

    log_dir = os.getenv("COUNCIL_LOG_DIR", "").strip()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_entry = {
            "timestamp": ts,
            "mode": args.mode,
            "members": members,
            "purpose": args.purpose,
            "static_context": static_context,
            "extra_context": extra_context,
            "content": content,
            "result": result,
        }
        log_path = os.path.join(log_dir, f"council_{ts}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
