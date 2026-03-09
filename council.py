#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
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


def thinking_kwargs(model: str, level: str) -> dict:
    if model.startswith("gemini/"):
        return {"thinking_level": level}
    return {"reasoning_effort": level}  # OpenAI and Anthropic via LiteLLM


async def review_one(model: str, messages: list[dict], temperature: float, thinking: str) -> dict | None:
    extra = thinking_kwargs(model, thinking)
    # Both OpenAI and Anthropic require temperature=1 when extended thinking is active
    if "reasoning_effort" in extra:
        temperature = 1.0
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            **extra,
        )
        raw = response.choices[0].message.content
        parsed = parse_review(raw)
        parsed["model"] = model
        return parsed
    except Exception as e:
        print(f"[warning] {model} failed: {e}", file=sys.stderr)
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
    threshold = 2  # majority for both 2-model and 3-model cases
    if counts["PASS"] >= threshold:
        return "PASS"
    if counts["BLOCK"] >= threshold:
        return "BLOCK"
    return "REVISE"


async def run(models: list[str], mode: str, content: str, purpose: str, static_context: str, extra_context: str, temperature: float, thinking: str) -> dict:
    messages = build_messages(mode, content, purpose, static_context, extra_context)
    raw_results = await asyncio.gather(*[review_one(model, messages, temperature, thinking) for model in models])
    reviews = [r for r in raw_results if r is not None]

    if not reviews:
        raise SystemExit("All model calls failed. Check your API keys and model names.")

    overall_verdict = aggregate_verdict(reviews)

    result = {
        "mode": mode,
        "overall_verdict": overall_verdict,
        "average_score": round(mean(r["score"] for r in reviews), 1),
        "top_strengths": summarize_texts([r["main_strength"] for r in reviews]),
        "top_issues": summarize_texts([r["main_issue"] for r in reviews]),
        "fix_now": (summarize_texts([r["fix_now"] for r in reviews]) or [""])[0],
        "reviews": reviews,
    }
    return result


def get_models(cli_models: str | None) -> list[str]:
    raw = cli_models or os.getenv(
        "COUNCIL_MODELS",
        "gpt-5.4,claude-opus-4-6,gemini/gemini-3.1-pro-preview",
    )
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models specified. Set --models or COUNCIL_MODELS.")
    return models[:3]


VALID_THINKING = {"low", "medium", "high"}


def get_thinking(cli_thinking: str | None) -> str:
    env_thinking = os.getenv("COUNCIL_THINKING", "").strip().lower() or None
    if env_thinking and env_thinking not in VALID_THINKING:
        raise SystemExit(f"Invalid COUNCIL_THINKING value: {env_thinking!r}. Must be one of: low, medium, high")
    if cli_thinking and env_thinking:
        raise SystemExit("Conflict: --thinking and COUNCIL_THINKING are both set. Use only one.")
    return cli_thinking or env_thinking or "medium"


def get_content(input_file: str | None, text: str | None) -> str:
    if text and text.strip():
        return text.strip()
    if input_file:
        return read_text(input_file)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --text, --input-file, or pipe content via stdin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple 3-model LLM council")
    parser.add_argument("--mode", choices=["plan_review", "output_review"], required=True)
    parser.add_argument("--input-file", help="File containing the plan/output to review")
    parser.add_argument("--text", help="Raw text to review")
    parser.add_argument("--purpose", default="", help="Short sentence on what this is for")
    parser.add_argument("--context", default="", help="Per-run situation as inline text")
    parser.add_argument("--context-file", default="", help="Per-run situation from a file")
    parser.add_argument("--static-context", default="", help="Stable background context as inline text")
    parser.add_argument("--static-context-file", default="", help="Stable background context from a file")
    parser.add_argument("--models", default="", help="Comma-separated list of 3 models")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", choices=["low", "medium", "high"], default=None,
                        help="Reasoning effort level (default: medium; overridable via COUNCIL_THINKING)")
    args = parser.parse_args()

    content = get_content(args.input_file, args.text)
    extra_context = "\n\n".join(filter(None, [args.context.strip(), read_text(args.context_file)]))
    static_context = "\n\n".join(filter(None, [args.static_context.strip(), read_text(args.static_context_file)]))
    models = get_models(args.models)
    thinking = get_thinking(args.thinking)

    result = asyncio.run(
        run(
            models=models,
            mode=args.mode,
            content=content,
            purpose=args.purpose,
            static_context=static_context,
            extra_context=extra_context,
            temperature=args.temperature,
            thinking=thinking,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
