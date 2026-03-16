#!/usr/bin/env python3
"""Unit tests for council.py — run with: python test_council.py"""

import sys
import unittest

sys.path.insert(0, ".")

from council import (
    aggregate_needs_input,
    aggregate_severity,
    aggregate_verdict,
    build_review_schema,
    parse_review,
    strip_fences,
    summarize_texts,
)


def make_review(verdict="PASS", score=8, severity="low", needs_input="",
                strength="Good", issue="Minor", fix="Fix it"):
    return {
        "verdict": verdict,
        "score": score,
        "severity": severity,
        "needs_input": needs_input,
        "main_strength": strength,
        "main_issue": issue,
        "fix_now": fix,
    }


class TestParseReview(unittest.TestCase):

    def _parse(self, d):
        import json
        return parse_review(json.dumps(d))

    def test_valid_pass(self):
        r = self._parse({"verdict": "PASS", "score": 9, "severity": "low",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["score"], 9)
        self.assertEqual(r["severity"], "low")
        self.assertEqual(r["needs_input"], "")

    def test_valid_revise(self):
        r = self._parse({"verdict": "REVISE", "score": 4, "severity": "high",
                         "needs_input": "What is the deadline?",
                         "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["verdict"], "REVISE")
        self.assertEqual(r["severity"], "high")
        self.assertEqual(r["needs_input"], "What is the deadline?")

    def test_block_falls_back_to_revise(self):
        r = self._parse({"verdict": "BLOCK", "score": 2, "severity": "high",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["verdict"], "REVISE")

    def test_unknown_verdict_falls_back(self):
        r = self._parse({"verdict": "GARBAGE", "score": 5, "severity": "medium",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["verdict"], "REVISE")

    def test_missing_severity_defaults_to_medium(self):
        r = self._parse({"verdict": "PASS", "score": 7,
                         "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["severity"], "medium")

    def test_invalid_severity_defaults_to_medium(self):
        r = self._parse({"verdict": "PASS", "score": 7, "severity": "critical",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["severity"], "medium")

    def test_missing_needs_input_defaults_to_empty(self):
        r = self._parse({"verdict": "PASS", "score": 7, "severity": "low",
                         "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["needs_input"], "")

    def test_score_clamped_low(self):
        r = self._parse({"verdict": "PASS", "score": -5, "severity": "low",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["score"], 1)

    def test_score_clamped_high(self):
        r = self._parse({"verdict": "PASS", "score": 999, "severity": "low",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["score"], 10)

    def test_score_float_rounded(self):
        r = self._parse({"verdict": "PASS", "score": 7.7, "severity": "low",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["score"], 8)

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"verdict":"PASS","score":8,"severity":"low","needs_input":"","main_strength":"s","main_issue":"i","fix_now":"f"}\n```'
        r = parse_review(raw)
        self.assertEqual(r["verdict"], "PASS")

    def test_embedded_json_extracted(self):
        raw = 'Some text before {"verdict":"REVISE","score":5,"severity":"medium","needs_input":"","main_strength":"s","main_issue":"i","fix_now":"f"} after'
        r = parse_review(raw)
        self.assertEqual(r["verdict"], "REVISE")

    def test_case_insensitive_verdict(self):
        r = self._parse({"verdict": "pass", "score": 8, "severity": "low",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["verdict"], "PASS")

    def test_severity_case_insensitive(self):
        r = self._parse({"verdict": "PASS", "score": 8, "severity": "HIGH",
                         "needs_input": "", "main_strength": "s", "main_issue": "i", "fix_now": "f"})
        self.assertEqual(r["severity"], "high")


class TestAggregateVerdict(unittest.TestCase):

    def test_single_pass(self):
        self.assertEqual(aggregate_verdict([make_review("PASS")]), "PASS")

    def test_single_revise(self):
        self.assertEqual(aggregate_verdict([make_review("REVISE")]), "REVISE")

    def test_two_pass_one_revise(self):
        reviews = [make_review("PASS"), make_review("PASS"), make_review("REVISE")]
        self.assertEqual(aggregate_verdict(reviews), "PASS")

    def test_one_pass_two_revise(self):
        reviews = [make_review("PASS"), make_review("REVISE"), make_review("REVISE")]
        self.assertEqual(aggregate_verdict(reviews), "REVISE")

    def test_all_revise(self):
        reviews = [make_review("REVISE")] * 3
        self.assertEqual(aggregate_verdict(reviews), "REVISE")

    def test_all_pass(self):
        reviews = [make_review("PASS")] * 3
        self.assertEqual(aggregate_verdict(reviews), "PASS")

    def test_two_reviews_one_each(self):
        reviews = [make_review("PASS"), make_review("REVISE")]
        self.assertEqual(aggregate_verdict(reviews), "REVISE")

    def test_two_reviews_both_pass(self):
        reviews = [make_review("PASS"), make_review("PASS")]
        self.assertEqual(aggregate_verdict(reviews), "PASS")


class TestAggregateSeverity(unittest.TestCase):

    def test_all_low(self):
        reviews = [make_review(severity="low")] * 3
        self.assertEqual(aggregate_severity(reviews), "low")

    def test_escalates_to_high(self):
        reviews = [make_review(severity="low"), make_review(severity="medium"), make_review(severity="high")]
        self.assertEqual(aggregate_severity(reviews), "high")

    def test_medium_wins_over_low(self):
        reviews = [make_review(severity="low"), make_review(severity="medium")]
        self.assertEqual(aggregate_severity(reviews), "medium")

    def test_single_review(self):
        self.assertEqual(aggregate_severity([make_review(severity="high")]), "high")

    def test_invalid_severity_treated_as_medium(self):
        # aggregate_severity uses _SEVERITY_ORDER.get(s, 1) — unknown maps to medium rank
        reviews = [make_review(severity="low"), make_review(severity="unknown")]
        self.assertEqual(aggregate_severity(reviews), "medium")


class TestAggregateNeedsInput(unittest.TestCase):

    def test_all_empty(self):
        reviews = [make_review(needs_input="")] * 3
        self.assertEqual(aggregate_needs_input(reviews), "")

    def test_first_non_empty_returned(self):
        reviews = [
            make_review(needs_input=""),
            make_review(needs_input="What is the target audience?"),
            make_review(needs_input="What is the deadline?"),
        ]
        self.assertEqual(aggregate_needs_input(reviews), "What is the target audience?")

    def test_single_with_question(self):
        self.assertEqual(aggregate_needs_input([make_review(needs_input="Why?")]), "Why?")

    def test_whitespace_only_treated_as_empty(self):
        reviews = [make_review(needs_input="   "), make_review(needs_input="Real question?")]
        self.assertEqual(aggregate_needs_input(reviews), "Real question?")


class TestBuildReviewSchema(unittest.TestCase):

    def test_verdict_enum_no_block(self):
        schema = build_review_schema()
        self.assertIn("verdict", schema["properties"])
        enum = schema["properties"]["verdict"]["enum"]
        self.assertNotIn("BLOCK", enum)
        self.assertIn("PASS", enum)
        self.assertIn("REVISE", enum)

    def test_severity_field_present(self):
        schema = build_review_schema()
        self.assertIn("severity", schema["properties"])
        self.assertEqual(schema["properties"]["severity"]["enum"], ["low", "medium", "high"])

    def test_needs_input_field_present(self):
        schema = build_review_schema()
        self.assertIn("needs_input", schema["properties"])
        self.assertEqual(schema["properties"]["needs_input"]["type"], "string")

    def test_required_fields(self):
        schema = build_review_schema()
        required = schema["required"]
        for field in ["verdict", "score", "severity", "needs_input", "main_strength", "main_issue", "fix_now"]:
            self.assertIn(field, required)

    def test_no_additional_properties(self):
        schema = build_review_schema()
        self.assertFalse(schema["additionalProperties"])


class TestStripFences(unittest.TestCase):

    def test_no_fence(self):
        self.assertEqual(strip_fences('{"a":1}'), '{"a":1}')

    def test_json_fence(self):
        self.assertEqual(strip_fences('```json\n{"a":1}\n```'), '{"a":1}')

    def test_plain_fence(self):
        self.assertEqual(strip_fences('```\n{"a":1}\n```'), '{"a":1}')


class TestSummarizeTexts(unittest.TestCase):

    def test_deduplication_and_order(self):
        texts = ["a", "b", "a", "c", "b", "a"]
        result = summarize_texts(texts)
        self.assertEqual(result[0], "a")  # most frequent first
        self.assertEqual(len(result), 3)

    def test_empty_strings_filtered(self):
        result = summarize_texts(["", "  ", "real"])
        self.assertEqual(result, ["real"])

    def test_max_three_returned(self):
        result = summarize_texts(["a", "b", "c", "d", "e"])
        self.assertLessEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
