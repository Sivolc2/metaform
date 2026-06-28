"""Unit tests for llm.extract_json — no Ollama required."""

import pytest
from metaform.llm import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_plain_array():
    assert extract_json('[1, 2, 3]') == [1, 2, 3]


def test_code_fenced():
    raw = '```json\n{"x": true}\n```'
    assert extract_json(raw) == {"x": True}


def test_reasoning_wrapped():
    raw = 'some reasoning text {"result": "ok"} trailing stuff'
    assert extract_json(raw) == {"result": "ok"}


def test_escaped_backslash_then_quote():
    # JSON: {"k": "val\\"end"} — backslash-backslash followed by a literal quote
    # The outer string ends at the *second* quote, not the one after the \\
    obj = {"k": 'val\\"end'}
    raw = '{"k": "val\\\\\\"end"}'
    assert extract_json(raw) == obj


def test_nested_braces_in_string():
    raw = '{"key": "a {b} c", "n": 1}'
    assert extract_json(raw) == {"key": "a {b} c", "n": 1}


def test_raises_on_no_json():
    with pytest.raises(ValueError, match="no JSON found"):
        extract_json("no json here at all")
