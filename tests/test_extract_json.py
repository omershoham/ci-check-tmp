from src.deep_research import _extract_json


def test_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_stray_brace_in_prose_before_json():
    text = "Based on {today}'s data the answer is:\n{\"rating\": \"buy\", \"conviction\": 6}"
    assert _extract_json(text) == {"rating": "buy", "conviction": 6}


def test_prefers_richest_then_last():
    text = 'example {"x": 1} ... final {"x": 1, "y": 2} trailing {"z": 3}'
    assert _extract_json(text) == {"x": 1, "y": 2}


def test_nested_and_strings_with_braces():
    text = 'noise {"a": {"b": [1, 2]}, "s": "brace } inside \\" quote"} tail'
    out = _extract_json(text)
    assert out["a"] == {"b": [1, 2]}
    assert "}" in out["s"]


def test_no_object_returns_none():
    assert _extract_json("no json here") is None
    assert _extract_json("{unbalanced") is None
    assert _extract_json("[1, 2, 3]") is None
