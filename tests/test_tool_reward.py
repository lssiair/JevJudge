import json
import pytest
from src.rewards.tool_reward import tool_scores, parse_calls
def response(calls):
    return "<think>Reason.</think>\n<tool_call>\n" + "\n".join(json.dumps(c) for c in calls) + "\n</tool_call>"
A={"name":"a","parameters":{"x":1}}
B={"name":"b","parameters":{"y":"ok"}}

def test_perfect_and_reordered():
    gt=response([A,B])
    assert tool_scores(response([B,A]),gt)["reward"] == 4
    assert tool_scores(response([B,A]),gt)["exact_match"] == 1

def test_incorrect_parameter_partial_credit():
    gt=response([A]);bad=response([{"name":"a","parameters":{"x":2}}])
    score=tool_scores(bad,gt)
    assert score["correctness_reward"] == 1
    assert score["exact_match"] == 0

def test_wrong_tool():
    assert tool_scores(response([B]),response([A]))["correctness_reward"] == -3

def test_no_reuse_and_optimal_matching():
    a2={"name":"a","parameters":{"x":2}}
    assert tool_scores(response([a2,A]),response([A,a2]))["correctness_reward"] == 3
    assert tool_scores(response([A]),response([A,A]))["correctness_reward"] < 3

def test_malformed_and_truncated():
    for s in ("<tool_call>{broken}</tool_call>","<tool_call>"):
        assert tool_scores(s,response([A]))["correctness_reward"] == -3

def test_no_call_reference():
    ref="<think>Need more information.</think>\n<response>Please clarify.</response>"
    assert tool_scores(ref,ref)["reward"] == 1
    assert tool_scores(response([A]),ref)["reward"] == 0

def test_format_separate_from_correctness():
    s=response([A]).replace("<think>Reason.</think>","")
    result=tool_scores(s,response([A]))
    assert result["format_reward"] == 0
    assert result["correctness_reward"] == 3

def test_invalid_reference_raises():
    with pytest.raises(ValueError):
        tool_scores("", "<tool_call>bad</tool_call>")
