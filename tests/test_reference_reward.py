import pytest
from src.rewards.reference_reward import extract_answer,reference_reward,numeric

@pytest.mark.parametrize("text,expected",[
 ("#### 1,200",1200),("The answer is 4.",4),(r"\boxed{\frac{1}{2}}",numeric("1/2")),
 ("#### -2.5",numeric("-2.5")),("#### 50%",numeric(".5")),("#### 1e3",1000),
 ("<think>2 + 2 = 4</think>\n#### 4",4),
 ("<think>I think 4",None),("No answer.",None),("Could be 3 or 4",None),
 ("#### 2/0",None),("#### 3\n#### 4",4)])
def test_extract(text,expected):assert extract_answer(text)==expected

def test_reference():
 assert reference_reward([],["#### 4","#### 3","invalid"],["4","4","4"])==[1,0,0]
def test_no_substring_match():
 assert reference_reward([],["#### 14"],["4"])==[0]
def test_length_mismatch():
 with pytest.raises(ValueError):reference_reward([],["4"],[])
