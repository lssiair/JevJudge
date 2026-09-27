from src.data.gsm8k import format_record,load_gsm8k
from src.data.helpsteer2 import load_helpsteer2
def test_format_no_reference_leak():
 r=format_record({"question":"What is 2+2?","answer":"2+2=4 #### 4"},0)
 assert r["reference"]=="4"
 assert "2+2=4" not in r["prompt"][0]["content"]
def test_local_datasets():
 ds=load_gsm8k(limit=10)
 assert len(ds)==10
 assert ds["example_id"]==load_gsm8k(limit=10)["example_id"]
 hs=load_helpsteer2(limit=2)
 assert len(set(x["prompt"] for x in hs))==2
