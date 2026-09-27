"""Real asynchronous Jev rewards. No fallback rewards on API failure."""
import asyncio, math, os, random, time, fcntl
from contextlib import asynccontextmanager
from pathlib import Path
from email.utils import parsedate_to_datetime
from collections import defaultdict
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
from src.credentials import load_credentials
from src.common import append_jsonl, fingerprint
from src.rewards.cache import RewardCache
from src.rewards.reference_reward import completion_text
from src.rewards.rubrics import RUBRIC_VERSION, final_question, semantic_questions

def normalized(value, maximum=1.):
    value=float(value)/maximum
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid reward received from Jev")
    return value

def parse_response(response, semantic=False):
    raw=response.model_dump(mode="json")
    if semantic:
        dims={k:normalized(response.scores[k].score,4) for k in ("helpfulness","correctness","coherence")}
        dims["instruction_following"]=normalized(response.nouls["instruction_following"].noul)
        dims["composite"]=sum(dims[k]*w for k,w in zip(dims,(.35,.35,.15,.15)))
        return dict(reward=dims["composite"], dimensions=dims, raw=raw)
    return dict(reward=normalized(response.nouls["correct"].noul),raw=raw)

def retry_delay(error,attempt):
    headers=getattr(error,"headers",{})
    for field,scale in (("retry-after-ms",.001),("retry-after",1)):
        raw=headers.get(field)
        if raw is not None:
            try: delay=float(raw)*scale
            except ValueError:
                try: delay=parsedate_to_datetime(raw).timestamp()-time.time()
                except (ValueError,TypeError): continue
            if math.isfinite(delay): return max(0,delay)
    return min(30, .5*2**attempt)+random.random()*.25

@asynccontextmanager
async def request_slot():
    limit=int(os.environ.get("JEV_MAX_CONCURRENCY","8"))
    root=Path("cache/request_slots");root.mkdir(parents=True,exist_ok=True)
    held=None
    while held is None:
        for i in range(limit):
            f=(root/str(i)).open("a")
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: f.close();continue
            held=f;break
        if held is None: await asyncio.sleep(.05)
    try: yield
    finally:
        fcntl.flock(held,fcntl.LOCK_UN);held.close()

class JevReward:
    def __init__(self, cache=None, run_id=None, mode="strict", client=None):
        self.model=os.environ.get("JEV_MODEL","jev-latest")
        self.run_id=run_id or os.environ.get("JEV_RUN_ID","pilot")
        self.maximum=int(os.environ.get("JEV_MAX_REQUESTS","10000"))
        self.concurrency=max(1,int(os.environ.get("JEV_MAX_CONCURRENCY","8"))//int(os.environ.get("WORLD_SIZE","1")))
        self.timeout=float(os.environ.get("JEV_REQUEST_TIMEOUT","60"))
        self.retries=int(os.environ.get("JEV_MAX_RETRIES","5"))
        self.cache=cache or RewardCache()
        self.mode=mode
        self.client=client
        self.hits=0
        self.misses=0
        if mode not in ("strict","grouped"): raise ValueError(mode)
        if self.concurrency<1 or self.timeout<=0 or self.retries<0: raise ValueError("Invalid Jev limits")

    async def _request(self,client,state,questions):
        retry_started=time.perf_counter()
        retry_budget=self.timeout*(self.retries+1)
        for attempt in range(self.retries+1):
            request_id=self.cache.reserve(self.run_id,self.maximum)
            started=time.perf_counter()
            try:
                async with request_slot():
                    response=await client.system_one(state=state, questions=questions, model=self.model)
                latency=time.perf_counter()-started
                self.cache.finish(request_id,"ok",latency,response.usage.model_dump(),response.model)
                return response,latency
            except Exception as exc:
                latency=time.perf_counter()-started
                status=getattr(exc,"status",None)
                # Do not log bodies, headers or exception messages; may contain sensitive server echoes.
                safe=dict(type=type(exc).__name__,status=status,attempt=attempt,request_id=request_id,run=self.run_id)
                self.cache.finish(request_id,"failed",latency,error=safe["type"])
                append_jsonl("outputs/jev_errors.jsonl",safe)
                retryable=status in (408,429) or (status is not None and 500<=status<600) or isinstance(exc,(ConnectionError,TimeoutError))
                if not retryable or attempt==self.retries:
                    raise RuntimeError(f"Jev request failed: {type(exc).__name__}, status={status}") from None
                delay=retry_delay(exc,attempt)
                if time.perf_counter()-retry_started+delay>=retry_budget:
                    raise RuntimeError("Jev retry time budget exhausted") from None
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _with_client(self, operation):
        if self.client is not None:
            return await operation(self.client)
        load_credentials()
        async with AsyncTypeSafeClient(model=self.model,timeout=self.timeout,
                                       retry=RetryPolicy(max_retries=0)) as client:
            return await operation(client)

    async def semantic(self, rows):
        sem=asyncio.Semaphore(self.concurrency)
        locks=defaultdict(asyncio.Lock)
        async def run(client):
            async def one(row):
                key=self.cache.key(self.model,row["prompt"],None,row["response"],RUBRIC_VERSION+":semantic")
                async with locks[key]:
                    found=self.cache.get(key)
                    if found is not None: self.hits+=1; return found
                    self.misses+=1
                    async with sem:
                        resp,lat=await self._request(client,{"user_prompt":row["prompt"],"assistant_response":row["response"]},semantic_questions())
                    out=parse_response(resp,True)
                    out.update(latency=lat,timestamp=time.time(),rubric_version=RUBRIC_VERSION)
                    self.cache.put(key,out)
                    return out
            return await asyncio.gather(*(one(row) for row in rows))
        return await self._with_client(run)

    async def score(self, problems, completions, references):
        if not len(problems)==len(completions)==len(references): raise ValueError("Mismatched Jev inputs")
        texts=[completion_text(c) for c in completions]
        sem=asyncio.Semaphore(self.concurrency)
        locks=defaultdict(asyncio.Lock)
        async def run(client):
            if self.mode=="strict":
                async def one(p,c,r):
                    key=self.cache.key(self.model,p,r,c,RUBRIC_VERSION+":strict")
                    async with locks[key]:
                        found=self.cache.get(key)
                        if found is not None: self.hits+=1; return found["reward"]
                        self.misses+=1
                        async with sem:
                            resp,lat=await self._request(client,dict(problem=p,reference_final_answer=r,candidate_response=c),{"correct":final_question()})
                        out=parse_response(resp)
                        out.update(latency=lat,timestamp=time.time(),rubric_version=RUBRIC_VERSION)
                        self.cache.put(key,out)
                        return out["reward"]
                return await asyncio.gather(*(one(p,c,r) for p,c,r in zip(problems,texts,references)))
            groups=defaultdict(list)
            for i,(p,r) in enumerate(zip(problems,references)): groups[(p,r)].append(i)
            result=[None]*len(texts)
            async def group(p,r,ids):
                # Context includes every candidate: grouped judgments are not strict-cache compatible.
                context=[texts[i] for i in ids]
                key=self.cache.key(self.model,p,r,context,RUBRIC_VERSION+":grouped",context)
                async with locks[key]:
                    cached=self.cache.get(key)
                    if cached is None:
                        self.misses+=1
                        state=dict(problem=p,reference_final_answer=r,candidates=context)
                        qs={f"c{j}":final_question(f"candidates[{j}]") for j in range(len(ids))}
                        async with sem: resp,lat=await self._request(client,state,qs)
                        cached=dict(rewards=[normalized(resp.nouls[f"c{j}"].noul) for j in range(len(ids))],
                                    raw=resp.model_dump(mode="json"),latency=lat,
                                    timestamp=time.time(),rubric_version=RUBRIC_VERSION)
                        self.cache.put(key,cached)
                    else: self.hits+=1
                    for i,v in zip(ids,cached["rewards"]): result[i]=v
            await asyncio.gather(*(group(p,r,ids) for (p,r),ids in groups.items()))
            return result
        return await self._with_client(run)

async def jev_reward(prompts, completions, **kwargs) -> list[float]:
    problems=kwargs.get("problem") or ["\n".join(m["content"] for m in p) if isinstance(p,list) else p for p in prompts]
    if "reference" not in kwargs: raise ValueError("Reference answers are required")
    engine=JevReward(mode=os.environ.get("JEV_REWARD_MODE","strict"))
    try: return await engine.score(problems,completions,kwargs["reference"])
    finally: engine.cache.close()
