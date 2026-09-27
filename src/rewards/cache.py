"""SQLite WAL cache and cross-process request budget."""
import hashlib, json, os, sqlite3, time
from pathlib import Path

class BudgetExceeded(RuntimeError): pass

class RewardCache:
    def __init__(self, path="cache/jev_rewards.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db=sqlite3.connect(path, timeout=60, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=60000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS rewards(key TEXT PRIMARY KEY, payload TEXT NOT NULL, timestamp REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, run TEXT NOT NULL, status TEXT NOT NULL,
          latency REAL, input_tokens INTEGER, output_tokens INTEGER, model TEXT, error TEXT, timestamp REAL NOT NULL);
        """)
        columns={r[1] for r in self.db.execute("PRAGMA table_info(requests)")}
        if "budget" not in columns:
            try:self.db.execute("ALTER TABLE requests ADD COLUMN budget TEXT")
            except sqlite3.OperationalError:
                if "budget" not in {r[1] for r in self.db.execute("PRAGMA table_info(requests)")}:raise
        self.db.execute("UPDATE requests SET budget=run WHERE budget IS NULL")
    @staticmethod
    def key(model, prompt, reference, completion, rubric, context=None):
        return hashlib.sha256(json.dumps([model,prompt,reference,completion,rubric,context],
                                         sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def get(self,key):
        r=self.db.execute("SELECT payload FROM rewards WHERE key=?",(key,)).fetchone()
        return json.loads(r[0]) if r else None
    def put(self,key,payload):
        self.db.execute("INSERT OR REPLACE INTO rewards VALUES(?,?,?)",
                        (key,json.dumps(payload,allow_nan=False),time.time()))
    def reserve(self,run,maximum):
        if maximum <= 0: raise BudgetExceeded("JEV_MAX_REQUESTS must be positive")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            scope=os.environ.get("JEV_BUDGET_ID",run)
            n=self.db.execute("SELECT count(*) FROM requests WHERE budget=?",(scope,)).fetchone()[0]
            if n>=maximum: raise BudgetExceeded(f"Jev request budget {maximum} exhausted for {run}")
            cur=self.db.execute("INSERT INTO requests(run,status,timestamp,budget) VALUES(?,?,?,?)",
                                (run,"started",time.time(),scope))
            self.db.execute("COMMIT")
            return cur.lastrowid
        except BaseException:
            self.db.execute("ROLLBACK"); raise
    def finish(self,id,status,latency,usage=None,model=None,error=None):
        usage=usage or {}
        self.db.execute("""UPDATE requests SET status=?,latency=?,input_tokens=?,output_tokens=?,model=?,error=?
                           WHERE id=?""",
                        (status,latency,usage.get("input_tokens"),usage.get("output_tokens"),model,error,id))
    def close(self): self.db.close()
