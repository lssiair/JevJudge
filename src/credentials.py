"""Load credentials into this process only; never serialize secrets."""
import os, re
from pathlib import Path
def load_credentials():
    if os.environ.get("TYPESAFE_API_KEY"): return
    path=Path(os.environ.get("TYPESAFE_KEY_FILE","typesaft_api_key.txt"))
    if not path.is_file(): raise RuntimeError("Set TYPESAFE_API_KEY or TYPESAFE_KEY_FILE")
    value=path.read_text().strip()
    match=re.search(r"^(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.+)$",value,re.M)
    if match: value=match.group(1).strip()
    value=value.strip("'\"")
    if not value or any(c.isspace() for c in value):
        raise RuntimeError("Credential file must contain a single key or TYPESAFE_API_KEY assignment")
    os.environ["TYPESAFE_API_KEY"]=value
