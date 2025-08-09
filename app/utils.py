import json, orjson, os, math, time, uuid, pathlib
from typing import Any, Dict, Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES_DIR = DATA / "images"
QA_PATH = DATA / "qa.jsonl"
INDEX_PATH = DATA / "img.index"
EMB_PATH = DATA / "img_embeds.npy"
IDS_PATH = DATA / "ids.json"
TAU_PATH = DATA / "tau.txt"

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    if not path.exists(): 
        return []
    with path.open("rb") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path: pathlib.Path, obj: Dict[str, Any]):
    with path.open("ab") as f:
        f.write(orjson.dumps(obj) + b"\n")

def write_json(path: pathlib.Path, obj: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: pathlib.Path, default=None):
    if not path.exists(): return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_text(path: pathlib.Path, s: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(s)

def read_text(path: pathlib.Path, default: str = "") -> str:
    if not path.exists(): return default
    return path.read_text(encoding="utf-8")

def gen_id() -> str:
    return uuid.uuid4().hex[:12]


