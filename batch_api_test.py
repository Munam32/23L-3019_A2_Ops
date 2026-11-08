import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
url = os.environ["MODEL_API_URL"].rstrip("/")
headers = {"X-API-Key": os.environ["MODEL_API_KEY"]}
per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 4

test = pd.read_csv("splits/test.csv")
sample = pd.concat([test[test.label == 0].head(per_class), test[test.label == 1].head(per_class)])

rows = []
for _, r in sample.iterrows():
    started = time.perf_counter()
    resp = requests.post(f"{url}/predict", headers=headers, timeout=60,
                         files={"image": (Path(r["path"]).name, Path(r["path"]).read_bytes())})
    total_ms = int((time.perf_counter() - started) * 1000)
    body = resp.json() if resp.ok else {}
    rows.append({"file": Path(r["path"]).name[:32], "truth": "AI-generated" if r["label"] else "Real",
                 "http": resp.status_code, "verdict": body.get("label", "-"),
                 "p_ai": round(body.get("p_ai", float("nan")), 3), "server_ms": body.get("latency_ms", "-"),
                 "round_trip_ms": total_ms})
out = pd.DataFrame(rows)
out["correct"] = out["truth"] == out["verdict"]
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print(f"\n{out.correct.sum()}/{len(out)} correct at the champion threshold")
