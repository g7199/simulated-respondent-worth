"""Claude-Sonnet-5.5 sometimes wrote the ids of a one-question group call as q or q + g instead of 100*q + g (for q >= 100).
The 21 entries were complete and in the required order (everyone first, then groups 1 to 20), so the ids are mapped by position;
the numbers are not changed, and the file records the repair. The failed attempt is the tmp_ file that sol_common.run leaves behind.
  python scripts/collect/repair_ids.py outputs/sol/egjoint_claude-sonnet-5-5            (W_SET=fresh for the held-out reports)"""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO + "/src")
from w_groups import load
from w_util import read
D = sys.argv[1]; T = load(); N = len(T); P = read(D)
for q in [q for q in range(N) if not all(100 * q + g in P for g in range(21))]:
    f = f"{D}/tmp_s{q:03d}.json"
    if not os.path.exists(f):
        continue
    items = json.load(open(f))["items"]; ids = [x["id"] for x in items]; K = len(T[q]["options"])
    if len(items) == 21 and ids in ([q] + [100 * q + g for g in range(1, 21)], [q + g for g in range(21)]) and all(len(x["p"]) == K and min(x["p"]) >= 0 and sum(x["p"]) > 0 for x in items):
        json.dump({"items": {str(100 * q + g): items[g]["p"] for g in range(21)}, "usage": {}, "seconds": None, "effort": "medium", "model": os.path.basename(D).split("_")[-1],
                   "repaired": "ids written as q or q+g were mapped by position to 100*q+g"}, open(f"{D}/s{q:03d}.json", "w"))
        os.remove(f); print("repaired", q)
