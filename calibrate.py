"""Score the 8 past hires and compare with the rubric's calibration table (P2-P6).

Run once after adding your API key to check the scorer agrees with how the team
rated these people. Note: the past-hire CVs are also in the prompt as anchors,
so this checks consistency, not accuracy on new applicants.
    python calibrate.py
"""
from screener import PAST_HIRES, REF, extract_text, score_cv

rows = []
for f in sorted((REF / "past_hires").glob("*.docx")):
    key = next(k for k in PAST_HIRES if k in f.stem)
    name, rating, expected = PAST_HIRES[key]
    res = score_cv(extract_text(f), "PM")
    got = [p["score"] for p in res["params"][1:]]
    weights = [p["weight"] for p in res["params"][1:]]
    exp_w, got_w = sum(a * w for a, w in zip(expected, weights)), sum(a * w for a, w in zip(got, weights))
    diff = sum(abs(a - b) for a, b in zip(expected, got))
    rows.append((name, rating, expected, got, exp_w, got_w, diff))
    print(f"{name:9} {rating:8} expected {expected} /21={exp_w:2}  got {got} /21={got_w:2}  off by {diff}")

exact = sum(a == b for r in rows for a, b in zip(r[2], r[3]))
print(f"\nExact parameter matches: {exact}/{len(rows) * 5}")
top = [r[0] for r in sorted(rows, key=lambda r: -r[5])[:5]]
print("Top 5 by scorer:", top, "(should be the 5 'Exceeds' hires)")
