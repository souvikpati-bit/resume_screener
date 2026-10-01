# Resume Screener – PM & Sr PM hiring dashboard

A local web app. You upload resumes, and each one is scored against `reference/rubric.txt`. The 8 past-hire CVs are used as calibration examples. For each candidate the app:

- ranks them and places them in a band: SHORTLIST ≥17, BORDERLINE 12–16, NOT MOVING FORWARD ≤11
- writes an interview brief with probe questions tied to their weakest scores
- drafts an invite and a decline email
- lets you send the email with one click, after a second click to confirm

## Setup (once)

```
python -m pip install -r requirements.txt
copy .env.example .env      # then fill in ANTHROPIC_API_KEY, and the SMTP_* lines to enable sending
```

## Run

```
python app.py
```

Then open http://127.0.0.1:8765 and drag in resumes (PDF, DOCX or TXT).

Optional: `python calibrate.py` re-scores the 8 past hires and compares the results with the rubric's calibration table.

## How scoring works

- The model scores P1–P6 from 0 to 3, and must quote one line from the CV for each score. If there is no evidence, the score is 0.
- The weighted total (out of 24) and the band are calculated in code, not by the model.
- Every quote is checked against the CV text. If a quote doesn't appear in the CV, it gets a ⚠ flag so you can check it.
- Role: filenames starting `pm_` or `spm_` set the role. Otherwise the model picks PM or Sr PM from the CV. You can override the role and rescore.
- Re-uploading the same file is detected and not scored again.
- Data is stored in `data/` (`candidates.json` plus the uploaded files). To keep it in Supabase instead, run `supabase_setup.sql` once in the Supabase SQL Editor, then set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in `.env`. Candidates go to the `candidates` table and resume files to the private `resumes` bucket.

The server listens on localhost only and has no login. Don't expose it to a network.

## Hosting on Vercel

- Import this GitHub repo in Vercel (Framework preset: Other). `vercel.json` routes every URL to `api/index.py`.
- Add the same settings as `.env` under Project Settings → Environment Variables. `SUPABASE_URL`, `SUPABASE_SECRET_KEY` and `DASHBOARD_PASSWORD` are required there.
- On Vercel, each resume is scored during its upload (about 20–30 s), files up to 3 MB.
- `EMAIL_TEST_TO` keeps every email going to that address instead of candidates; remove it once a domain is verified in Resend.
