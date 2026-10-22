#!/usr/bin/env python3
"""Show what has been recorded so far.

    python3 sessions_report.py              every session, newest first
    python3 sessions_report.py --trial      trial participants only
    python3 sessions_report.py --csv out.csv    one row per message, for pandas
    python3 sessions_report.py --watch      refresh every 10s during a live trial

Reads sessions/<storage>/session_<id>.json. Nothing is modified.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SESSIONS = HERE / "sessions"


def load_all(trial_only=False):
    out = []
    for path in SESSIONS.glob("*/session_*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  (unreadable: {path})", file=sys.stderr)
            continue
        if trial_only and d.get("mode") != "trial":
            continue
        d["_path"] = path
        out.append(d)
    out.sort(key=lambda d: d.get("started_at", ""), reverse=True)
    return out


def pct_of(d):
    score = ((d.get("assessment") or {}).get("score") or {})
    f = score.get("fraction")
    return round(f * 100) if isinstance(f, (int, float)) else None


def table(rows):
    if not rows:
        print("No sessions recorded yet.")
        print(f"(looking in {SESSIONS})")
        return

    head = ("%-16s  %-20s  %-8s  %-7s  %4s  %-11s  %5s  %-8s" %
            ("STARTED", "STORAGE", "CODE", "PATIENT", "MSG", "CLOSED", "SCORE", "AFFECT"))
    print(head)
    print("-" * len(head))

    graded = ungraded = failed = open_count = 0
    for d in rows:
        a = d.get("assessment") or {}
        p = pct_of(d)
        if a.get("error") or (a and p is None):
            failed += 1
            score = "ERR"
        elif p is None:
            ungraded += 1
            score = "-"
        else:
            graded += 1
            score = f"{p}%"
        if not d.get("close_reason"):
            open_count += 1
        print("%-16s  %-20s  %-8s  %-7s  %4d  %-11s  %5s  %-8s" % (
            d.get("started_at", "")[:16].replace("T", " "),
            d.get("username", "")[:20],
            (d.get("participant_code") or "-")[:8],
            (d.get("character") or "")[:7],
            d.get("exchange_count", 0),
            d.get("close_reason") or "open",
            score,
            ((d.get("assessment") or {}).get("affect") or {}).get("overall", "-")[:8],
        ))

    print(f"\n{len(rows)} session(s): {graded} graded, {failed} grading failed, "
          f"{ungraded} with no assessment ({open_count} still open)")
    if graded:
        scores = [pct_of(d) for d in rows if pct_of(d) is not None]
        print(f"scores: mean {sum(scores)/len(scores):.0f}%  "
              f"min {min(scores)}%  max {max(scores)}%")
    if failed:
        print("\nGrading failed on some sessions. Check that the CARC API is reachable "
              "and run: python3 check_config.py")


def to_csv(rows, dest):
    """One row per message — the shape pandas wants."""
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "storage", "participant_code", "mode", "character",
                    "started_at", "closed_at", "close_reason", "duration_seconds",
                    "exchange_count", "score_pct", "affect", "affect_trajectory",
                    "mean_valence", "valence_shift", "tips", "notes",
                    "turn", "author", "message", "turn_valence", "turn_arousal", "turn_label"])
        for d in rows:
            a = d.get("assessment") or {}
            sent = (a.get("sentiment") or {})
            summary = sent.get("summary") or {}
            by_turn = {t.get("turn"): t for t in sent.get("turns", [])}
            base = [
                d.get("session_id", ""), d.get("username", ""),
                d.get("participant_code", ""), d.get("mode", ""), d.get("character", ""),
                d.get("started_at", ""), d.get("closed_at", ""), d.get("close_reason", ""),
                d.get("duration_seconds", ""), d.get("exchange_count", 0),
                pct_of(d) if pct_of(d) is not None else "",
                (a.get("affect") or {}).get("overall", ""),
                (a.get("affect") or {}).get("trajectory", ""),
                summary.get("mean_valence", ""), summary.get("valence_shift", ""),
                a.get("tips", ""), d.get("notes", ""),
            ]
            for i, ex in enumerate(d.get("exchanges", []), 1):
                t = by_turn.get(i, {})
                w.writerow(base + [i, "student", ex.get("question", ""), "", "", ""])
                w.writerow(base + [i, d.get("character", "patient"), ex.get("response", ""),
                                   t.get("valence", ""), t.get("arousal", ""),
                                   t.get("label", "")])
    n = sum(d.get("exchange_count", 0) for d in rows) * 2
    print(f"Wrote {dest} — {len(rows)} sessions, about {n} message rows.")
    print("\nIn pandas:")
    print(f"    import pandas as pd; df = pd.read_csv({dest!r})")


def item_breakdown(rows):
    """Which of the 10 items get missed most often across all graded sessions."""
    counts = {}
    for d in rows:
        for it in ((d.get("assessment") or {}).get("items") or []):
            key = (it.get("n"), it.get("id"))
            c = counts.setdefault(key, {"got": 0, "partial": 0, "missed": 0,
                                        "not_applicable": 0, "error": 0})
            if it.get("result") in c:
                c[it["result"]] += 1
    if not counts:
        return
    print("\nPer-item results across all graded sessions")
    print("%-3s %-23s %5s %8s %7s %6s" % ("#", "ITEM", "GOT", "PARTIAL", "MISSED", "N/A"))
    for (n, cid), c in sorted(counts.items(), key=lambda kv: kv[0][0] or 0):
        print("%-3s %-23s %5d %8d %7d %6d" %
              (n, cid, c["got"], c["partial"], c["missed"], c["not_applicable"]))


def main():
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0
    if not SESSIONS.exists():
        print(f"No sessions directory yet at {SESSIONS}")
        return 0

    trial_only = "--trial" in args
    rows = load_all(trial_only)

    if "--csv" in args:
        i = args.index("--csv")
        dest = args[i + 1] if len(args) > i + 1 else "sessions_export.csv"
        to_csv(rows, dest)
        return 0

    if "--watch" in args:
        try:
            while True:
                subprocess.run(["clear"], check=False)
                print(time.strftime("%H:%M:%S"), "— refreshing every 10s, Ctrl+C to stop\n")
                table(load_all(trial_only))
                time.sleep(10)
        except KeyboardInterrupt:
            return 0

    table(rows)
    item_breakdown(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
