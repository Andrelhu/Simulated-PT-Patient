#!/usr/bin/env python3
"""Check the config files before a trial run.

    python3 check_config.py

Validates rubric.json, trial_config.json and character_meta.json against what
app2.py actually requires, and warns about the things that break quietly:
a malformed rubric kills the whole assessment, a missing avatar shows a grey
silhouette, a tip with an internal comma makes the advice unreadable.

Exits 0 if everything is usable, 1 if something would break.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
problems, warnings = [], []


def load(name, required=True):
    path = HERE / name
    if not path.exists():
        (problems if required else warnings).append(f"{name}: not found")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        problems.append(f"{name}: invalid JSON — {e}")
        return None


def check_rubric():
    r = load("rubric.json")
    if r is None:
        problems.append("rubric.json is unusable, so NO encounter will be graded at all")
        return
    print("rubric.json")

    items = r.get("items", [])
    if len(items) != 10:
        warnings.append(f"rubric.json: {len(items)} items (the faculty list has 10)")

    seen = set()
    for it in items:
        tag = it.get("id", "?")
        for key in ("id", "n", "question", "type", "applies"):
            if key not in it:
                problems.append(f"rubric item {tag}: missing '{key}'")
        if it.get("applies") not in ("always", "if_opportunity"):
            problems.append(f"rubric item {tag}: applies must be 'always' or "
                            f"'if_opportunity', got {it.get('applies')!r}")
        if it.get("applies") == "if_opportunity" and not it.get("opportunity"):
            problems.append(f"rubric item {tag}: if_opportunity needs an 'opportunity' "
                            "describing when it applies, or it can never be graded")
        if it.get("type") not in ("content", "judgment"):
            warnings.append(f"rubric item {tag}: unknown type {it.get('type')!r}")
        if not it.get("tip"):
            warnings.append(f"rubric item {tag}: no tip, so missing it gives no advice")
        if "," in it.get("tip", ""):
            warnings.append(f"rubric item {tag}: tip has an internal comma — joining "
                            "three tips becomes hard to read")
        low = " " + it.get("tip", "").lower() + " "
        if any(f" {w} " in low for w in ("her", "she", "his", "he", "hers")):
            warnings.append(f"rubric item {tag}: tip uses a gendered pronoun; the rubric "
                            "is shared across all characters")
        if tag in seen:
            problems.append(f"rubric: duplicate item id {tag!r}")
        seen.add(tag)

    th = r.get("thresholds", {})
    g, a = th.get("green"), th.get("amber")
    if not (isinstance(g, (int, float)) and isinstance(a, (int, float))):
        problems.append("rubric thresholds: green and amber must be numbers")
    elif not (0 < a < g <= 1):
        problems.append(f"rubric thresholds: need 0 < amber < green <= 1, got "
                        f"amber={a} green={g}. These are fractions, not percentages.")

    if set(r.get("credit", {})) != {"got", "partial", "missed"}:
        problems.append("rubric credit: needs exactly got, partial and missed")

    mt = r.get("tips", {}).get("max_tips")
    if mt is not None and not (isinstance(mt, int) and mt > 0):
        problems.append(f"rubric tips.max_tips must be a positive integer, got {mt!r}")

    print(f"   {len(items)} items, "
          f"{sum(1 for i in items if i.get('applies') == 'if_opportunity')} conditional, "
          f"green at {g}, amber at {a}")


def check_trial():
    cfg = load("trial_config.json")
    if cfg is None:
        warnings.append("trial_config.json missing — defaults apply and / serves the trial")
        return
    print("\ntrial_config.json")

    dur = cfg.get("duration_seconds")
    if not (isinstance(dur, int) and dur > 0):
        problems.append(f"trial duration_seconds must be a positive integer, got {dur!r}")
        dur = 0
    elif dur < 120:
        warnings.append(f"trial duration is only {dur}s — fine for testing, raise it "
                        "to 600 before students use it")

    warn = cfg.get("warn_at_seconds", 0)
    if isinstance(warn, int) and isinstance(dur, int) and warn >= dur > 0:
        problems.append(f"warn_at_seconds ({warn}) is not less than duration ({dur}), "
                        "so the warning would show immediately")

    code = str(cfg.get("class_code", ""))
    if not code:
        warnings.append("class_code is empty — anyone with the URL can start an encounter "
                        "and consume CARC API calls")
    elif code == "PT2026":
        warnings.append("class_code is still the placeholder PT2026 — change it before "
                        "handing the link out")

    char = cfg.get("character", "")
    if not (HERE / f"character_{char}.txt").exists():
        problems.append(f"trial character is {char!r} but character_{char}.txt does not "
                        "exist, so the encounter cannot start")

    print(f"   {'enabled' if cfg.get('enabled') else 'DISABLED'}, "
          f"{char}, {dur}s, warn at {warn}s, "
          f"code {'set' if code else 'OPEN'}")


def check_characters():
    meta = load("character_meta.json", required=False) or {}
    files = sorted(p.stem.replace("character_", "", 1)
                   for p in HERE.glob("character_*.txt"))
    print("\ncharacters")
    if not files:
        problems.append("no character_*.txt files found")
        return

    avatars = HERE / "avatars"
    for name in files:
        m = meta.get(name, {})
        img = next((e for e in ("png", "jpg", "jpeg", "webp")
                    if (avatars / f"{name}.{e}").exists()), None)
        gender = m.get("gender", "female (default)")
        voice = m.get("voice", "default for gender")
        print(f"   {name:10} avatar={img or 'generated silhouette':20} "
              f"gender={gender:16} voice={voice}")
        if name not in meta:
            warnings.append(f"{name}: not in character_meta.json, so it gets the default "
                            "female voice regardless of the character")
        elif m.get("gender") not in ("male", "female"):
            warnings.append(f"{name}: gender {m.get('gender')!r} is not male or female, "
                            "so voice selection falls back to the default")
        if not img:
            warnings.append(f"{name}: no avatar image in avatars/, students see a grey "
                            f"silhouette. Add avatars/{name}.jpg")


def main():
    check_rubric()
    check_trial()
    check_characters()

    print()
    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"BREAK {p}")

    if problems:
        print(f"\n{len(problems)} thing(s) would break. Fix those before running the trial.")
        return 1
    if warnings:
        print(f"\nUsable, with {len(warnings)} warning(s) above.")
        return 0
    print("\nAll config valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
