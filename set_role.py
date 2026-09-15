#!/usr/bin/env python3
"""Manage user roles for the Simulated PT Patient app.

    python3 set_role.py --list
    python3 set_role.py andre developer
    python3 set_role.py prof_smith teacher

Roles:
    student    chat + own sessions (default for new signups)
    teacher    student + will gain assignment/roster views
    developer  full access: system-context panel, Test runner

Run it from the app directory. Restart Flask afterwards is NOT required —
roles are read from disk on each request.
"""
import json
import sys
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"
ROLES = ("student", "teacher", "developer")


def load():
    if not USERS_FILE.exists():
        print(f"No {USERS_FILE.name} yet — register an account in the web app first.")
        sys.exit(1)
    raw = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    users = {}
    for name, val in raw.items():
        if isinstance(val, str):            # migrate old flat format
            users[name] = {"password": val, "role": "student", "created_at": ""}
        else:
            val.setdefault("role", "student")
            val.setdefault("created_at", "")
            users[name] = val
    return users


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    users = load()

    if args[0] in ("-l", "--list"):
        width = max((len(u) for u in users), default=4)
        print(f"{'USER'.ljust(width)}  ROLE")
        for name, rec in sorted(users.items()):
            print(f"{name.ljust(width)}  {rec['role']}")
        return

    if len(args) != 2:
        print("Usage: python3 set_role.py <username> <student|teacher|developer>")
        sys.exit(1)

    username, role = args[0].strip().lower(), args[1].strip().lower()

    if role not in ROLES:
        print(f"Unknown role '{role}'. Choose one of: {', '.join(ROLES)}")
        sys.exit(1)
    if username not in users:
        print(f"No such user '{username}'. Known users: {', '.join(sorted(users)) or '(none)'}")
        sys.exit(1)

    was = users[username]["role"]
    users[username]["role"] = role
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{username}: {was} -> {role}")


if __name__ == "__main__":
    main()
