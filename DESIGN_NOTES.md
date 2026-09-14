# Design notes

Working notes for the Simulated PT Patient app. Covers decisions already made
and open questions that need a call before the next round of code.

---

## 1. Roles (implemented)

Three roles, stored in `users.json`:

| Role | Sees | Notes |
|---|---|---|
| `student` | Chat, character name, avatar, scrap notes, own sessions | Default for new signups |
| `teacher` | Same as student today | Assignment/roster views not built yet — see §2 |
| `developer` | Everything: system-context panel, Test runner | First account ever registered gets this automatically |

Change a role with the CLI helper:

```bash
python3 set_role.py --list
python3 set_role.py andre developer
python3 set_role.py prof_smith teacher
```

Roles are re-read from disk on every request, so no restart is needed.

### Why the prompt moved server-side

The system context used to be held in browser JavaScript and sent with every
request. That meant anyone could open devtools and rewrite the patient's
personality — which would quietly corrupt the research data.

Now `/chat` resolves the context from `character_<Name>.txt` on the server using
only the character *name* sent by the client. Developers can still send an
`override_context`, and the server honours it *only* if that account's role is
`developer`. Students cannot alter the prompt at all.

---

## 2. Teacher assignment + review — proposed, NOT built

This is the largest remaining piece. Sketch below; the enrollment question at
the end needs your decision before it gets built.

### Data model

New `assignments.json`:

```json
{
  "asg_001": {
    "title": "Week 3 — Subjective history, acute ankle",
    "character": "Ana",
    "instructions": "Complete a full subjective examination. Identify the mechanism of injury and at least two functional limitations.",
    "created_by": "prof_smith",
    "created_at": "2026-09-14T14:00:00Z",
    "due_at": "2026-10-01T23:59:00Z",
    "min_exchanges": 10,
    "open": true
  }
}
```

Session files gain two fields: `assignment_id` and `submitted_at`.

### Student flow

1. Landing page lists open assignments (title, character, due date, status).
2. "Start" opens the chat with that character locked and `assignment_id` attached
   to the session.
3. A progress indicator shows exchanges completed against `min_exchanges`.
4. "Submit" stamps `submitted_at`. Sessions stay editable until submitted.

### Teacher flow

- `/teacher` — list of assignments they created, with a completion count.
- `/teacher/<asg_id>` — roster table: one row per student, showing status
  (not started / in progress / submitted), exchange count, submission time,
  and per-row **View** and **Excel** links.
- **Download all** — one workbook, one sheet per student, for bulk grading or
  analysis in pandas.

### Open question — how do teachers know which students are theirs?

| Option | How it works | Trade-off |
|---|---|---|
| **A. No enrollment** | Teacher sees everyone who completed the assignment | Simplest. Fine for one instructor piloting. Breaks down with multiple sections. |
| **B. Class codes** | Teacher creates a class with a join code; student enters it at registration | Small amount of extra UI. Scales to multiple sections and instructors. |
| **C. Roster upload** | Teacher uploads a CSV of usernames | Most control, most admin work, and needs usernames to exist first. |

**Recommendation:** ship **A** for the faculty pilot, design the data model so
**B** can be added without migrating anything (i.e. put an optional `class_id`
on the user record from day one).

---

## 3. Text to speech — resource report

Browser-based TTS is live today (the **Voice** button). Voice gender is chosen
per character from `character_meta.json`.

### What we use now: Web Speech API

| | |
|---|---|
| **Cost** | Free |
| **Server load** | Zero — synthesis happens on the student's machine |
| **Setup** | None. Already working. |
| **Offline** | Yes for OS voices; Google's higher-quality voices need network |

**The catch:** voice quality and availability depend entirely on the student's
computer. Windows SAPI voices (David, Zira, Mark) sound noticeably robotic.
macOS and iOS voices are considerably better. Chrome on desktop additionally
exposes Google's network voices, which are better again but are not available
offline and are not present in every browser.

The real pedagogical problem is **inconsistency** — "Ana" sounds like a
different person on every student's laptop, and some students get a markedly
worse experience than others through no choice of their own. For a research
study where voice may influence how students respond, that is an uncontrolled
variable.

### Server-side options, cheapest first

**Piper** (`rhasspy/piper`) — *recommended upgrade*
- ONNX neural TTS built for Raspberry Pi-class hardware; CPU-only
- Generates considerably faster than real time on a modest CPU
- Voice models roughly 20–110 MB depending on quality tier
- Many English voices, both genders, several accents
- MIT licensed, fully offline — **no student data leaves the VM**, which keeps
  our current informed-consent wording accurate
- Quality: clearly better than Windows SAPI, a step below top-tier cloud
- *Cons:* one more moving part to install and keep running; voices are good but
  not indistinguishable from human

**Kokoro** (82M parameters)
- Apache-2.0, small model, quality noticeably above Piper
- CPU inference is workable but slower — expect a few seconds per utterance
- *Cons:* fewer voices to choose from; more CPU per reply than Piper

**Coqui XTTS-v2**
- Very high quality, supports voice cloning from a short sample
- *Cons:* realistically wants a GPU, ~2 GB model, slow on CPU. Coqui the company
  wound down, so this is a maintenance risk. Overkill for our needs.

**espeak-ng**
- Tiny and instant
- *Cons:* markedly robotic. Not an improvement over what students already have.

**Cloud APIs** (Azure Speech, ElevenLabs, OpenAI, Google)
- Best quality available, and identical for every student
- Azure Neural TTS is the natural institutional fit if HU already runs Azure
- *Cons, and they are real:* per-character cost that scales with class size;
  requires dependable outbound HTTPS from the VM, which has already bitten us
  more than once; and it sends session content to a third party, which **would
  require updating the informed consent**, since our current wording implies
  data stays within HU.

### Recommendation

Keep browser TTS as the default — it costs nothing and already works. Add Piper
as an **opt-in** server-side voice when you want consistency across students.

The integration is small: a `/tts?character=Ana&text=...` endpoint returning
WAV audio, and a frontend that plays that when available and silently falls back
to the browser voice when it is not. Roughly 60 MB resident per loaded voice
model plus a brief CPU spike per reply — comfortable alongside Flask on the
current VM.

Worth doing **before** a formal study, and safe to skip for the faculty
feedback round.

---

## 4. Institutional (Microsoft) login

### The proper answer: Entra ID / Azure AD via OAuth2

Use the `msal` Python library. IT registers an application in the HU tenant and
gives us a client ID, a client secret, and a registered redirect URI
(`https://<our-domain>/auth/callback`).

**What it buys us**
- No password storage at all — a whole category of risk disappears
- Verified real identity (name + institutional email) attached to every session,
  which is a meaningful upgrade for the research data
- Students use the login they already have; nothing new to remember
- Automatic deprovisioning when someone leaves the university
- Group claims can auto-assign the `teacher` role from an existing HU group,
  removing manual role management

**What it costs us**
- An IT request with institutional lead time — this is usually the long pole,
  so **worth starting now even if we build it later**
- The VM needs reliable outbound HTTPS to `login.microsoftonline.com`. Given the
  network trouble we have already had, verify this early:
  ```bash
  curl -sS -o /dev/null -w "%{http_code}\n" https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration
  ```
- Local accounts still need to exist as a fallback for external collaborators

### Cheap interim step

Keep local accounts, but restrict registration to `@harrisburgu.edu` addresses
and confirm with an emailed verification code. That ties every account to a real
institutional identity without waiting on IT, and it is perhaps an hour of work.

### Recommendation

File the IT request for Entra ID now, because the lead time dominates. Ship the
email-domain restriction in the meantime so the faculty pilot is not blocked.

---

## 5. Avatars

`/avatar/<Name>` serves `avatars/<Name>.png|jpg|jpeg|webp` when such a file
exists, and otherwise generates a coloured silhouette whose hue is derived from
the character's name — so every character looks distinct even with no images
present.

To use real images, drop files into `avatars/` named after the character. No
code change needed. See `avatars/README.md` for sizing and prompt suggestions.

---

## 6. Data layout for analysis

One JSON file per session at `sessions/<username>/session_<uuid>.json`:

```json
{
  "session_id": "…",
  "username": "…",
  "character": "Ana",
  "started_at": "2026-09-14T…",
  "last_updated": "2026-09-14T…",
  "exchange_count": 12,
  "notes": "student's scrap notes",
  "exchanges": [{"question": "…", "response": "…"}]
}
```

Load the whole corpus into pandas:

```python
import json, pandas as pd
from pathlib import Path

rows = []
for p in Path("sessions").glob("*/session_*.json"):
    d = json.loads(p.read_text())
    for i, ex in enumerate(d["exchanges"], 1):
        rows.append({
            "username":  d["username"],
            "character": d["character"],
            "started_at": d["started_at"],
            "turn":      i,
            "question":  ex["question"],
            "response":  ex["response"],
            "notes":     d.get("notes", ""),
        })

df = pd.DataFrame(rows)
```

---

## Open decisions

1. **Enrollment model** for teacher assignments — A, B, or C in §2?
2. **Piper TTS** — add now, or after the faculty feedback round?
3. **Entra ID** — should the IT request go in now?
4. Should teachers be able to view sessions for characters they did **not**
   assign, or only assignment-linked sessions?
