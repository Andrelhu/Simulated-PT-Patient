import glob
import json
import os
import secrets
import requests
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, request, jsonify, render_template_string, session, redirect
from pathlib import Path

# --- Configuration ---
API_URL    = "https://ood.harrisburgu.cloud/api/v1/chat/completions"
API_KEY    = open("/home/elhuillier/apikey.txt").read().strip()
MODEL_NAME = "gemma4:e4b"

CHARS_DIR    = Path(__file__).parent
SESSIONS_DIR = CHARS_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
PASSWORD     = open("/home/elhuillier/apppassword.txt").read().strip()

# --- HTML UI ---
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Simulated PT Patient</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* ── header ── */
  header { background: #0C6157; color: white; padding: 12px 20px;
           display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  #header-logo { height: 48px; width: auto; flex-shrink: 0; }
  #header-center { flex: 1; text-align: center; }
  header h1 { font-size: 1.05rem; font-weight: 600; }
  header p  { font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }
  #toggle-settings { background: #CBB778; border: none; color: white;
                     border-radius: 8px; padding: 6px 14px; cursor: pointer;
                     font-size: 0.85rem; white-space: nowrap; }
  #toggle-settings:hover { background: #b5a265; }
  #test-btn { background: #CBB778; border: none; color: white;
              border-radius: 8px; padding: 6px 14px; cursor: pointer;
              font-size: 0.85rem; white-space: nowrap; }
  #test-btn:hover:not(:disabled) { background: #b5a265; }
  #test-btn:disabled { opacity: 0.5; cursor: default; }
  #test-btn.running { background: rgba(220,50,50,0.85); }
  #test-btn.running:hover { background: rgba(220,50,50,1); }
  #test-progress { background: #e8f0fe; color: #1a73e8; font-size: 0.8rem;
                   padding: 6px 16px; text-align: center; flex-shrink: 0;
                   display: none; border-bottom: 1px solid #c5d4fb; }

  /* ── settings panel ── */
  #settings { background: #fff; border-bottom: 1px solid #ddd; padding: 14px 20px;
              display: none; flex-shrink: 0; gap: 12px; flex-direction: column; }
  #settings.open { display: flex; }
  .settings-row { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
  .settings-row label { font-size: 0.82rem; color: #555; font-weight: 500;
                        display: flex; flex-direction: column; gap: 4px; }
  #char-select { padding: 7px 10px; border: 1px solid #ccc; border-radius: 8px;
                 font-size: 0.9rem; min-width: 160px; }
  #ctx-area { width: 100%; height: 180px; padding: 8px 10px; border: 1px solid #ccc;
              border-radius: 8px; font-size: 0.82rem; font-family: monospace;
              resize: vertical; }
  .settings-actions { display: flex; gap: 8px; }
  .btn-sm { padding: 7px 16px; font-size: 0.85rem; border-radius: 8px; border: none;
            cursor: pointer; }
  .btn-primary { background: #CBB778; color: white; }
  .btn-primary:hover { background: #b5a265; }
  .btn-secondary { background: #CBB778; color: white; }
  .btn-secondary:hover { background: #b5a265; }
  #status-msg { font-size: 0.8rem; color: #388e3c; align-self: center; }

  /* ── chat ── */
  #chat { flex: 1; overflow-y: auto; padding: 16px;
          display: flex; flex-direction: column; gap: 10px; }
  .bubble { max-width: 72%; padding: 10px 14px; border-radius: 18px;
            line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap; }
  .user { background: #1a73e8; color: white; align-self: flex-end;
          border-bottom-right-radius: 4px; }
  .agent { background: white; color: #111; align-self: flex-start;
           border-bottom-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,.1); }
  .label { font-size: 0.72rem; color: #888; margin-bottom: 2px; }
  .user-wrap  { align-self: flex-end;  display: flex; flex-direction: column; align-items: flex-end; }
  .agent-wrap { align-self: flex-start; display: flex; flex-direction: column; }
  .typing { color: #888; font-style: italic; font-size: 0.9rem; padding: 4px 14px; }

  /* ── input bar ── */
  #input-bar { display: flex; gap: 8px; padding: 12px 16px;
               background: white; border-top: 1px solid #ddd; flex-shrink: 0; }
  #msg { flex: 1; padding: 10px 14px; border: 1px solid #ccc; border-radius: 24px;
         font-size: 0.95rem; outline: none; }
  #msg:focus { border-color: #CBB778; }
  #send { background: #CBB778; color: white; border: none; border-radius: 24px;
          padding: 10px 20px; cursor: pointer; font-size: 0.95rem; }
  #send:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>

<header>
  <img id="header-logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <div id="header-center">
    <h1 id="header-title">Simulated PT Patient</h1>
    <p>Conduct yourself as you would in a real clinical setting.</p>
  </div>
  <div style="display:flex;gap:8px">
    <button id="test-btn">&#9654; Test</button>
    <button id="toggle-settings">&#9881; Settings</button>
  </div>
</header>
<div id="test-progress"></div>

<div id="settings">
  <div class="settings-row">
    <label>
      Character
      <select id="char-select"></select>
    </label>
    <div style="display:flex;align-items:flex-end;gap:8px;padding-bottom:1px">
      <button class="btn-sm btn-secondary" id="load-char-btn">Load</button>
    </div>
  </div>
  <label>
    System context
    <textarea id="ctx-area" placeholder="Paste or edit the system context here…"></textarea>
  </label>
  <div class="settings-actions">
    <button class="btn-sm btn-primary" id="apply-btn">Apply &amp; reset chat</button>
    <span id="status-msg"></span>
  </div>
</div>

<div id="chat"></div>

<div id="input-bar">
  <input id="msg" type="text" placeholder="Type your question…" autocomplete="off">
  <button id="send">Send</button>
</div>

<script>
  const chat       = document.getElementById('chat');
  const msgInput   = document.getElementById('msg');
  const sendBtn    = document.getElementById('send');
  const ctxArea    = document.getElementById('ctx-area');
  const charSelect = document.getElementById('char-select');
  const statusMsg  = document.getElementById('status-msg');

  let history       = [];
  let systemContext = '';
  let sessionId     = crypto.randomUUID();

  // ── Settings panel toggle ──
  document.getElementById('toggle-settings').addEventListener('click', () => {
    document.getElementById('settings').classList.toggle('open');
  });

  // ── Load character list on page load ──
  async function loadCharacterList() {
    const res  = await fetch('/characters');
    const data = await res.json();
    charSelect.innerHTML = '';
    data.characters.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      charSelect.appendChild(opt);
    });
    if (data.characters.length > 0) {
      await loadCharacter(data.characters[0]);
    }
  }

  // ── Load a character's context into the textarea ──
  async function loadCharacter(name) {
    const res  = await fetch('/character/' + encodeURIComponent(name));
    const data = await res.json();
    ctxArea.value = data.context;
    document.getElementById('header-title').textContent = name + ' — Simulated PT Patient';
  }

  document.getElementById('load-char-btn').addEventListener('click', () => {
    loadCharacter(charSelect.value);
  });

  // ── Apply button: set context and reset chat ──
  document.getElementById('apply-btn').addEventListener('click', () => {
    systemContext = ctxArea.value.trim();
    history = [];
    sessionId = crypto.randomUUID();
    chat.innerHTML = '';
    document.getElementById('header-title').textContent =
      charSelect.value + ' — Simulated PT Patient';
    statusMsg.textContent = 'Context applied. Chat reset.';
    setTimeout(() => statusMsg.textContent = '', 2500);
    document.getElementById('settings').classList.remove('open');
    msgInput.focus();
  });

  // ── Chat ──
  function addBubble(role, text) {
    const wrap   = document.createElement('div');
    wrap.className = role === 'user' ? 'user-wrap' : 'agent-wrap';
    const label  = document.createElement('div');
    label.className = 'label';
    label.textContent = role === 'user' ? 'You' : charSelect.value || 'Agent';
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (role === 'user' ? 'user' : 'agent');
    bubble.textContent = text;
    wrap.appendChild(label);
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }

  async function sendMsg() {
    const text = msgInput.value.trim();
    if (!text || !systemContext) {
      if (!systemContext) {
        statusMsg.textContent = 'Load a character first, then click Apply.';
        document.getElementById('settings').classList.add('open');
      }
      return;
    }
    msgInput.value = '';
    sendBtn.disabled = true;
    addBubble('user', text);

    const typing = document.createElement('div');
    typing.className = 'typing';
    typing.textContent = (charSelect.value || 'Agent') + ' is typing…';
    chat.appendChild(typing);
    chat.scrollTop = chat.scrollHeight;

    try {
      const res  = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history, system_context: systemContext,
                                session_id: sessionId, character: charSelect.value })
      });
      const data = await res.json();
      typing.remove();
      addBubble('agent', data.reply);
      history.push([text, data.reply]);
    } catch(e) {
      typing.remove();
      addBubble('agent', 'Connection error — please try again.');
    }
    sendBtn.disabled = false;
    msgInput.focus();
  }

  sendBtn.addEventListener('click', sendMsg);
  msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMsg(); });

  // ── Test runner ──
  const testBtn      = document.getElementById('test-btn');
  const testProgress = document.getElementById('test-progress');
  let   stopRequested = false;

  async function runTest() {
    // If already running, act as Stop
    if (testBtn.classList.contains('running')) {
      stopRequested = true;
      return;
    }
    if (!systemContext) {
      statusMsg.textContent = 'Load a character first, then click Apply.';
      document.getElementById('settings').classList.add('open');
      return;
    }

    history = [];
    sessionId = crypto.randomUUID();
    chat.innerHTML = '';
    stopRequested = false;
    testBtn.textContent = '■ Stop';
    testBtn.classList.add('running');
    sendBtn.disabled = true;
    msgInput.disabled = true;
    testProgress.style.display = 'block';

    const res       = await fetch('/test-questions');
    const data      = await res.json();
    const questions = data.questions;

    for (let i = 0; i < questions.length; i++) {
      if (stopRequested) {
        testProgress.textContent = `Stopped at question ${i + 1}.`;
        break;
      }
      const q = questions[i];
      testProgress.textContent = `Test running — question ${i + 1} of ${questions.length}`;
      addBubble('user', q);

      const typing = document.createElement('div');
      typing.className = 'typing';
      typing.textContent = (charSelect.value || 'Agent') + ' is typing…';
      chat.appendChild(typing);
      chat.scrollTop = chat.scrollHeight;

      try {
        const r    = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: q, history, system_context: systemContext,
                                  session_id: sessionId, character: charSelect.value })
        });
        const resp = await r.json();
        typing.remove();
        addBubble('agent', resp.reply);
        history.push([q, resp.reply]);
      } catch(e) {
        typing.remove();
        addBubble('agent', 'Connection error — test aborted.');
        break;
      }
    }

    if (!stopRequested) testProgress.textContent = `Test complete — ${questions.length} questions answered.`;
    setTimeout(() => { testProgress.style.display = 'none'; }, 4000);
    testBtn.textContent = '&#9654; Test';
    testBtn.classList.remove('running');
    sendBtn.disabled = false;
    msgInput.disabled = false;
    msgInput.focus();
  }

  testBtn.addEventListener('click', runTest);

  // ── Init ──
  loadCharacterList();
</script>
</body>
</html>"""


LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login — Simulated PT Patient</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5;
         display: flex; align-items: center; justify-content: center; height: 100vh; }
  .card { background: white; border-radius: 12px; padding: 40px 36px;
          box-shadow: 0 2px 16px rgba(0,0,0,.12); width: 100%; max-width: 360px; }
  .logo { display: block; height: 56px; margin: 0 auto 20px; }
  h1 { text-align: center; font-size: 1.1rem; color: #0C6157; margin-bottom: 6px; }
  p  { text-align: center; font-size: 0.82rem; color: #777; margin-bottom: 24px; }
  input { width: 100%; padding: 10px 14px; border: 1px solid #ccc; border-radius: 8px;
          font-size: 0.95rem; margin-bottom: 14px; outline: none; }
  input:focus { border-color: #CBB778; }
  button { width: 100%; padding: 11px; background: #CBB778; color: white; border: none;
           border-radius: 8px; font-size: 1rem; cursor: pointer; }
  button:hover { background: #b5a265; }
  .error { color: #c0392b; font-size: 0.83rem; margin-bottom: 10px; text-align: center; }
</style>
</head>
<body>
<div class="card">
  <img class="logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <h1>Simulated PT Patient</h1>
  <p>Enter the access password to continue.</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <input type="password" name="password" placeholder="Password" autofocus>
    <button type="submit">Enter</button>
  </form>
</div>
</body>
</html>"""


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def save_session(session_id, character, exchanges):
    path = SESSIONS_DIR / f"session_{session_id}.json"
    data = {
        "session_id":  session_id,
        "character":   character,
        "saved_at":    datetime.now(timezone.utc).isoformat(),
        "exchanges":   [{"question": q, "response": r} for q, r in exchanges],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if request.form.get("password", "") == PASSWORD:
            session["authenticated"] = True
            return redirect("/")
        error = "Incorrect password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def index():
    return render_template_string(HTML)


@app.route("/characters")
@login_required
def characters():
    files = sorted(CHARS_DIR.glob("character_*.txt"))
    names = [f.stem.replace("character_", "", 1) for f in files]
    return jsonify({"characters": names})


@app.route("/character/<name>")
@login_required
def character(name):
    path = CHARS_DIR / f"character_{name}.txt"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify({"context": path.read_text(encoding="utf-8")})


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data           = request.get_json()
    user_message   = data["message"]
    history        = data.get("history", [])
    system_context = data.get("system_context", "")
    session_id     = data.get("session_id", "")
    character_name = data.get("character", "")

    messages = []
    for past_user, past_bot in history:
        messages.append({"role": "user",      "content": past_user})
        messages.append({"role": "assistant", "content": past_bot})

    first_msg = (system_context + "\n\n" + user_message) if (not history and system_context) else user_message
    messages.append({"role": "user", "content": first_msg})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        reply = f"Connection error: {e}"
    except (KeyError, IndexError):
        reply = f"Unexpected API response: {resp.text[:300]}"

    # Save session once it reaches 5 exchanges, then on every message after
    full_history = history + [[user_message, reply]]
    if session_id and len(full_history) >= 5:
        save_session(session_id, character_name, full_history)

    return jsonify({"reply": reply})


@app.route("/test-questions")
@login_required
def test_questions():
    path = CHARS_DIR / "test_questions.txt"
    if not path.exists():
        return jsonify({"questions": []})
    questions = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2601, debug=False)
