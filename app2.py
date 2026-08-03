import json
import secrets
import requests
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, request, jsonify, render_template_string, session, redirect
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# --- Configuration ---
API_URL    = "https://carc.harrisburgu.edu/api/v1/projects/vm-for-r-projects/llm/chat/completions"
API_KEY    = open("/home/ubuntu/apikey.txt").read().strip()
MODEL_NAME = "gemma"

CHARS_DIR    = Path(__file__).parent
SESSIONS_DIR = CHARS_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
USERS_FILE   = CHARS_DIR / "users.json"


# ── User helpers ──────────────────────────────────────────────────────────────
def load_users():
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))

def save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Session helpers ───────────────────────────────────────────────────────────
def save_session(session_id, username, character, exchanges):
    user_dir = SESSIONS_DIR / username
    user_dir.mkdir(exist_ok=True)
    path = user_dir / f"session_{session_id}.json"
    now = datetime.now(timezone.utc).isoformat()
    started_at = now
    if path.exists():
        try:
            started_at = json.loads(path.read_text(encoding="utf-8")).get("started_at", now)
        except Exception:
            pass
    data = {
        "session_id":     session_id,
        "username":       username,
        "character":      character,
        "started_at":     started_at,
        "last_updated":   now,
        "exchange_count": len(exchanges),
        "exchanges":      [{"question": q, "response": r} for q, r in exchanges],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_user_sessions(username):
    user_dir = SESSIONS_DIR / username
    if not user_dir.exists():
        return []
    result = []
    for p in sorted(user_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "session_id":     d.get("session_id", p.stem.replace("session_", "")),
                "character":      d.get("character", "Unknown"),
                "started_at":     d.get("started_at", "")[:19].replace("T", " "),
                "exchange_count": d.get("exchange_count", 0),
            })
        except Exception:
            continue
    return result


def get_session_data(username, session_id):
    path = SESSIONS_DIR / username / f"session_{session_id}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["started_at"] = d.get("started_at", "")[:19].replace("T", " ")
        return d
    except Exception:
        return None


# ── Main chat UI ──────────────────────────────────────────────────────────────
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
  #header-actions { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
  #username-display { font-size: 0.75rem; opacity: 0.8; margin-right: 4px; white-space: nowrap; }
  .hdr-btn { background: #CBB778; border: none; color: white; border-radius: 8px;
             padding: 6px 14px; cursor: pointer; font-size: 0.85rem; white-space: nowrap;
             text-decoration: none; display: inline-block; }
  .hdr-btn:hover { background: #b5a265; }
  #test-btn.running { background: rgba(220,50,50,0.85); }
  #test-btn.running:hover { background: rgba(220,50,50,1); }
  #test-btn:disabled { opacity: 0.5; cursor: default; }
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
              border-radius: 8px; font-size: 0.82rem; font-family: monospace; resize: vertical; }
  .settings-actions { display: flex; gap: 8px; }
  .btn-sm { padding: 7px 16px; font-size: 0.85rem; border-radius: 8px; border: none; cursor: pointer; }
  .btn-primary   { background: #CBB778; color: white; }
  .btn-primary:hover { background: #b5a265; }
  .btn-secondary { background: #CBB778; color: white; }
  .btn-secondary:hover { background: #b5a265; }
  #status-msg { font-size: 0.8rem; color: #388e3c; align-self: center; }

  /* ── chat ── */
  #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .bubble { max-width: 72%; padding: 10px 14px; border-radius: 18px;
            line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap; }
  .user  { background: #1a73e8; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
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
  <div id="header-actions">
    <span id="username-display">{{ username }}</span>
    <button id="new-chat-btn" class="hdr-btn">&#43; New</button>
    <button id="test-btn" class="hdr-btn">&#9654; Test</button>
    <button id="toggle-settings" class="hdr-btn">&#9881; Settings</button>
    <a href="/sessions" class="hdr-btn">Sessions</a>
    <a href="/logout" class="hdr-btn">Logout</a>
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

  // ── New Chat ──
  document.getElementById('new-chat-btn').addEventListener('click', () => {
    history   = [];
    sessionId = crypto.randomUUID();
    chat.innerHTML = '';
    statusMsg.textContent = 'New chat started.';
    setTimeout(() => statusMsg.textContent = '', 2000);
    msgInput.focus();
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
    history   = [];
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
    testBtn.innerHTML = '&#9654; Test';
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


# ── Login page ────────────────────────────────────────────────────────────────
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
           border-radius: 8px; font-size: 1rem; cursor: pointer; margin-bottom: 12px; }
  button:hover { background: #b5a265; }
  .error { color: #c0392b; font-size: 0.83rem; margin-bottom: 10px; text-align: center; }
  .alt-link { text-align: center; font-size: 0.83rem; color: #555; }
  .alt-link a { color: #0C6157; text-decoration: none; font-weight: 500; }
  .alt-link a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="card">
  <img class="logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <h1>Simulated PT Patient</h1>
  <p>Sign in to continue.</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <input type="text" name="username" placeholder="Username" autofocus autocomplete="username">
    <input type="password" name="password" placeholder="Password" autocomplete="current-password">
    <button type="submit">Sign In</button>
  </form>
  <div class="alt-link">New user? <a href="/register">Create an account</a></div>
</div>
</body>
</html>"""


# ── Registration page ─────────────────────────────────────────────────────────
REGISTER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Register — Simulated PT Patient</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5;
         display: flex; align-items: center; justify-content: center;
         min-height: 100vh; padding: 24px; }
  .card { background: white; border-radius: 12px; padding: 36px;
          box-shadow: 0 2px 16px rgba(0,0,0,.12); width: 100%; max-width: 480px; }
  .logo { display: block; height: 48px; margin: 0 auto 16px; }
  h1 { text-align: center; font-size: 1.1rem; color: #0C6157; margin-bottom: 4px; }
  .subtitle { text-align: center; font-size: 0.82rem; color: #777; margin-bottom: 22px; }
  .consent-box { background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;
                 padding: 14px 16px; font-size: 0.8rem; color: #444; line-height: 1.6;
                 margin-bottom: 16px; max-height: 200px; overflow-y: auto; }
  .consent-box h3 { font-size: 0.85rem; color: #0C6157; margin-bottom: 8px; }
  .consent-box ol { padding-left: 18px; }
  .consent-box li { margin-bottom: 6px; }
  input[type=text], input[type=password] {
    width: 100%; padding: 10px 14px; border: 1px solid #ccc; border-radius: 8px;
    font-size: 0.95rem; margin-bottom: 12px; outline: none; }
  input:focus { border-color: #CBB778; }
  .checkbox-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 18px;
                  font-size: 0.82rem; color: #444; cursor: pointer; }
  .checkbox-row input { width: auto; margin: 0; margin-top: 2px; cursor: pointer; }
  button { width: 100%; padding: 11px; background: #CBB778; color: white; border: none;
           border-radius: 8px; font-size: 1rem; cursor: pointer; margin-bottom: 12px; }
  button:hover { background: #b5a265; }
  .error { color: #c0392b; font-size: 0.83rem; margin-bottom: 12px; text-align: center; }
  .alt-link { text-align: center; font-size: 0.83rem; color: #555; }
  .alt-link a { color: #0C6157; text-decoration: none; font-weight: 500; }
  .alt-link a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="card">
  <img class="logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <h1>Create Account</h1>
  <p class="subtitle">Simulated PT Patient — Harrisburg University</p>

  <div class="consent-box">
    <h3>Informed Consent</h3>
    <ol>
      <li><strong>Session recording.</strong> All chat interactions are automatically saved and may be reviewed by course instructors and the development team at Harrisburg University.</li>
      <li><strong>Intended use only.</strong> This AI simulation is designed exclusively for physical therapy (PT) patient roleplay exercises. Do not use it for medical advice, diagnosis, or any purpose outside the assigned educational activity.</li>
      <li><strong>Research and improvement.</strong> Your session data may be used &mdash; in anonymized or identifiable form &mdash; to evaluate system performance and improve the simulation tool as part of ongoing educational research.</li>
      <li><strong>Subject to change.</strong> The AI model, patient characters, interface, and all other system components may be updated or replaced at any time without prior notice.</li>
      <li><strong>Voluntary participation.</strong> Use of this tool is voluntary. You may stop at any time by logging out.</li>
    </ol>
  </div>

  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <input type="text" name="username" placeholder="Choose a username (min. 3 characters)" autofocus autocomplete="username">
    <input type="password" name="password" placeholder="Choose a password (min. 6 characters)" autocomplete="new-password">
    <input type="password" name="confirm" placeholder="Confirm password" autocomplete="new-password">
    <label class="checkbox-row">
      <input type="checkbox" name="consent" value="yes">
      I have read and understood the informed consent above, and I agree to participate.
    </label>
    <button type="submit">Create Account</button>
  </form>
  <div class="alt-link">Already have an account? <a href="/login">Sign in</a></div>
</div>
</body>
</html>"""


# ── Sessions list page ────────────────────────────────────────────────────────
SESSIONS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Sessions — Simulated PT Patient</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5; }
  header { background: #0C6157; color: white; padding: 12px 20px;
           display: flex; align-items: center; justify-content: space-between; }
  .logo { height: 40px; }
  header h1 { font-size: 1rem; font-weight: 600; }
  .hdr-btn { background: #CBB778; border: none; color: white; border-radius: 8px;
             padding: 6px 14px; cursor: pointer; font-size: 0.85rem;
             text-decoration: none; display: inline-block; }
  .hdr-btn:hover { background: #b5a265; }
  .container { max-width: 860px; margin: 32px auto; padding: 0 20px; }
  h2 { font-size: 1rem; color: #0C6157; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; background: white;
          border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  th { background: #0C6157; color: white; padding: 10px 16px; text-align: left; font-size: 0.83rem; }
  td { padding: 10px 16px; font-size: 0.9rem; border-bottom: 1px solid #eee; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafafa; }
  .view-link { color: #0C6157; text-decoration: none; font-weight: 500; }
  .view-link:hover { text-decoration: underline; }
  .empty { text-align: center; color: #888; padding: 48px 20px; font-size: 0.95rem;
           background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
</style>
</head>
<body>
<header>
  <img class="logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <h1>My Sessions</h1>
  <a href="/" class="hdr-btn">&#8592; Back to Chat</a>
</header>
<div class="container">
  <h2>Sessions for {{ username }}</h2>
  {% if sessions %}
  <table>
    <thead>
      <tr>
        <th>Date &amp; Time</th>
        <th>Character</th>
        <th>Exchanges</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {% for s in sessions %}
      <tr>
        <td>{{ s.started_at }}</td>
        <td>{{ s.character }}</td>
        <td>{{ s.exchange_count }}</td>
        <td><a class="view-link" href="/sessions/{{ s.session_id }}">View</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No sessions yet. Start a chat to create your first session.</div>
  {% endif %}
</div>
</body>
</html>"""


# ── Session detail page ───────────────────────────────────────────────────────
SESSION_DETAIL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session — Simulated PT Patient</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5; }
  header { background: #0C6157; color: white; padding: 12px 20px;
           display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .logo { height: 40px; flex-shrink: 0; }
  header h1 { font-size: 0.95rem; font-weight: 600; }
  .hdr-btn { background: #CBB778; border: none; color: white; border-radius: 8px;
             padding: 6px 14px; cursor: pointer; font-size: 0.85rem;
             text-decoration: none; display: inline-block; white-space: nowrap; }
  .hdr-btn:hover { background: #b5a265; }
  .container { max-width: 760px; margin: 32px auto; padding: 0 20px 40px; }
  .meta { color: #777; font-size: 0.82rem; margin-bottom: 24px; }
  .exchange { margin-bottom: 20px; }
  .q-wrap { text-align: right; margin-bottom: 6px; }
  .a-wrap { text-align: left; }
  .label { font-size: 0.72rem; color: #888; margin-bottom: 3px; }
  .q { background: #1a73e8; color: white; padding: 10px 14px;
       border-radius: 18px 18px 4px 18px; font-size: 0.9rem;
       display: inline-block; max-width: 80%; text-align: left; }
  .a { background: white; color: #111; padding: 10px 14px;
       border-radius: 18px 18px 18px 4px; font-size: 0.9rem;
       display: inline-block; max-width: 80%; text-align: left;
       box-shadow: 0 1px 2px rgba(0,0,0,.1); white-space: pre-wrap; }
</style>
</head>
<body>
<header>
  <img class="logo" src="https://www.arcgis.com/sharing/rest/content/items/088d68905927400bb34449dc1b387446/resources/images/widget_2/1709839675447.png" alt="logo">
  <h1>{{ data.character }} &mdash; {{ data.started_at }}</h1>
  <a href="/sessions" class="hdr-btn">&#8592; My Sessions</a>
</header>
<div class="container">
  <div class="meta">{{ data.exchange_count }} exchanges &nbsp;&middot;&nbsp; {{ data.username }}</div>
  {% for ex in data.exchanges %}
  <div class="exchange">
    <div class="q-wrap">
      <div class="label">You</div>
      <div class="q">{{ ex.question }}</div>
    </div>
    <div class="a-wrap">
      <div class="label">{{ data.character }}</div>
      <div class="a">{{ ex.response }}</div>
    </div>
  </div>
  {% endfor %}
</div>
</body>
</html>"""


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        consent  = request.form.get("consent", "")
        if not username or not password:
            error = "All fields are required."
        elif " " in username or len(username) < 3:
            error = "Username must be at least 3 characters with no spaces."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif not consent:
            error = "You must accept the informed consent to register."
        else:
            users = load_users()
            if username in users:
                error = "Username already taken. Choose another."
            else:
                users[username] = generate_password_hash(password)
                save_users(users)
                session["authenticated"] = True
                session["username"] = username
                return redirect("/")
    return render_template_string(REGISTER_HTML, error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        users = load_users()
        if username in users and check_password_hash(users[username], password):
            session["authenticated"] = True
            session["username"] = username
            return redirect("/")
        error = "Incorrect username or password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def index():
    return render_template_string(HTML, username=session.get("username", ""))


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
    username       = session.get("username", "anonymous")

    messages = []
    for past_user, past_bot in history:
        messages.append({"role": "user",      "content": past_user})
        messages.append({"role": "assistant", "content": past_bot})

    first_msg = (system_context + "\n\n" + user_message) if (not history and system_context) else user_message
    messages.append({"role": "user", "content": first_msg})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    MODEL_NAME,
        "messages": messages,
        "stream":   False,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        reply = f"Connection error: {e}"
    except (KeyError, IndexError):
        reply = f"Unexpected API response: {resp.text[:300]}"

    full_history = history + [[user_message, reply]]
    if session_id:
        save_session(session_id, username, character_name, full_history)

    return jsonify({"reply": reply})


@app.route("/test-questions")
@login_required
def test_questions():
    path = CHARS_DIR / "test_questions.txt"
    if not path.exists():
        return jsonify({"questions": []})
    questions = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return jsonify({"questions": questions})


@app.route("/sessions")
@login_required
def sessions_list():
    username = session.get("username", "")
    user_sessions = get_user_sessions(username)
    return render_template_string(SESSIONS_HTML, username=username, sessions=user_sessions)


@app.route("/sessions/<sid>")
@login_required
def session_detail(sid):
    username = session.get("username", "")
    data = get_session_data(username, sid)
    if data is None:
        return "Session not found.", 404
    return render_template_string(SESSION_DETAIL_HTML, data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
