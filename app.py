import glob
import os
import requests
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path

# --- Configuration ---
API_URL    = "https://ood.harrisburgu.cloud/api/v1/chat/completions"
API_KEY    = open("/home/elhuillier/apikey.txt").read().strip()
MODEL_NAME = "gemma4:e4b"

CHARS_DIR  = Path(__file__).parent  # character_*.txt files live next to app.py

# Appended server-side to every system context — no character file edits needed
FORMATTING_REMINDER = (
    "\n\n[RESPONSE FORMAT — always follow these rules:\n"
    "- Plain conversational text only\n"
    "- No em-dashes (—), no asterisks, no bold, no italics\n"
    "- No bullet points, no numbered lists, no markdown of any kind\n"
    "- Write exactly as a real person would speak in a chat message]"
)

# Injected silently every REMINDER_EVERY turns (shown as a chip in chat, full text goes to LLM)
REMINDER_EVERY = 3
HIDDEN_REMINDER = "[Reminder: plain conversational text only — no em-dashes, no asterisks, no bold, no bullets, no numbered lists.]"

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
  header { background: #1a73e8; color: white; padding: 12px 20px;
           display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  header h1 { font-size: 1.05rem; font-weight: 600; }
  header p  { font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }
  #toggle-settings { background: rgba(255,255,255,0.2); border: none; color: white;
                     border-radius: 8px; padding: 6px 14px; cursor: pointer;
                     font-size: 0.85rem; white-space: nowrap; }
  #toggle-settings:hover { background: rgba(255,255,255,0.35); }
  #test-btn { background: rgba(255,255,255,0.2); border: none; color: white;
              border-radius: 8px; padding: 6px 14px; cursor: pointer;
              font-size: 0.85rem; white-space: nowrap; }
  #test-btn:hover:not(:disabled) { background: rgba(255,255,255,0.35); }
  #test-btn:disabled { opacity: 0.5; cursor: default; }
  #test-btn.running { background: rgba(255,80,80,0.7); }
  #test-btn.running:hover { background: rgba(255,80,80,0.9); }
  .reminder-chip { font-size: 0.72rem; color: #aaa; font-style: italic;
                   padding: 1px 0 6px 4px; align-self: flex-end; }
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
  .btn-primary { background: #1a73e8; color: white; }
  .btn-primary:hover { background: #1558b0; }
  .btn-secondary { background: #e8eaed; color: #333; }
  .btn-secondary:hover { background: #d2d5db; }
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
  #msg:focus { border-color: #1a73e8; }
  #send { background: #1a73e8; color: white; border: none; border-radius: 24px;
          padding: 10px 20px; cursor: pointer; font-size: 0.95rem; }
  #send:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>

<header>
  <div>
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
      // Auto-apply first character so chat is ready immediately
      systemContext = ctxArea.value.trim();
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
        body: JSON.stringify({ message: text, history, system_context: systemContext })
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

  const REMINDER_EVERY = 3;
  const HIDDEN_REMINDER_TEXT = '[Reminder: plain conversational text only — no em-dashes, no asterisks, no bold, no bullets, no numbered lists.]';

  function addReminderChip() {
    const chip = document.createElement('div');
    chip.className = 'reminder-chip';
    chip.textContent = '📎 reminder sent';
    chat.appendChild(chip);
    chat.scrollTop = chat.scrollHeight;
  }

  function testSetRunning(running) {
    stopRequested = !running;
    testBtn.textContent  = running ? '■ Stop' : '▶ Test';
    testBtn.classList.toggle('running', running);
    sendBtn.disabled     = running;
    msgInput.disabled    = running;
    testProgress.style.display = running ? 'block' : 'none';
  }

  async function runTest() {
    if (!systemContext) {
      statusMsg.textContent = 'Load a character first, then click Apply.';
      document.getElementById('settings').classList.add('open');
      return;
    }
    if (testBtn.classList.contains('running')) {
      stopRequested = true;
      return;
    }

    history = [];
    chat.innerHTML = '';
    stopRequested = false;
    testSetRunning(true);

    const res       = await fetch('/test-questions');
    const data      = await res.json();
    const questions = data.questions;

    for (let i = 0; i < questions.length; i++) {
      if (stopRequested) {
        testProgress.textContent = `Test stopped at question ${i + 1}.`;
        break;
      }

      const injectReminder = (i > 0 && i % REMINDER_EVERY === 0);
      const q        = questions[i];
      const msgToLLM = injectReminder ? q + '\n\n' + HIDDEN_REMINDER_TEXT : q;

      testProgress.textContent = `Test running — question ${i + 1} of ${questions.length}`;
      addBubble('user', q);
      if (injectReminder) addReminderChip();

      const typing = document.createElement('div');
      typing.className = 'typing';
      typing.textContent = (charSelect.value || 'Agent') + ' is typing…';
      chat.appendChild(typing);
      chat.scrollTop = chat.scrollHeight;

      try {
        const r    = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msgToLLM, history, system_context: systemContext })
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
    testSetRunning(false);
    msgInput.focus();
  }

  testBtn.addEventListener('click', runTest);

  // ── Init ──
  loadCharacterList();
</script>
</body>
</html>"""


app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/characters")
def characters():
    files = sorted(CHARS_DIR.glob("character_*.txt"))
    names = [f.stem.replace("character_", "", 1) for f in files]
    return jsonify({"characters": names})


@app.route("/character/<name>")
def character(name):
    path = CHARS_DIR / f"character_{name}.txt"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify({"context": path.read_text(encoding="utf-8")})


@app.route("/chat", methods=["POST"])
def chat():
    data           = request.get_json()
    user_message   = data["message"]
    history        = data.get("history", [])
    system_context = data.get("system_context", "")

    messages = []
    for past_user, past_bot in history:
        messages.append({"role": "user",      "content": past_user})
        messages.append({"role": "assistant", "content": past_bot})

    full_ctx  = (system_context + FORMATTING_REMINDER) if system_context else ""
    first_msg = (full_ctx + "\n\n" + user_message) if (not history and full_ctx) else user_message
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

    return jsonify({"reply": reply})


@app.route("/test-questions")
def test_questions():
    path = CHARS_DIR / "test_questions.txt"
    if not path.exists():
        return jsonify({"questions": []})
    questions = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
