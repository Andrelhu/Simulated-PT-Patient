#!/usr/bin/env bash
# Install the app as a systemd service: no root, auto-restart, survives reboot.
#
#   sudo bash deploy/install.sh
#
# Checks everything that would make the service fail BEFORE enabling it, so a
# broken unit never replaces a working tmux process mid-trial.
set -euo pipefail

APP_USER=ubuntu
APP_DIR=/home/$APP_USER/Simulated-PT-Patient
UNIT=ptpatient
VENV=$APP_DIR/venv

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  ok   %s\n' "$*"; }
bad()  { printf '  FAIL %s\n' "$*"; FAILED=1; }
FAILED=0

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo bash deploy/install.sh" >&2
  exit 1
fi

say "Checking prerequisites"

[ -d "$APP_DIR" ] && ok "app directory $APP_DIR" || bad "no $APP_DIR"
[ -f "$APP_DIR/app2.py" ] && ok "app2.py present" || bad "app2.py missing"
[ -x "$VENV/bin/python3" ] && ok "venv present" || bad "no venv at $VENV"

if [ -x "$VENV/bin/gunicorn" ]; then
  ok "gunicorn installed ($("$VENV"/bin/gunicorn --version 2>&1 | head -1))"
else
  bad "gunicorn not installed — run: $VENV/bin/pip install -r $APP_DIR/requirements.txt"
fi

# The service runs as $APP_USER, so Path.home() becomes /home/$APP_USER.
# Running under sudo it was /root, which is why the key may only be there.
KEY=/home/$APP_USER/apikey.txt
if sudo -u "$APP_USER" test -r "$KEY"; then
  ok "$APP_USER can read $KEY"
else
  bad "$KEY is not readable by $APP_USER."
  if [ -r /root/apikey.txt ]; then
    echo "       The key exists at /root/apikey.txt (left over from running with sudo)."
    echo "       Copy it across:"
    echo "         sudo cp /root/apikey.txt $KEY"
    echo "         sudo chown $APP_USER:$APP_USER $KEY && sudo chmod 600 $KEY"
  fi
fi

# Config sanity, same checks the trial depends on.
if sudo -u "$APP_USER" "$VENV/bin/python3" "$APP_DIR/check_config.py" >/tmp/cfgcheck.$$ 2>&1; then
  ok "check_config.py passed"
  grep -E '^WARN' /tmp/cfgcheck.$$ | sed 's/^/       /' || true
else
  bad "check_config.py reported something that would break:"
  grep -E '^(BREAK|WARN)' /tmp/cfgcheck.$$ | sed 's/^/       /' || cat /tmp/cfgcheck.$$
fi
rm -f /tmp/cfgcheck.$$

# Writable by the service account.
for d in "$APP_DIR/sessions" "$APP_DIR/avatars"; do
  mkdir -p "$d"; chown -R "$APP_USER:$APP_USER" "$d"
done
ok "sessions/ and avatars/ writable by $APP_USER"

# The signing key must persist, or restarting logs every student out.
SECRET=$APP_DIR/.flask_secret
if [ -f "$SECRET" ]; then
  chown "$APP_USER:$APP_USER" "$SECRET"; chmod 600 "$SECRET"
  ok "signing key exists and is owned by $APP_USER"
else
  ok "signing key will be created on first start"
fi

if [ "$FAILED" -ne 0 ]; then
  say "Stopping: fix the FAIL items above, then run this again."
  echo "Nothing was changed. Your current process is untouched."
  exit 1
fi

say "Freeing port 80"
systemctl stop "$UNIT" 2>/dev/null || true
if command -v tmux >/dev/null && sudo -u "$APP_USER" tmux ls >/dev/null 2>&1; then
  echo "  tmux sessions found — killing them so they do not hold port 80:"
  sudo -u "$APP_USER" tmux ls 2>/dev/null | sed 's/^/       /'
  sudo -u "$APP_USER" tmux kill-server 2>/dev/null || true
fi
fuser -k 80/tcp 2>/dev/null || true
sleep 1
ok "port 80 free"

say "Installing the unit"
install -m 644 "$APP_DIR/deploy/$UNIT.service" "/etc/systemd/system/$UNIT.service"
systemctl daemon-reload
systemctl enable "$UNIT" >/dev/null
ok "/etc/systemd/system/$UNIT.service installed and enabled at boot"

say "Starting"
systemctl restart "$UNIT"
for i in $(seq 1 15); do
  sleep 1
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost/ 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
done

if [ "$code" = "200" ]; then
  ok "responding on http://localhost/ (HTTP $code)"
  say "Done"
  cat <<EOF
  The app now restarts on its own if it crashes, and comes back after a reboot.
  No tmux, no sudo.

    journalctl -u $UNIT -f          watch the logs live
    systemctl status $UNIT          is it running
    systemctl restart $UNIT         after a git pull
    systemctl stop $UNIT            take it down
EOF
else
  bad "not responding (HTTP $code). The unit is installed but something is wrong."
  echo
  echo "  Last 30 log lines:"
  journalctl -u "$UNIT" -n 30 --no-pager | sed 's/^/       /'
  echo
  echo "  Your old tmux process was stopped. To go back to it for now:"
  echo "    sudo systemctl stop $UNIT && sudo systemctl disable $UNIT"
  echo "    cd $APP_DIR && sudo venv/bin/python3 app2.py"
  exit 1
fi
