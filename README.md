# Simulated Patient (PT chat agent)

A Flask web app that lets PT students practice history-taking with Ana Lopez, a simulated patient with a left ankle injury. The app connects to a hosted LLM endpoint and is accessed from a browser on any machine that can reach the server.

---

## Requirements

- Python 3.10 or later
- Port **7860** open on the VM's firewall
- Internet connection from the VM (to reach the hosted model)

---

## 1 — Install Python (Ubuntu / Debian VM)

```bash
sudo apt update && sudo apt install -y python3 python3-pip
python3 --version
```

---

## 2 — Clone the repo on the VM

```bash
git clone https://github.com/Andrelhu/Simulated-PT-Patient.git
cd Simulated-PT-Patient
```

---

## 3 — Install dependencies

```bash
pip3 install -r requirements.txt
```

---

## 4 — Run the app

```bash
python3 app.py
```

The terminal will print a line like:

```
Running on http://0.0.0.0:7860
```

Open a browser on your local machine and go to:

```
http://<VM-IP-ADDRESS>:7860
```

Replace `<VM-IP-ADDRESS>` with the public or internal IP of your VM. The app will keep running until you press `Ctrl+C` in the console.

---

## 5 — Keep it running after you close the console (optional)

Use `nohup` to detach the process so it survives when you log out:

```bash
nohup python3 app.py > app.log 2>&1 &
echo "PID: $!"
```

To stop it later, find the PID and kill it:

```bash
kill <PID>
```

Or use `screen` / `tmux` if available on your VM.

---

## Usage

Type your questions in the chat box as if you are a PT student conducting an initial history-taking interview with Ana. The simulation is adaptive — Ana's responses change based on how focused and professional your questions are.
