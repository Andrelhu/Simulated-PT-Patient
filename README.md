# Ana — Simulated PT Patient

A Gradio web app that lets PT students practice history-taking with Ana Lopez, a simulated patient with a left ankle injury. The app connects to a hosted LLM endpoint and opens automatically in your browser.

---

## Requirements

- Python 3.10 or later
- Internet connection (to reach the hosted model)

---

## 1 — Install Python

### Windows
1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest **Python 3.x** installer.
2. Run the installer. **Check the box that says "Add Python to PATH"** before clicking Install.
3. Open **Command Prompt** and verify: `python --version`

### macOS
1. Open **Terminal**.
2. Install via Homebrew (recommended):
   ```
   brew install python
   ```
   Or download the installer from [python.org/downloads](https://www.python.org/downloads/).
3. Verify: `python3 --version`

### Linux (Ubuntu / Debian)
```bash
sudo apt update && sudo apt install -y python3 python3-pip
python3 --version
```

---

## 2 — Download the project files

Clone the repository or download it as a ZIP from GitHub:

```bash
git clone https://github.com/Andrelhu/Simulated-PT-Patient.git
cd Simulated-PT-Patient
```

---

## 3 — Install dependencies

From inside the project folder, run:

```bash
pip install -r requirements.txt
```

On macOS/Linux you may need `pip3` instead of `pip`.

---

## 4 — Run the app

```bash
python app.py
```

On macOS/Linux:

```bash
python3 app.py
```

A browser window will open automatically at `http://127.0.0.1:7860`. If it does not open, paste that address into your browser manually.

---

## Usage

Type your questions in the chat box as if you are a PT student conducting an initial history-taking interview with Ana. The simulation is adaptive — Ana's responses change based on how focused and professional your questions are.

To stop the app, press `Ctrl+C` in the terminal.
