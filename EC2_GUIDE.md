# EC2 Commands Reference List

## 1. Connection (Login)

_Run this from your local computer terminal (Windows CMD/PowerShell)_

```bash
ssh -i "your.pem" ubuntu@your-ec2-ip
```

## 2. Transfer Code

**Automatic Method (Recommended):**

1.  Edit `deploy.bat` and put your Key Name and EC2 IP address in it.
2.  Double-click `deploy.bat`!

This will upload `run_cli.py`, `matcher.py`, `config.py`, `prompts.py`, `requirements.txt`, `.env`, and the `data/` folder.

**Manual Method (Command Line):**

If you cannot use `deploy.bat`, use `scp`:

```bash
scp -i "your-key.pem" run_cli.py app.py config.py matcher.py prompts.py requirements.txt .env ubuntu@your-ec2-ip:~/saudi_matcher/
scp -i "your-key.pem" -r data docs ubuntu@your-ec2-ip:~/saudi_matcher/
```

## 3. Server Setup (First Time Only)

_Run these on the EC2 server to update and install tools_

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python, Pip, Virtualenv, Tmux, and Htop
sudo apt install python3-pip python3-venv tmux htop -y

# Create project directory (if not uploaded automatically)
mkdir -p saudi_matcher

# Enter the directory
cd saudi_matcher
```

## 4. Python Environment Setup

```bash
# Create virtual environment (once)
python3 -m venv venv

# Activate virtual environment (run this every time you log in)
source venv/bin/activate
```

_Tip: After activation, you will see `(venv)` at the start of your command line._

## 5. Install Dependencies

_Make sure (venv) is active first_

```bash
# Option A: If you uploaded requirements.txt
pip install -r requirements.txt

# Option B: Install manually
pip install pandas openpyxl pinecone-client boto3 python-dotenv langchain
```

## 6. Running with Tmux (Keep script running in background)

`tmux` ensures your script doesn't stop if your internet disconnects.

### Start a New Session

```bash
tmux new -s matcher_job
```

### Run the Script (Inside Tmux)

```bash
# 1. Activate venv inside tmux
source venv/bin/activate

# 2. Run your script
python run_cli.py
```

### Detach (Exit Tmux but keep script running)

Press these keys in order:

1. `Ctrl` + `B`
2. Release both keys
3. Press `D`

### Re-connect to Session (Check progress)

```bash
tmux attach -t matcher_job
```

### Kill Session (Stop script forcefully)

```bash
tmux kill-session -t matcher_job
```

## 7. Monitoring & Maintenance

```bash
# Check CPU/RAM usage
htop

# Check disk space usage
df -h
```
