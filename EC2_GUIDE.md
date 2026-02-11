# EC2 Deployment Guide for Saudi Code Matcher

This guide explains how to run your Python script on a small AWS EC2 instance safely using `tmux` and `FileZilla`.

## Prerequisites

- An active AWS EC2 Instance (Ubuntu or Amazon Linux recommended).
- The `.pem` key file you downloaded when creating the instance.
- **FileZilla** installed on your computer.

---

## Step 1: Connect to Your EC2 Instance

1. Open your terminal (or Command Prompt on Windows).
2. Navigate to where your `.pem` key is.
3. Run this command (replace with your details):
   ```bash
   ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
   ```
   _(If using Amazon Linux, use `ec2-user@...` instead of `ubuntu@...`)_

---

## Step 2: Set Up the Environment (First Time Only)

Run these commands one by one to install Python and necessary tools:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python pip and basic tools
sudo apt install python3-pip python3-venv htop tmux -y

# Create a project folder
mkdir saudi_matcher
cd saudi_matcher

# Create a virtual environment (recommended for small RAM)
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Transfer Files using FileZilla

1. Open **FileZilla**.
2. Go to **File > Site Manager > New Site**.
   - **Protocol**: SFTP - SSH File Transfer Protocol.
   - **Host**: Your EC2 Public IP.
   - **Logon Type**: Key file.
   - **User**: `ubuntu` (or `ec2-user`).
   - **Key file**: Browse and select your `.pem` file.
3. Click **Connect**.
4. On the **Remote site** (right side), navigate to `/home/ubuntu/saudi_matcher`.
5. Upload these files from your local computer:
   - `run_cli.py` (The new script I created for you)
   - `matcher.py`
   - `config.py`
   - `prompts.py`
   - `.env` (Make sure this has your PINECONE_API_KEY and AWS credentials)
   - `requirements.txt` (Create this locally if you haven't: usage `pip freeze > requirements.txt`)
   - Create a folder `data/input` on the server and upload your Excel file there. **Rename your excel file to `input_file.xlsx`** or update the script.

---

## Step 4: Install Python Libraries

Back in your SSH terminal:

```bash
# Make sure you are in the folder and venv is active
source venv/bin/activate

# Install requirements (if you uploaded requirements.txt)
pip install -r requirements.txt

# OR Install manually if you don't have requirements.txt
pip install pandas openpyxl pinecone-client boto3 python-dotenv langchain
```

---

## Step 5: Run with Tmux (The Important Part)

`tmux` allows the script to keep running even if you close your computer or disconnect.

1. **Start a new tmux session**:

   ```bash
   tmux new -s matcher_job
   ```

   _(You will see a fresh terminal screen with a green bar at the bottom)_

2. **Run the script**:

   ```bash
   # Ensure venv is active
   source venv/bin/activate

   # Run the robust CLI script
   python run_cli.py
   ```

3. **Detach from tmux** (Leave it running in background):
   - Press `Ctrl + B`, then release both and press `D`.
   - You will return to your normal terminal. The script is still running!

---

## Step 6: Monitor or Stop

- **To check the process later**:
  ```bash
  tmux attach -t matcher_job
  ```
- **To check RAM usage**: Open a new terminal connection and run `htop`.
- **To kill the process**: Inside tmux, press `Ctrl + C`.

---

## Step 7: Download Results

1. Once the script finishes, open FileZilla using the same connection.
2. Navigate to `saudi_matcher/data/output/`.
3. You will see:
   - `checkpoint_progress.csv` (Real-time logs, saved row-by-row)
   - `final_output.xlsx` (Final formatted Excel)
4. Drag them to your local computer.

## Safety Features Added

- **Incremental Saving**: The `run_cli.py` writes to a `.csv` file after EVERY row. If the server crashes or runs out of RAM, you only lose the last row being processed.
- **Low RAM Usage**: The script processes row-by-row instead of loading massive objects.
