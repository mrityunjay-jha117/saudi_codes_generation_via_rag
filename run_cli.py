import pandas as pd
import os
import time
import csv
import config
from matcher import CodeMatcher
import asyncio

# --- Configuration ---
INPUT_FILE = "data/input/input_file.xlsx"  # Change this to your actual file name
OUTPUT_FILE = "data/output/final_output.xlsx"
CHECKPOINT_FILE = "data/output/checkpoint_progress.csv"

def process_file_safely():
    """
    Runs the matching process row-by-row and saves to CSV immediately.
    Safe for small RAM execution and prevents data loss.
    """
    print("Initializing Matcher...")
    matcher = CodeMatcher()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Check input
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        print("Please upload your file to 'data/input/' folder via FileZilla.")
        return

    print(f"Reading {INPUT_FILE}...")
    xls = pd.ExcelFile(INPUT_FILE)
    
    # Prepare Checkpoint CSV
    file_exists = os.path.exists(CHECKPOINT_FILE)
    csv_headers = [
        "Sheet", "Input Description", "Matched Code", "Code System", 
        "Matched Description", "Confidence", "Reasoning"
    ]
    
    # Open CSV in append mode
    with open(CHECKPOINT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_headers)
            
        total_sheets = len(xls.sheet_names)
        
        for i, sheet_name in enumerate(xls.sheet_names):
            print(f"\nProcessing Sheet {i+1}/{total_sheets}: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            if "Service Description" not in df.columns:
                print(f"Skipping {sheet_name} (Missing 'Service Description' column)")
                continue
                
            # Filter for already processed rows if resuming?
            # For simplicity in this script, we append. You can manually delete csv to restart.
            
            rows = df.to_dict('records')
            total_rows = len(rows)
            
            for idx, row in enumerate(rows):
                desc = str(row.get("Service Description", "")).strip()
                
                # Visual logging
                print(f"[{idx+1}/{total_rows}] Matching: {desc[:50]}...", end="\r")
                
                if not desc or desc.lower() == 'nan':
                    writer.writerow([sheet_name, desc, "", "", "", "NONE", "Empty/Invalid"])
                    continue
                
                # Perform Match
                try:
                    result = matcher.match_single(desc)
                    
                    # Write to CSV IMMEDIATELY
                    writer.writerow([
                        sheet_name,
                        desc,
                        result.get("matched_code", ""),
                        result.get("code_system", ""),
                        result.get("matched_description", ""),
                        result.get("confidence", "NONE"),
                        result.get("reasoning", "")
                    ])
                    f.flush()  # Force write to disk
                    
                except Exception as e:
                    print(f"\nError on row {idx+1}: {e}")
                    writer.writerow([sheet_name, desc, "ERROR", "ERROR", str(e), "NONE", "Error"])
                    f.flush()

    print(f"\n\nProcessing Complete! Checkpoint saved to {CHECKPOINT_FILE}")
    print("Converting Checkpoint CSV to Final Excel...")
    
    # Convert CSV to Excel for final easy reading
    try:
        df_final = pd.read_csv(CHECKPOINT_FILE)
        df_final.to_excel(OUTPUT_FILE, index=False)
        print(f"Final Excel saved: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Could not create Excel (CSV is safe though): {e}")

if __name__ == "__main__":
    process_file_safely()
