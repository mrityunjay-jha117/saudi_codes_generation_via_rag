#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys

# Add current directory to path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from matcher import CodeMatcher

async def main():
    parser = argparse.ArgumentParser(description="Run Saudi Code Matcher via CLI")
    parser.add_argument("input_file", nargs="?", default="data/input/input.xlsx", help="Path to input Excel file")
    parser.add_argument("output_file", nargs="?", default="data/output/output.xlsx", help="Path to output Excel file")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        print("Please upload your Excel file or specify the correct path.")
        print(f"Example: python run_cli.py my_data.xlsx results.xlsx")
        return

    print("="*60)
    print("SAUDI CODE MATCHER - CLI MODE")
    print("="*60)
    print(f"Input File:  {args.input_file}")
    print(f"Output File: {args.output_file}")
    print(f"Model:       {config.LLM_MODEL}")
    print("-" * 60)
    
    try:
        print("Initializing Matcher (connecting to databases)...")
        matcher = CodeMatcher()
        
        print("\nStarting Async Matching Process...")
        print("This may take a few minutes depending on file size.\n")
        
        # Run the async batch matcher
        await matcher.match_batch_async(args.input_file, args.output_file)
        
        print("\n" + "="*60)
        print(f"SUCCESS! Output saved to: {args.output_file}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.name == 'nt':  # Windows
        asyncio.run(main())
    else:  # Unix/Linux (handling loops properly)
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
