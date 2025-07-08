# Hearts Combined Script

## What does this script actually do?

This script helps you take your daily media spend data (from multiple channels like audio, video, and social) and apply a set of business rules or corrections from an instructions file. The main goal: make sure your final numbers match what you want—especially making sure spend is zeroed out for any package or campaign where it should be.

## How does it work?

1. **You provide your files.**
   - When you run the script, it will ask you for your data file (e.g., `data.csv`), your instructions file (e.g., `instructions.csv`), and the year you want to process.

2. **It cleans and prepares everything.**
   - The script standardizes column names, cleans up numbers (so '0', '0.00', and 0 all mean the same thing), and makes sure dates and periods are in a format that can be matched.

3. **It goes channel by channel.**
   - For each channel (audio, video, social), it looks at the instructions and finds the matching rows in your daily data.
   - For each instruction, it:
     - Finds all the matching rows (by channel, package/campaign, and period/month).
     - If the instruction says spend should be zero, it sets all those rows' spend to zero—no matter what.
     - Otherwise, it applies any scaling, CPM, or flat fee logic as needed, so your numbers match the instructions.

4. **It double-checks at the end.**
   - Even after all the above, it does a final sweep: for every instruction that says spend should be zero, it goes back and zeros out any matching rows in the final output. This is your safety net.

5. **It saves your results.**
   - You get an adjusted output file (`adjusted_output_combined.csv`) with all the corrections applied.
   - You also get a status file (`instructions_with_status_combined.csv`) that tells you which instructions were applied and if any didn't match.
   - A log file (`hearts_combined.log`) is created so you can see what happened, including any warnings or rows that didn't match.

## What do you need to run it?
- Python 3 and pandas installed.
- Your daily data file (CSV, must have a `Day of date` column and all your usual campaign columns).
- Your instructions file (CSV, must have at least `channel`, `package` or `campaign_name`, `month`, and `correct_spend`).

## What will you see?
- When you run the script, you'll be prompted for your filenames and year.
- The script will process everything and print a success message when done.
- If there are any issues (like a package in the instructions that doesn't match your data), you'll see a warning in the log file.

## Tips and troubleshooting
- If you see spend that isn't zeroed when you expect, check that `correct_spend` in your instructions is really a number (not a string or blank).
- If you get a warning about no match found, double-check your package/campaign names and periods in both files.
- The log file is your friend! It will tell you exactly what was matched, what was zeroed, and what was skipped.
- For big files, the script is pretty fast, but make sure your computer has enough memory.

## Customizing the script
- Want to keep different columns in your output? Edit the `header` list in the script.
- Need to change how corrections are applied? Look for the per-instruction logic in the script and tweak as needed.
- Want more logging? Add more `logging.info()` or `logging.warning()` calls wherever you want more detail.

---

**In short:**
- You give it your data and instructions.
- It makes sure your spend and impressions match your business rules.
- It double-checks that zeros are really zeros.
- You get clean, corrected output and a log of everything that happened.

If you have questions or want to tweak how it works, just open up the script and make it your own!

