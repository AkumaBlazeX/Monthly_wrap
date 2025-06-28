# Monthly Wrap Report QA Automation Tool

This tool automates the aggregation and QA of daily-grain marketing data for monthly wrap reports. It processes a CSV file (e.g., `data.csv`) containing campaign, package, spend, and impression data, and outputs a summarized instructions file for further analysis or reporting.

## Features
- Groups data by month and unique package (or campaign if package is missing)
- Sums Spend and Impressions for each group
- Handles missing or invalid Spend values by treating them as 0.0
- Outputs a clean CSV (`instructions_output.csv`) ready for reporting
- Logs any data issues found during processing

## Requirements
- Python 3.8+
- pandas
- numpy

## Setup
1. **Clone the repository**
2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install pandas numpy
   ```

## Usage
1. Place your input data file (e.g., `data.csv`) in the project directory.
2. Run the script:
   ```bash
   source venv/bin/activate  # if not already activated
   python generate_instructions.py
   ```
3. When prompted, enter the name of your data file (e.g., `data.csv`).
4. The output will be saved as `instructions_output.csv` in the same directory.

## Output
The output file (`instructions_output.csv`) contains the following columns:
- `channel`
- `campaign_name` (blank if package is present)
- `package` (blank if not present)
- `month` (YYYY-MM)
- `Spend` (numeric, sum for the group)
- `Impressions` (numeric, sum for the group)

## Troubleshooting
- If you see a `DtypeWarning`, it is safe to ignore unless you have issues with specific columns.
- All Spend values are cleaned and converted to numeric. Invalid or missing values are set to 0.0.
- If you encounter errors, ensure your input file matches the expected column structure (see sample header in `data.csv`).

## License
MIT License
