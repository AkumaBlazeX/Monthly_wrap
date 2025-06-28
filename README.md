# Monthly Wrap Report QA Automation Tool

This tool automates the aggregation and QA of daily-grain marketing data for monthly wrap reports. It processes a CSV file (e.g., `data.csv`) containing campaign, package, spend, and impression data, and outputs a summarized instructions file for further analysis or reporting.

## Features
- Groups data by month and unique package (or campaign if package is missing)
- Sums Spend and Impressions for each group
- Handles missing or invalid Spend values by treating them as 0.0
- Outputs a clean CSV (`instructions_output.csv`) ready for reporting
- Logs any data issues found during processing
- **Diagnostics:**
  - Normalizes column names (removes leading/trailing spaces)
  - Checks for required columns and warns if any are missing
  - Warns if any rows have invalid or missing dates
  - Warns if any Spend or Impressions values are negative
  - Prints summary statistics for Spend and Impressions

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

## Diagnostics & Troubleshooting
- **Column Normalization:** The script removes leading/trailing spaces from column names and prints both original and normalized names.
- **Required Columns:** If any required columns are missing, the script will print an error and may not proceed.
- **Date Validation:** If any rows have invalid or missing dates in the `Day of date` column, a warning will be printed with the count of affected rows. These rows will have `NaN` for their month and may be grouped together or excluded in further processing.
- **Negative Values:** The script warns if any Spend or Impressions values are negative.
- **Summary Statistics:** The script prints summary statistics (count, mean, min, max, etc.) for Spend and Impressions after cleaning.
- **No Data Issues:** If no issues are found, the script will print `No data issues found.`

### What to do if you see warnings
- **Invalid or missing dates:** Review the affected rows in your input file. You may want to correct or remove them if they are important for your analysis.
- **Missing columns:** Ensure your input file matches the expected column structure (see sample header in `data.csv`).
- **Negative values:** Check if negative Spend or Impressions are expected in your data. If not, investigate and correct them in your source file.

## License
MIT License 
