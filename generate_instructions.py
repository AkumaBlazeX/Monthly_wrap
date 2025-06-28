import pandas as pd
import numpy as np
from datetime import datetime

# Prompt for input filename only
filename = input("Enter data file name (e.g., data.csv): ")
instructions_file = 'instructions_output.csv'

# Read the CSV file
df = pd.read_csv(filename)
print('Columns:', list(df.columns))  # Debug: print columns

# Helper function to parse month
def parse_month(date_str):
    try:
        return datetime.strptime(date_str, '%d-%b-%y').strftime('%Y-%m')
    except Exception:
        return np.nan

# Clean Spend and Impressions, log issues
issues = []
def to_numeric(val, col, idx):
    try:
        return float(str(val).replace('$', '').replace(',', ''))
    except Exception:
        issues.append(f'Non-numeric value in {col} at row {idx+2}: {val}')
        return 0.0  # Treat missing/invalid as 0.0

# Parse month column for grouping
df['group_month'] = df['Day of date'].apply(parse_month)

# Clean Spend and Impressions
df['Spend'] = [to_numeric(v, 'Spend', i) for i, v in enumerate(df['Spend'])]
df['Impressions'] = [to_numeric(v, 'Impressions', i) for i, v in enumerate(df['Impressions'])]

# Assign grouping columns directly

def assign_group_keys(row):
    if pd.notnull(row.get('package', None)) and str(row['package']).strip() != '':
        return pd.Series({
            'group_channel': row['channel'],
            'group_campaign_name': '',
            'group_package': row['package'],
            'group_month': row['group_month']
        })
    else:
        return pd.Series({
            'group_channel': row['channel'],
            'group_campaign_name': row['campaign_name'],
            'group_package': '',
            'group_month': row['group_month']
        })

df[['group_channel', 'group_campaign_name', 'group_package', 'group_month']] = df.apply(assign_group_keys, axis=1)

# Group and aggregate using the new unique columns
agg = df.groupby(['group_channel', 'group_campaign_name', 'group_package', 'group_month'], dropna=False).agg({
    'Spend': 'sum',
    'Impressions': 'sum'
}).reset_index()

# Output to instructions file with the required column names
agg.rename(columns={
    'group_channel': 'channel',
    'group_campaign_name': 'campaign_name',
    'group_package': 'package',
    'group_month': 'month'
}, inplace=True)
agg[['channel', 'campaign_name', 'package', 'month', 'Spend', 'Impressions']].to_csv(instructions_file, index=False)

# Print issues if any
if issues:
    print('Data issues found:')
    for issue in issues:
        print(issue)
else:
    print('No data issues found.') 