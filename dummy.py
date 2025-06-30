import pandas as pd

# Read the input file
df = pd.read_csv('instructions_output.csv')

# Add 30,000 to Spend and 1,000,000 to Impressions in new columns
# Use fillna(0) to ensure no NaN values interfere

df['corrected_spend'] = df['Spend'].fillna(0) + 30000

df['corrected_impressions'] = df['Impressions'].fillna(0) + 1_000_000

# Extract method: last but one underscore-separated string from 'package'
def extract_method(pkg):
    if pd.isna(pkg):
        return ''
    parts = str(pkg).split('_')
    if len(parts) >= 2:
        return parts[-2]
    return ''

df['method'] = df['package'].apply(extract_method)

# Calculate ecpm: corrected_spend / corrected_impressions * 1000
def calc_ecpm(row):
    if row['corrected_impressions'] == 0:
        return 0.0
    return row['corrected_spend'] / row['corrected_impressions'] * 1000

df['ecpm'] = df.apply(calc_ecpm, axis=1)

# Write to output file
df.to_csv('dummy_testing.csv', index=False)

print('Dummy data created and saved to dummy_testing.csv') 