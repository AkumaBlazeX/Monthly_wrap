import pandas as pd

# Read the input file
df = pd.read_csv('dummy_testing.csv')

# For each (package, month) group, scale Impressions and Spend
for (pkg, month), group in df.groupby(['package', 'month']):
    total_impr = group['Impressions'].sum()
    total_corr = group['corrected_impressions'].sum()
    if pd.notna(total_corr) and total_impr > 0:
        scale = total_corr / total_impr
        idxs = group.index
        df.loc[idxs, 'Impressions'] = df.loc[idxs, 'Impressions'] * scale
        df.loc[idxs, 'Spend'] = df.loc[idxs, 'Spend'] * scale
    # If total_impr is 0 or total_corr is NaN, skip adjustment for this group

# Optionally round values for clarity
# df['Impressions'] = df['Impressions'].round(0)
# df['Spend'] = df['Spend'].round(2)

# Write the output file
output_file = 'hearts_output.csv'
df.to_csv(output_file, index=False)
print(f'Adjusted Spend and Impressions written to {output_file}') 