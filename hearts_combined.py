import pandas as pd
import numpy as np
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# -----------------------------
# User Inputs
# -----------------------------
data_file = input("Enter data file name (e.g., data.csv): ").strip()
instructions_file = input("Enter instructions file name (e.g., instructions.csv): ").strip()
year = input("Enter the year for the data (e.g., 2025): ").strip()

# -----------------------------
# Load and Normalize Data
# -----------------------------
raw_data = pd.read_csv(data_file, parse_dates=['Day of date'])
instructions = pd.read_csv(instructions_file)

# Normalize column names and strip spaces
raw_data.columns = raw_data.columns.str.lower().str.strip()
instructions.columns = instructions.columns.str.lower().str.strip()

# Rename columns with extra spaces to standard names
col_rename = {}
for col in raw_data.columns:
    if col.replace(' ', '') == 'spend':
        col_rename[col] = 'spend'
    if col.replace(' ', '') == 'impressions':
        col_rename[col] = 'impressions'
raw_data = raw_data.rename(columns=col_rename)

# Clean numeric columns
for col in ['spend', 'impressions', 'data_spend', 'data_impressions', 'correct_spend', 'correct_impressions', 'ecpm']:
    for df in [raw_data, instructions]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )

# Drop unnamed columns
raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('unnamed', case=False)]

# Map month names to numbers
month_map = {m: i for i, m in enumerate([
    'january','february','march','april','may','june','july','august','september','october','november','december'], 1)}

# Use a string version of the month for mapping
instructions['month_str'] = instructions['month'].astype(str)
instructions['month_num'] = instructions['month_str'].str.strip().str.lower().map(month_map)
instructions['period'] = pd.to_datetime(
    year + '-' + instructions['month_num'].astype(str) + '-01',
    errors='coerce'
).dt.to_period('M')

# Normalize month in raw_data
raw_data['period'] = raw_data['day of date'].dt.to_period('M')

# Calculate ecpm_ours for each instruction (if possible)
instructions['ecpm_ours'] = np.where(
    (instructions['correct_spend'].notna()) & (instructions['correct_impressions'].notna()) & (instructions['correct_impressions'] > 0),
    instructions['correct_spend'] / instructions['correct_impressions'] * 1000,
    np.nan
)

# Map method from package for Video, use as-is for Audio/others
def extract_method(row):
    channel = str(row.get('channel', '')).lower().strip()
    pkg = str(row.get('package', ''))
    method = str(row.get('method', '')).lower().strip()
    if channel == 'video':
        parts = pkg.split('_')
        if len(parts) >= 2:
            method_raw = parts[-2].strip().lower()
            if method_raw in ['pav', 'jav']:
                return 'added value'
            elif method_raw in ['cpm', 'dcpm']:
                return 'cpm'
            else:
                return method_raw
        else:
            return ''
    else:
        return method

instructions['method_ours'] = instructions.apply(extract_method, axis=1)

# Normalize channel values: treat 'Digital Audio' as 'Audio' for matching
raw_data['channel'] = raw_data['channel'].str.strip().str.lower().replace({'digital audio': 'audio'})
instructions['channel'] = instructions['channel'].str.strip().str.lower().replace({'digital audio': 'audio'})

# Only process Audio and Video channels
allowed_channels = ['audio', 'video']
instructions = instructions[instructions['channel'].isin(allowed_channels)]
raw_data = raw_data[raw_data['channel'].isin(allowed_channels)]

# Only process Audio channel in the main loop
allowed_channels_audio = ['audio']
audio_instructions = instructions[instructions['channel'].isin(allowed_channels_audio)]
audio_raw_data = raw_data[raw_data['channel'].isin(allowed_channels_audio)]

# Prepare output DataFrames for audio
adjusted_all = []
instructions_all = []

# Process only audio in the main loop
for channel in audio_instructions['channel'].dropna().unique():
    channel_mask = audio_instructions['channel'].str.lower().str.strip() == channel.lower().strip()
    inst_channel = audio_instructions[channel_mask].copy()
    data_channel = audio_raw_data[audio_raw_data['channel'].str.lower().str.strip() == channel.lower().strip()].copy()
    if inst_channel.empty or data_channel.empty:
        continue
    adjusted = data_channel.copy()
    inst_channel['status'] = None
    for index, row in inst_channel.iterrows():
        method = str(row['method_ours']).lower()
        pkg = str(row['package']).lower().strip()
        period = row['period']
        channel_val = str(row['channel']).lower().strip()
        correct_spend = row.get('correct_spend')
        correct_impressions = row.get('correct_impressions')
        ecpm_ours = row.get('ecpm_ours')
        mask = (
            (adjusted['channel'].astype(str).str.lower().str.strip() == channel_val) &
            (adjusted['package'].astype(str).str.lower().str.strip() == pkg) &
            (adjusted['period'] == period)
        )
        if adjusted[mask].empty:
            inst_channel.loc[index, 'status'] = 'error: no exact match found'
            continue
        actual_spend = adjusted.loc[mask, 'spend'].sum()
        actual_impressions = adjusted.loc[mask, 'impressions'].sum()
        if pd.notna(correct_spend) and correct_spend == 0:
            adjusted.loc[mask, 'spend'] = 0
            inst_channel.loc[index, 'status'] = 'applied'
            continue
        if pd.notna(correct_impressions) and actual_impressions > 0:
            scale = correct_impressions / actual_impressions
            adjusted.loc[mask, 'impressions'] = adjusted.loc[mask, 'impressions'] * scale
        if method == 'cpm' and pd.notna(ecpm_ours):
            adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'impressions'] * (ecpm_ours / 1000)
        if method in ['flat fee', 'added value', 'takeover'] and pd.notna(correct_spend):
            spend_diff = correct_spend - actual_spend
            total_imps = adjusted.loc[mask, 'impressions'].sum()
            if total_imps > 0:
                ratio = adjusted.loc[mask, 'impressions'] / total_imps
                adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'spend'] + ratio * spend_diff
        inst_channel.loc[index, 'status'] = 'applied'
    # Zero spend for all (channel, period, package) if any instruction for that combination has correct_spend == 0
    for index, row in inst_channel.iterrows():
        if pd.notna(row.get('correct_spend')) and row.get('correct_spend') == 0:
            pkg = str(row['package']).lower().strip()
            period = row['period']
            channel_val = str(row['channel']).lower().strip()
            mask_all = (
                (adjusted['channel'].astype(str).str.lower().str.strip() == channel_val) &
                (adjusted['package'].astype(str).str.lower().str.strip() == pkg) &
                (adjusted['period'] == period)
            )
            adjusted.loc[mask_all, 'spend'] = 0
    adjusted_all.append(adjusted)
    instructions_all.append(inst_channel)

# Move process_video definition here so it is available before use
def process_video(raw_data, instructions, year):
    """Process video channel using the logic from hearts-video.py and return adjusted and instructions_with_status DataFrames."""
    # Normalize column names and strip spaces
    raw_data.columns = raw_data.columns.str.lower().str.strip()
    instructions.columns = instructions.columns.str.lower().str.strip()

    # Rename columns with extra spaces to standard names
    col_rename = {}
    for col in raw_data.columns:
        if col.replace(' ', '') == 'spend':
            col_rename[col] = 'spend'
        if col.replace(' ', '') == 'impressions':
            col_rename[col] = 'impressions'
    raw_data = raw_data.rename(columns=col_rename)

    # Clean numeric columns
    for col in ['spend', 'impressions', 'data_spend', 'data_impressions', 'correct_spend', 'correct_impressions', 'ecpm']:
        for df in [raw_data, instructions]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').str.replace('$', ''),
                    errors='coerce'
                )

    # Drop unnamed columns
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('unnamed', case=False)]

    # Add 'month' column to raw_data
    raw_data['month'] = raw_data['day of date'].dt.to_period('M')

    # Enhanced month construction logic for instructions
    try:
        if 'month' in instructions.columns and (
            'end_date' in instructions.columns or 'start_date' in instructions.columns):
            if 'end_date' in instructions.columns:
                instructions['end_date'] = pd.to_datetime(instructions['end_date'], errors='coerce')
                year_col = instructions['end_date'].dt.year
            else:
                instructions['start_date'] = pd.to_datetime(instructions['start_date'], errors='coerce')
                year_col = instructions['start_date'].dt.year
            month_map = {m: i for i, m in enumerate([
                'january','february','march','april','may','june','july','august','september','october','november','december'], 1)}
            instructions['month_num'] = instructions['month'].str.strip().str.lower().map(month_map)
            instructions['month'] = pd.to_datetime(
                year_col.astype(str) + '-' + instructions['month_num'].astype(str) + '-01',
                errors='coerce'
            ).dt.to_period('M')
        elif 'end_date' in instructions.columns:
            instructions['end_date'] = pd.to_datetime(instructions['end_date'], errors='coerce')
            instructions['month'] = instructions['end_date'].dt.to_period('M')
        elif 'month' in instructions.columns:
            instructions['month'] = pd.to_datetime(instructions['month'], errors='coerce').dt.to_period('M')
        else:
            raise ValueError("No usable 'month', 'start_date', or 'end_date' column found in instructions file.")
    except Exception as e:
        raise

    # Map month names to numbers
    month_map = {m: i for i, m in enumerate([
        'january','february','march','april','may','june','july','august','september','october','november','december'], 1)}

    # Use a string version of the month for mapping
    instructions['month_str'] = instructions['month'].astype(str)
    instructions['month_num'] = instructions['month_str'].str.strip().str.lower().map(month_map)
    instructions['period'] = pd.to_datetime(
        year + '-' + instructions['month_num'].astype(str) + '-01',
        errors='coerce'
    ).dt.to_period('M')

    # Normalize month in raw_data
    raw_data['period'] = raw_data['day of date'].dt.to_period('M')

    # Filter for Video channel only
    raw_data = raw_data[raw_data['channel'].str.lower() == 'video']
    instructions = instructions[instructions['channel'].str.lower() == 'video']

    # Calculate ecpm_ours for each instruction
    instructions['ecpm_ours'] = np.where(
        (instructions['correct_spend'].notna()) & (instructions['correct_impressions'].notna()) & (instructions['correct_impressions'] > 0),
        instructions['correct_spend'] / instructions['correct_impressions'] * 1000,
        np.nan
    )

    # Map method from package
    def extract_method_ours(pkg):
        try:
            parts = str(pkg).split('_')
            if len(parts) >= 2:
                method_raw = parts[-2].strip().lower()
                if method_raw in ['pav', 'jav']:
                    return 'added value'
                elif method_raw in ['cpm', 'dcpm']:
                    return 'cpm'
                else:
                    return method_raw
            else:
                return ''
        except Exception:
            return ''

    instructions['method_ours'] = instructions['package'].apply(extract_method_ours)

    adjusted = raw_data.copy()
    instructions['status'] = None

    for index, row in instructions.iterrows():
        method = str(row['method_ours']).lower()
        pkg = str(row['package']).lower().strip()
        period = row['period']
        channel = str(row['channel']).lower().strip() if 'channel' in row else None
        correct_spend = row.get('correct_spend')
        correct_impressions = row.get('correct_impressions')
        ecpm_ours = row.get('ecpm_ours')
        mask = (
            (adjusted['channel'].astype(str).str.lower().str.strip() == channel) &
            (adjusted['package'].astype(str).str.lower().str.strip() == pkg) &
            (adjusted['period'] == period)
        )
        if adjusted[mask].empty:
            instructions.loc[index, 'status'] = 'error: no exact match found'
            continue
        actual_spend = adjusted.loc[mask, 'spend'].sum()
        actual_impressions = adjusted.loc[mask, 'impressions'].sum()
        if pd.notna(correct_spend) and correct_spend == 0:
            adjusted.loc[mask, 'spend'] = 0
            instructions.loc[index, 'status'] = 'applied'
            continue
        if pd.notna(correct_impressions) and actual_impressions > 0:
            scale = correct_impressions / actual_impressions
            adjusted.loc[mask, 'impressions'] = adjusted.loc[mask, 'impressions'] * scale
        if method == 'cpm' and pd.notna(ecpm_ours):
            adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'impressions'] * (ecpm_ours / 1000)
        if method in ['flat fee', 'added value', 'takeover'] and pd.notna(correct_spend):
            spend_diff = correct_spend - actual_spend
            total_imps = adjusted.loc[mask, 'impressions'].sum()
            if total_imps > 0:
                ratio = adjusted.loc[mask, 'impressions'] / total_imps
                adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'spend'] + ratio * spend_diff
        instructions.loc[index, 'status'] = 'applied'

    # Zero spend for all (channel, period, package) if any instruction for that combination has correct_spend == 0
    for index, row in instructions.iterrows():
        if pd.notna(row.get('correct_spend')) and row.get('correct_spend') == 0:
            pkg = str(row['package']).lower().strip()
            period = row['period']
            channel = str(row['channel']).lower().strip() if 'channel' in row else None
            mask_all = (
                (adjusted['channel'].astype(str).str.lower().str.strip() == channel) &
                (adjusted['package'].astype(str).str.lower().str.strip() == pkg) &
                (adjusted['period'] == period)
            )
            adjusted.loc[mask_all, 'spend'] = 0

    return adjusted, instructions

# Process video using the dedicated function
adjusted_video, instructions_video = process_video(raw_data.copy(), instructions.copy(), year)

# Process audio using the existing logic (filter and process as before)
# Normalize channel values for audio
raw_data_audio = raw_data[raw_data['channel'] == 'audio'].copy()
instructions_audio = instructions[instructions['channel'] == 'audio'].copy()

# Only run if there is audio data
adjusted_audio = pd.DataFrame()
instructions_audio_status = pd.DataFrame()
if not raw_data_audio.empty and not instructions_audio.empty:
    # Calculate ecpm_ours for each instruction (if possible)
    instructions_audio['ecpm_ours'] = np.where(
        (instructions_audio['correct_spend'].notna()) & (instructions_audio['correct_impressions'].notna()) & (instructions_audio['correct_impressions'] > 0),
        instructions_audio['correct_spend'] / instructions_audio['correct_impressions'] * 1000,
        np.nan
    )
    # Use method as is for audio
    instructions_audio['method_ours'] = instructions_audio['method']
    adjusted_audio = raw_data_audio.copy()
    instructions_audio['status'] = None
    for index, row in instructions_audio.iterrows():
        method = str(row['method_ours']).lower()
        pkg = str(row['package']).lower().strip()
        period = row['period']
        channel_val = str(row['channel']).lower().strip()
        correct_spend = row.get('correct_spend')
        correct_impressions = row.get('correct_impressions')
        ecpm_ours = row.get('ecpm_ours')
        mask = (
            (adjusted_audio['channel'].astype(str).str.lower().str.strip() == channel_val) &
            (adjusted_audio['package'].astype(str).str.lower().str.strip() == pkg) &
            (adjusted_audio['period'] == period)
        )
        if adjusted_audio[mask].empty:
            instructions_audio.loc[index, 'status'] = 'error: no exact match found'
            continue
        actual_spend = adjusted_audio.loc[mask, 'spend'].sum()
        actual_impressions = adjusted_audio.loc[mask, 'impressions'].sum()
        if pd.notna(correct_spend) and correct_spend == 0:
            adjusted_audio.loc[mask, 'spend'] = 0
            instructions_audio.loc[index, 'status'] = 'applied'
            continue
        if pd.notna(correct_impressions) and actual_impressions > 0:
            scale = correct_impressions / actual_impressions
            adjusted_audio.loc[mask, 'impressions'] = adjusted_audio.loc[mask, 'impressions'] * scale
        if method == 'cpm' and pd.notna(ecpm_ours):
            adjusted_audio.loc[mask, 'spend'] = adjusted_audio.loc[mask, 'impressions'] * (ecpm_ours / 1000)
        if method in ['flat fee', 'added value', 'takeover'] and pd.notna(correct_spend):
            spend_diff = correct_spend - actual_spend
            total_imps = adjusted_audio.loc[mask, 'impressions'].sum()
            if total_imps > 0:
                ratio = adjusted_audio.loc[mask, 'impressions'] / total_imps
                adjusted_audio.loc[mask, 'spend'] = adjusted_audio.loc[mask, 'spend'] + ratio * spend_diff
        instructions_audio.loc[index, 'status'] = 'applied'
    instructions_audio_status = instructions_audio

# Save original data and instructions for social processing
raw_data_orig = pd.read_csv(data_file, parse_dates=['Day of date'])
instructions_orig = pd.read_csv(instructions_file)
raw_data_orig.columns = raw_data_orig.columns.str.lower().str.strip()
instructions_orig.columns = instructions_orig.columns.str.lower().str.strip()

# Move process_social definition here so it is available before use
def process_social(raw_data, instructions, year):
    """Process social channel using the logic from hearts-social.py and return adjusted and instructions_with_status DataFrames."""
    # Normalize column names and strip spaces
    raw_data.columns = raw_data.columns.str.lower().str.strip()
    instructions.columns = instructions.columns.str.lower().str.strip()

    # Clean numeric columns
    for col in ['spend', 'impressions', 'data_spend', 'data_impressions', 'correct_spend', 'correct_impressions', 'ecpm']:
        for df in [raw_data, instructions]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').str.replace('$', ''),
                    errors='coerce'
                )

    # Drop unnamed columns
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('unnamed', case=False)]

    # Map month names to numbers
    month_map = {m: i for i, m in enumerate([
        'january','february','march','april','may','june','july','august','september','october','november','december'], 1)}

    # Use a string version of the month for mapping
    instructions['month_str'] = instructions['month'].astype(str)
    instructions['month_num'] = instructions['month_str'].str.strip().str.lower().map(month_map)
    instructions['period'] = pd.to_datetime(
        year + '-' + instructions['month_num'].astype(str) + '-01',
        errors='coerce'
    ).dt.to_period('M')

    # Normalize month in raw_data
    data_date_col = 'day of date' if 'day of date' in raw_data.columns else 'day_of_date'
    raw_data['period'] = raw_data[data_date_col].dt.to_period('M')

    # Filter for Social channel only
    raw_data = raw_data[raw_data['channel'].str.lower().str.strip() == 'social']
    instructions = instructions[instructions['channel'].str.lower().str.strip() == 'social']

    # Map method for each campaign in instructions using the first placement_name in data
    method_map = {}
    for campaign in instructions['campaign_name'].dropna().unique():
        placements = raw_data[raw_data['campaign_name'].astype(str).str.lower().str.strip() == campaign.lower().strip()]['placement_name']
        if not placements.empty:
            placement = str(placements.iloc[0])
            parts = placement.split('_')
            if len(parts) >= 2:
                method_raw = parts[-2].strip().lower()
                if method_raw in ['pav', 'jav']:
                    method = 'added value'
                elif method_raw in ['cpm', 'dcpm']:
                    method = 'cpm'
                else:
                    method = method_raw
            else:
                method = ''
            method_map[campaign] = method
        else:
            method_map[campaign] = ''

    instructions['method_ours'] = instructions['campaign_name'].map(method_map)

    # Calculate ecpm_ours for each instruction (use data_impressions if correct_impressions is missing or zero)
    def calc_ecpm_ours(row):
        correct_spend = row.get('correct_spend')
        correct_impressions = row.get('correct_impressions')
        data_impressions = row.get('data_impressions')
        if pd.notna(correct_spend) and correct_spend != 0:
            if pd.notna(correct_impressions) and correct_impressions > 0:
                return correct_spend / correct_impressions * 1000
            elif pd.notna(data_impressions) and data_impressions > 0:
                return correct_spend / data_impressions * 1000
        return np.nan
    instructions['ecpm_ours'] = instructions.apply(calc_ecpm_ours, axis=1)

    adjusted = raw_data.copy()
    instructions['status'] = None

    for index, row in instructions.iterrows():
        method = str(row['method_ours']).lower()
        campaign = str(row['campaign_name']).lower().strip()
        period = row['period']
        channel_val = str(row['channel']).lower().strip()
        correct_spend = row.get('correct_spend')
        correct_impressions = row.get('correct_impressions')
        ecpm_ours = row.get('ecpm_ours')
        mask = (
            (adjusted['channel'].astype(str).str.lower().str.strip() == channel_val) &
            (adjusted['campaign_name'].astype(str).str.lower().str.strip() == campaign) &
            (adjusted['period'] == period)
        )
        if adjusted[mask].empty:
            instructions.loc[index, 'status'] = 'error: no exact match found'
            continue
        actual_spend = adjusted.loc[mask, 'spend'].sum()
        actual_impressions = adjusted.loc[mask, 'impressions'].sum()
        if pd.notna(correct_spend) and correct_spend == 0:
            adjusted.loc[mask, 'spend'] = 0
            instructions.loc[index, 'status'] = 'applied'
            continue
        if pd.notna(correct_impressions) and actual_impressions > 0:
            scale = correct_impressions / actual_impressions
            adjusted.loc[mask, 'impressions'] = adjusted.loc[mask, 'impressions'] * scale
        if pd.notna(ecpm_ours):
            adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'impressions'] * (ecpm_ours / 1000)
        if method in ['flat fee', 'added value', 'takeover'] and pd.notna(correct_spend) and correct_spend != 0:
            spend_diff = correct_spend - actual_spend
            total_imps = adjusted.loc[mask, 'impressions'].sum()
            if total_imps > 0:
                ratio = adjusted.loc[mask, 'impressions'] / total_imps
                adjusted.loc[mask, 'spend'] = adjusted.loc[mask, 'spend'] + ratio * spend_diff
        instructions.loc[index, 'status'] = 'applied'

    # Zero spend for all (channel, period, campaign) if any instruction for that combination has correct_spend == 0
    for index, row in instructions.iterrows():
        if pd.notna(row.get('correct_spend')) and row.get('correct_spend') == 0:
            camp = str(row['campaign_name']).lower().strip()
            period = row['period']
            channel_val = str(row['channel']).lower().strip()
            mask_all = (
                (adjusted['channel'].astype(str).str.lower().str.strip() == channel_val) &
                (adjusted['campaign_name'].astype(str).str.lower().str.strip() == camp) &
                (adjusted['period'] == period)
            )
            adjusted.loc[mask_all, 'spend'] = 0

    return adjusted, instructions

# Process social using the dedicated function on original data
adjusted_social, instructions_social = process_social(raw_data_orig.copy(), instructions_orig.copy(), year)

frames_adjusted = []
frames_status = []
if adjusted_all:
    frames_adjusted.extend(adjusted_all)
    frames_status.extend(instructions_all)
if not adjusted_video.empty:
    frames_adjusted.append(adjusted_video)
    frames_status.append(instructions_video)
if not adjusted_social.empty:
    frames_adjusted.append(adjusted_social)
    frames_status.append(instructions_social)

# Ensure consistent columns for merging
all_columns = set()
for df in frames_adjusted:
    all_columns.update(df.columns)
for df in frames_status:
    all_columns.update(df.columns)
all_columns = list(all_columns)
frames_adjusted = [df.reindex(columns=all_columns) for df in frames_adjusted]
frames_status = [df.reindex(columns=all_columns) for df in frames_status]

adjusted_combined = pd.concat(frames_adjusted, ignore_index=True)
instructions_combined = pd.concat(frames_status, ignore_index=True)

# Filter columns for output to match data.csv plus period and month (case-insensitive)
header = [
    'funded_by','consumer_journey','target_market','campaign_initiatives','media_initiatives',
    'channel','publisher','funnel','audience_stage','audience_segment','account','campaign_name',
    'package','placement_name','creative_name','Day of date','Spend','Impressions','Clicks',
    'Site Visits','BFE','Online Lines','Upgrade Lines'
]
# Map DataFrame columns to lower for matching
col_map = {col.lower(): col for col in adjusted_combined.columns}
output_cols = []
for col in header:
    col_lc = col.lower()
    if col_lc in col_map:
        output_cols.append(col_map[col_lc])
for extra in ['period', 'month']:
    if extra in adjusted_combined.columns and extra not in output_cols:
        output_cols.append(extra)
adjusted_combined = adjusted_combined[output_cols]

# Save combined outputs
adjusted_combined.to_csv("adjusted_output_combined.csv", index=False, float_format="%.2f")
instructions_combined.to_csv("instructions_with_status_combined.csv", index=False, float_format="%.2f")

print("\n✅ Processing complete. Files saved:")
print("- adjusted_output_combined.csv")
print("- instructions_with_status_combined.csv") 
