import re
import pandas as pd

file_path = "dataset/FY19_to_FY23.xlsx"
df = pd.read_excel(file_path)

bill_status_list = []
data_rows = []

for i in range(len(df)):
    worked_date_value = str(df.iloc[i]['Worked Date']).lower() if pd.notna(df.iloc[i]['Worked Date']) else ''
    
    if worked_date_value in ['billable', 'non billable', 'not billable']:
        continue
    
    has_data = (
        pd.notna(df.iloc[i]['Project Name']) or 
        pd.notna(df.iloc[i]['Task or Ticket Title']) or 
        pd.notna(df.iloc[i]['Billing Code Name']) or 
        pd.notna(df.iloc[i]['Billable Hours'])
    )
    
    if has_data:
        bill_status = 'Non Billable'
        
        if i + 1 < len(df):
            next_worked_date = str(df.iloc[i + 1]['Worked Date']).lower() if pd.notna(df.iloc[i + 1]['Worked Date']) else ''
            if next_worked_date == 'billable':
                bill_status = 'Billable'
            elif next_worked_date in ['non billable', 'not billable']:
                bill_status = 'Non Billable'
        
        data_rows.append(i)
        bill_status_list.append(bill_status)

cleaned_df = df.iloc[data_rows].copy()
cleaned_df['Bill'] = bill_status_list
cleaned_df = cleaned_df.reset_index(drop=True)

# ✅ Universal text cleaner (removes ^, *, bullets, extra spaces)
def _clean_text(x):
    if pd.isna(x):
        return x
    s = str(x)
    # Remove leading ^, *, bullet symbols, and spaces
    s = re.sub(r'^[\s\^*•●○◦▪▫-]+', '', s)
    # Remove ^, *, bullets anywhere in text
    s = re.sub(r'[\^*•●○◦▪▫-]+', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Apply cleaning to ALL columns (not just text ones)
for col in cleaned_df.columns:
    cleaned_df[col] = cleaned_df[col].apply(_clean_text)

# Keep Bill column next to Resource Name if present
columns_order = list(cleaned_df.columns)
if 'Bill' in columns_order:
    columns_order.remove('Bill')
    if 'Resource Name' in columns_order:
        idx = columns_order.index('Resource Name') + 1
        columns_order.insert(idx, 'Bill')
    else:
        columns_order.insert(4, 'Bill')
    cleaned_df = cleaned_df[columns_order]

output_path = "FY19_to_FY23_Cleaned.xlsx"
cleaned_df.to_excel(output_path, index=False)
