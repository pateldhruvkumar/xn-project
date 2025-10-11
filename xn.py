# =========================
# ALY6080 - Experimental Learning
# =========================
# Requires: pandas, openpyxl, matplotlib, seaborn
# Install if needed:
# pip install pandas openpyxl matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---- 1) Loading data ----
# File path for the dataset
file_path = "D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx"

df = pd.read_excel(file_path, sheet_name=0)

# WorkedDate parsing
if "Worked Date" in df.columns:
    df["Worked Date"] = pd.to_datetime(df["Worked Date"], errors='coerce')

# Ensure numeric columns are numeric
num_cols = [col for col in ["Billable Hours", "Billed Hours", "Hourly Billing Rate",
                             "Extended Price", "Amount Billed"] if col in df.columns]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace NAs in numeric columns with 0
df[num_cols] = df[num_cols].fillna(0)

# ---- 2) Core derived features ----
df['billingEfficiencyPct'] = df.apply(
    lambda row: 100 * (row['Billed Hours'] / row['Billable Hours']) 
    if row['Billable Hours'] > 0 else None, axis=1
)

df['revenuePerBilledHour'] = df.apply(
    lambda row: row['Amount Billed'] / row['Billed Hours'] 
    if row['Billed Hours'] > 0 else None, axis=1
)

df['revenuePerBillableHour'] = df.apply(
    lambda row: row['Amount Billed'] / row['Billable Hours'] 
    if row['Billable Hours'] > 0 else None, axis=1
)

df['effectiveRateVsListPct'] = df.apply(
    lambda row: 100 * (row['revenuePerBilledHour'] / row['Hourly Billing Rate']) 
    if row['Hourly Billing Rate'] > 0 and pd.notna(row['revenuePerBilledHour']) else None, axis=1
)

df['differenceExtRev'] = df['Extended Price'] - df['Amount Billed']

df['discountPct'] = df.apply(
    lambda row: 100 * (row['differenceExtRev'] / row['Extended Price']) 
    if row['Extended Price'] > 0 else None, axis=1
)

# ---- 3) Calculating new features ----
by_client_fy = df.groupby(['Client_Name', 'Fiscal_Year']).agg(
    projects=('Project Name', 'nunique'),
    totalBillableHr=('Billable Hours', 'sum'),
    totalBilledHr=('Billed Hours', 'sum'),
    revenue=('Amount Billed', 'sum'),
    extPrice=('Extended Price', 'sum')
).reset_index()

# Calculate weighted average list rate
weighted_rates = df.groupby(['Client_Name', 'Fiscal_Year']).apply(
    lambda x: (x['Hourly Billing Rate'] * x['Billed Hours']).sum() / x['Billed Hours'].sum() 
    if x['Billed Hours'].sum() > 0 else 0
).reset_index(name='avgListRate')

by_client_fy = by_client_fy.merge(weighted_rates, on=['Client_Name', 'Fiscal_Year'])

# Calculate additional metrics
by_client_fy['billingEfficiencyPct'] = by_client_fy.apply(
    lambda row: 100 * (row['totalBilledHr'] / row['totalBillableHr']) 
    if row['totalBillableHr'] > 0 else 0, axis=1
)

by_client_fy['effectiveRateVsListPct'] = by_client_fy.apply(
    lambda row: 100 * ((row['revenue'] / max(row['totalBilledHr'], 1e-9)) / row['avgListRate']) 
    if row['avgListRate'] > 0 else 0, axis=1
)

by_client_fy['differenceExtRev'] = by_client_fy.apply(
    lambda row: 0 if row['revenue'] == 0 else row['extPrice'] - row['revenue'], axis=1
)

by_client_fy['differenceExtRevPercentage'] = by_client_fy.apply(
    lambda row: 0 if row['revenue'] == 0 else 
    (100 * ((row['extPrice'] - row['revenue']) / row['extPrice']) if row['extPrice'] > 0 else None), axis=1
)

# Fill NAs
by_client_fy['avgListRate'] = by_client_fy['avgListRate'].fillna(0)
by_client_fy['billingEfficiencyPct'] = by_client_fy['billingEfficiencyPct'].fillna(0)
by_client_fy['effectiveRateVsListPct'] = by_client_fy['effectiveRateVsListPct'].fillna(0)

# Sort by revenue descending
by_client_fy = by_client_fy.sort_values('revenue', ascending=False)

# Rename columns for easier access
by_client_fy.columns = ['clientName', 'fiscalYear', 'projects', 'totalBillableHr', 
                        'totalBilledHr', 'revenue', 'extPrice', 'avgListRate', 
                        'billingEfficiencyPct', 'effectiveRateVsListPct', 
                        'differenceExtRev', 'differenceExtRevPercentage']

print(by_client_fy)

# ---- 4) Visualization ----

# 4.1 Number of Projects by Fiscal Year
projects_by_fy = by_client_fy.groupby('fiscalYear')['projects'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(projects_by_fy['fiscalYear'], projects_by_fy['projects'], color='skyblue')
plt.title('Number of Projects by Fiscal Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Projects')
plt.tight_layout()
plt.show()

# 4.2 Billing Efficiency (< 95%) vs Client Name (excluding 0)
filtered_data = by_client_fy[
    (by_client_fy['billingEfficiencyPct'].notna()) & 
    (by_client_fy['billingEfficiencyPct'] < 95) & 
    (by_client_fy['billingEfficiencyPct'] > 0)
].sort_values('billingEfficiencyPct')

plt.figure(figsize=(10, 8))
plt.barh(filtered_data['clientName'], filtered_data['billingEfficiencyPct'], color='skyblue')
plt.title('Billing Efficiency (0 < % < 95) vs Client Name')
plt.xlabel('Billing Efficiency (%)')
plt.ylabel('Client Name')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# 4.3 Billing Efficiency (> 100%) vs Client Name for FY22
filtered_data_high = by_client_fy[
    (by_client_fy['billingEfficiencyPct'].notna()) & 
    (by_client_fy['billingEfficiencyPct'] > 100) & 
    (by_client_fy['billingEfficiencyPct'] < 600) &
    (by_client_fy['fiscalYear'] == 'FY22')
].sort_values('billingEfficiencyPct')

plt.figure(figsize=(10, 8))
plt.barh(filtered_data_high['clientName'], filtered_data_high['billingEfficiencyPct'], color='skyblue')
plt.title('Billing Efficiency (100 < % < 600) vs Client Name')
plt.xlabel('Billing Efficiency (%)')
plt.ylabel('Client Name')
plt.tight_layout()
plt.show()

# 4.4 Discount % (>0) vs Client Name by Fiscal Year
filtered_discount = by_client_fy[
    (by_client_fy['differenceExtRevPercentage'].notna()) & 
    (by_client_fy['differenceExtRevPercentage'] > 0)
]

fiscal_years = filtered_discount['fiscalYear'].unique()
fig, axes = plt.subplots(len(fiscal_years), 1, figsize=(12, 6 * len(fiscal_years)))

if len(fiscal_years) == 1:
    axes = [axes]

for idx, fy in enumerate(fiscal_years):
    fy_data = filtered_discount[filtered_discount['fiscalYear'] == fy].sort_values('differenceExtRevPercentage')
    axes[idx].barh(fy_data['clientName'], fy_data['differenceExtRevPercentage'], color='skyblue')
    axes[idx].set_title(f'Difference % (>0) vs Client Name - {fy}')
    axes[idx].set_xlabel('Percentage Difference between Extended and Revenue Price (%)')
    axes[idx].set_ylabel('Client Name')

plt.tight_layout()
plt.show()

# 4.5 Difference by Fiscal Year
diff_by_fy = by_client_fy.groupby('fiscalYear')['differenceExtRevPercentage'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(diff_by_fy['fiscalYear'], diff_by_fy['differenceExtRevPercentage'], color='skyblue')
plt.title('Difference (Extended vs Revenue) by Fiscal Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Percentage Difference')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# 4.6 Client Vs Revenue
by_client_fy_sorted = by_client_fy.sort_values('revenue')
plt.figure(figsize=(10, 12))
plt.barh(by_client_fy_sorted['clientName'], by_client_fy_sorted['revenue'] / 1000, color='darkorange')
plt.title('Clients by Revenue')
plt.xlabel('Revenue (in K)')
plt.ylabel('Client Name')
plt.tight_layout()
plt.show()

# Display basic info
print("\nDataFrame Info:")
print(by_client_fy.info())
print("\nDataFrame Description:")
print(by_client_fy.describe())