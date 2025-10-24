# =========================
# ALY6080 - Experimental Learning
# =========================
# Requires: pandas, openpyxl, matplotlib, seaborn, yellowbrick
# Install if needed:
# pip install pandas openpyxl matplotlib seaborn yellowbrick

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from yellowbrick.features import PCA
from yellowbrick.target import FeatureCorrelation
import numpy as np

# ---- 1) Loading data ----
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

weighted_rates = df.groupby(['Client_Name', 'Fiscal_Year']).apply(
    lambda x: (x['Hourly Billing Rate'] * x['Billed Hours']).sum() / x['Billed Hours'].sum() 
    if x['Billed Hours'].sum() > 0 else 0
).reset_index(name='avgListRate')

by_client_fy = by_client_fy.merge(weighted_rates, on=['Client_Name', 'Fiscal_Year'])

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

by_client_fy['avgListRate'] = by_client_fy['avgListRate'].fillna(0)
by_client_fy['billingEfficiencyPct'] = by_client_fy['billingEfficiencyPct'].fillna(0)
by_client_fy['effectiveRateVsListPct'] = by_client_fy['effectiveRateVsListPct'].fillna(0)

by_client_fy = by_client_fy.sort_values('revenue', ascending=False)

by_client_fy.columns = ['clientName', 'fiscalYear', 'projects', 'totalBillableHr', 
                        'totalBilledHr', 'revenue', 'extPrice', 'avgListRate', 
                        'billingEfficiencyPct', 'effectiveRateVsListPct', 
                        'differenceExtRev', 'differenceExtRevPercentage']

print(by_client_fy)

# ---- 4) Visualization with Yellowbrick Style ----

# Set Yellowbrick style
plt.style.use('seaborn-v0_8-darkgrid')

# 4.1 Number of Projects by Fiscal Year
projects_by_fy = by_client_fy.groupby('fiscalYear')['projects'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors: dark green for FY21 and FY23, light green for others
colors = ['#a8e6cf' if fy not in ['FY21',"FY22", 'FY23'] else '#2ecc71' 
          for fy in projects_by_fy['fiscalYear']]
edge_colors = ['#8fd9bf' if fy not in ['FY21', "FY22", 'FY23'] else '#27ae60' 
               for fy in projects_by_fy['fiscalYear']]

bars = ax.bar(projects_by_fy['fiscalYear'], projects_by_fy['projects'], 
              color=colors, edgecolor=edge_colors, linewidth=1.5)
ax.set_title('Number of Projects by Fiscal Year', fontsize=16, fontweight='bold')
ax.set_xlabel('Fiscal Year', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# 4.2 Correlation Matrix
numeric_cols_for_corr = by_client_fy[['projects', 'totalBillableHr', 'totalBilledHr', 
                                       'revenue', 'extPrice', 'avgListRate', 
                                       'billingEfficiencyPct', 'effectiveRateVsListPct', 
                                       'differenceExtRev', 'differenceExtRevPercentage']]
correlation_matrix = numeric_cols_for_corr.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 4.3 Billing Efficiency (< 95%) vs Client Name by Fiscal Year
filtered_data = by_client_fy[
    (by_client_fy['billingEfficiencyPct'].notna()) & 
    (by_client_fy['billingEfficiencyPct'] < 95) & 
    (by_client_fy['billingEfficiencyPct'] > 0)
]

fiscal_years = sorted(filtered_data['fiscalYear'].unique())

for fy in fiscal_years:
    fy_data = filtered_data[filtered_data['fiscalYear'] == fy].sort_values(
        'billingEfficiencyPct'
    )
    
    if len(fy_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Greens(fy_data['billingEfficiencyPct'] / 100)
        bars = ax.barh(fy_data['clientName'], fy_data['billingEfficiencyPct'], 
                   color=colors, edgecolor='black', linewidth=0.2)
        ax.set_title(f'Billing Efficiency (0 < % < 95) vs Client Name - {fy}', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Billing Efficiency (%)', fontsize=12)
        ax.set_ylabel('Client Name', fontsize=12)
        ax.set_xlim(0, 100)
        ax.axvline(x=80, color='#27ae60', linestyle='--', alpha=0.5, label='80% threshold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

# 4.4 Billing Efficiency (> 100%) vs Client Name for FY22
filtered_data_high = by_client_fy[
    (by_client_fy['billingEfficiencyPct'].notna()) & 
    (by_client_fy['billingEfficiencyPct'] > 100) & 
    (by_client_fy['billingEfficiencyPct'] < 600) &
    (by_client_fy['fiscalYear'] == 'FY22')
].sort_values('billingEfficiencyPct')

if len(filtered_data_high) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Greens(
        (filtered_data_high['billingEfficiencyPct'] - 100) / 500
    )
    bars = ax.barh(filtered_data_high['clientName'], 
                   filtered_data_high['billingEfficiencyPct'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Billing Efficiency (100 < % < 600) vs Client Name - FY22', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Billing Efficiency (%)', fontsize=12)
    ax.set_ylabel('Client Name', fontsize=12)
    ax.axvline(x=100, color='#27ae60', linestyle='--', alpha=0.5, label='100% baseline')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 4.5 Difference % (>0) vs Client Name by Fiscal Year
filtered_discount = by_client_fy[
    (by_client_fy['differenceExtRevPercentage'].notna()) & 
    (by_client_fy['differenceExtRevPercentage'] > 0)
]

fiscal_years = sorted(filtered_discount['fiscalYear'].unique())

for fy in fiscal_years:
    fy_data = filtered_discount[filtered_discount['fiscalYear'] == fy].sort_values(
        'differenceExtRevPercentage'
    )
    
    if len(fy_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Greens(fy_data['differenceExtRevPercentage'] / 100)
        ax.barh(fy_data['clientName'], fy_data['differenceExtRevPercentage'], 
                       color=colors, edgecolor='black', linewidth=0.2)
        ax.set_title(f'Difference % (>0) vs Client Name - {fy}', 
                           fontsize=16, fontweight='bold')
        ax.set_xlabel('Percentage Difference between Extended and Revenue Price (%)', 
                            fontsize=12)
        ax.set_ylabel('Client Name', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

# 4.6 Difference by Fiscal Year
diff_by_fy = by_client_fy.groupby('fiscalYear')['differenceExtRevPercentage'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors: dark green for FY20, light green for others
colors = ['#2ecc71' if fy == 'FY20' else '#a8e6cf' 
          for fy in diff_by_fy['fiscalYear']]
edge_colors = ['#27ae60' if fy == 'FY20' else '#8fd9bf' 
               for fy in diff_by_fy['fiscalYear']]

bars = ax.bar(diff_by_fy['fiscalYear'], diff_by_fy['differenceExtRevPercentage'], 
              color=colors, edgecolor=edge_colors, linewidth=1.5)
ax.set_title('Average Difference (Extended vs Revenue) by Fiscal Year', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Fiscal Year', fontsize=12)
ax.set_ylabel('Average Percentage Difference', fontsize=12)
ax.set_ylim(0, max(diff_by_fy['differenceExtRevPercentage']) * 1.1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# 4.7 Client Vs Revenue (Top 20)
by_client_fy_sorted = by_client_fy.sort_values('revenue', ascending=False).head(20).sort_values('revenue')
fig, ax = plt.subplots(figsize=(10, 12))
revenue_normalized = by_client_fy_sorted['revenue'] / by_client_fy_sorted['revenue'].max()
colors = plt.cm.Greens(0.3 + 0.7 * revenue_normalized)  # Range from light to dark green
bars = ax.barh(by_client_fy_sorted['clientName'], 
               by_client_fy_sorted['revenue'] / 1000, 
               color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Top 20 Clients by Revenue', fontsize=16, fontweight='bold')
ax.set_xlabel('Revenue (in K)', fontsize=12)
ax.set_ylabel('Client Name', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()