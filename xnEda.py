# pandas: for data manipulation and analysis
import pandas as pd

# seaborn: for statistical data visualization (heatmap)
import seaborn as sns

# matplotlib: for plotting graphs and figures
import matplotlib.pyplot as plt

# qgrid: for interactive DataFrame grid display (only in Jupyter/IPython)
try:
    import qgrid
    from IPython.display import display
    QGRID_AVAILABLE = True
except Exception:
    QGRID_AVAILABLE = False

# Load your Excel file
df = pd.read_excel("dataset/FY19_to_FY23_Cleaned.xlsx")

# Count and print null values for each column
nullCounts = df.isnull().sum()
print("\nNull value counts by column:")
print(nullCounts)

# Create missing flags before filling
df["Billable_Hours_Missing"] = df["Billable Hours"].isna()
df["Billed_Hours_Missing"] = df["Billed Hours"].isna()

# Fill nulls with 0
df["Billable Hours"] = df["Billable Hours"].fillna(0)
df["Billed Hours"] = df["Billed Hours"].fillna(0)
df["Amount Billed"] = df["Amount Billed"].fillna(0)

# Add 'Bill_Num' column: 1 if 'Bill' is 'Billable', 0 if 'Non Billable'
df["Bill_Num"] = df["Bill"].apply(lambda x: 1 if x == "Billable" else 0)

# Print only the relevant columns to verify
print(df[["Billable Hours", "Billable_Hours_Missing", 
          "Billed Hours", "Billed_Hours_Missing"]].head(20))

# Count and print null values for each column
nullCounts = df.isnull().sum()
print("\nNull value counts by column:")
print(nullCounts)

# Plot heatmap of correlation between selected columns
corr = df[["Billable Hours", "Billed Hours", "Amount Billed"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Billable Hours, Billed Hours, Amount Billed")
plt.show()

# Print the first few rows of the DataFrame
print("\nDataFrame head:")
print(df.head())

# Compute Utilization % by Client Name and Fiscal_Year
# Sum hours per group, then compute utilization as billed / billable
util_group = df.groupby(["Client_Name", "Fiscal_Year"], dropna=False).agg({
    "Billed Hours": "sum",
    "Billable Hours": "sum"
}).reset_index()

# Avoid divide-by-zero; if Billable Hours is 0, utilization is 0
util_group["Utilization"] = util_group.apply(
    lambda r: (r["Billed Hours"] / r["Billable Hours"]) * 100 if r["Billable Hours"] else 0,
    axis=1
)

# Sort and prepare display
util_group = util_group.sort_values(["Fiscal_Year", "Client_Name"]).reset_index(drop=True)
util_display = util_group[["Client_Name", "Fiscal_Year", "Billable Hours", "Billed Hours", "Utilization"]]

# Always display CSV representation for utilization
csv_output = util_display.to_csv(index=False, float_format="%.2f")
print("\nUtilization % by Client Name and Fiscal_Year (CSV):")
print(csv_output.strip())

# Optionally show interactive grid if available
if QGRID_AVAILABLE:
    print("\nInteractive Utilization table (qgrid):")
    grid = qgrid.show_grid(util_display, show_toolbar=True)
    display(grid)
else:
    print("\nqgrid not available in this environment.")