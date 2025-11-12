# =========================
# ALY6080 - Experiential Learning (Python Rewrite)
# =========================
# Requires: pandas, openpyxl, matplotlib, seaborn, numpy
# Optional: yellowbrick (not required here)
# Install if needed:
#   pip install pandas openpyxl matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -----------------------------
# 0) SETTINGS
# -----------------------------
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

# -----------------------------
# 1) LOAD DATA
# -----------------------------
file_path = "D:/Projects/xn-project/dataset/FY19_to_FY25_Final.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# parse Worked Date if present
if "Worked Date" in df.columns:
    df["Worked Date"] = pd.to_datetime(df["Worked Date"], errors="coerce")

# ensure key numeric columns exist and are numeric
num_candidates = ["Billable Hours", "Billed Hours", "Hourly Billing Rate",
                  "Extended Price", "Amount Billed"]
for col in num_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        # create if missing (prevents KeyErrors later)
        df[col] = np.nan

# -----------------------------
# 2) COLUMN PICKER (ROBUST NAMES)
# -----------------------------
def pick(colnames, candidates, required=True):
    """Return the first candidate present in colnames."""
    for c in candidates:
        if c in colnames:
            return c
    if required:
        raise KeyError(f"None of the candidate names found: {candidates}")
    return None

COL_CLIENT = pick(df.columns, ["Client_Name", "Client Name", "clientName"])
COL_FY     = pick(df.columns, ["Fiscal_Year", "Fiscal Year", "fiscalYear"])
COL_PROJ   = pick(df.columns, ["Project Name", "Project_Name", "projectName"], required=False)

# -----------------------------
# 3) ROW-LEVEL FEATURES
# -----------------------------
# replace impossible 0 denominators with NaN; compute safely with np.where
df["billingEfficiencyPct"] = np.where(
    df["Billable Hours"] > 0,
    100 * (df["Billed Hours"] / df["Billable Hours"]),
    np.nan
)

df["revenuePerBilledHour"] = np.where(
    df["Billed Hours"] > 0,
    df["Amount Billed"] / df["Billed Hours"],
    np.nan
)

df["revenuePerBillableHour"] = np.where(
    df["Billable Hours"] > 0,
    df["Amount Billed"] / df["Billable Hours"],
    np.nan
)

df["effectiveRateVsListPct"] = np.where(
    (df["Hourly Billing Rate"] > 0) & pd.notna(df["revenuePerBilledHour"]),
    100 * (df["revenuePerBilledHour"] / df["Hourly Billing Rate"]),
    np.nan
)

df["differenceExtRev"] = df["Extended Price"] - df["Amount Billed"]

df["discountPct"] = np.where(
    df["Extended Price"] > 0,
    100 * (df["differenceExtRev"] / df["Extended Price"]),
    np.nan
)

# -----------------------------
# 4) GROUPED FEATURES (CLIENT x FISCAL YEAR)
# -----------------------------
def summarize_group(g: pd.DataFrame) -> pd.Series:
    projects_total = len(g)
    if COL_PROJ is not None and COL_PROJ in g.columns:
        projects_with_name = g[COL_PROJ].notna().sum()
        projects_distinct  = g[COL_PROJ].nunique(dropna=True)
    else:
        projects_with_name = np.nan
        projects_distinct  = np.nan

    totalBillableHr = g["Billable Hours"].sum(skipna=True)
    totalBilledHr   = g["Billed Hours"].sum(skipna=True)
    revenue         = g["Amount Billed"].sum(skipna=True)
    extPrice        = g["Extended Price"].sum(skipna=True)

    if totalBilledHr and totalBilledHr > 0:
        avgListRate = (g["Hourly Billing Rate"].fillna(0) * g["Billed Hours"].fillna(0)).sum() / totalBilledHr
    else:
        avgListRate = np.nan

    return pd.Series({
        "projects_total": projects_total,
        "projects_with_name": projects_with_name,
        "projects_distinct": projects_distinct,
        "totalBillableHr": totalBillableHr,
        "totalBilledHr": totalBilledHr,
        "revenue": revenue,
        "extPrice": extPrice,
        "avgListRate": avgListRate,
    })

by_client_fy = (
    df.groupby([COL_CLIENT, COL_FY], dropna=False)
      .apply(summarize_group)
      .reset_index()
      .rename(columns={COL_CLIENT: "clientName", COL_FY: "fiscalYear"})
)

# Derived grouped metrics
by_client_fy["billingEfficiencyPct"] = np.where(
    by_client_fy["totalBillableHr"] > 0,
    100 * (by_client_fy["totalBilledHr"] / by_client_fy["totalBillableHr"]),
    np.nan
)

by_client_fy["effectiveRateVsListPct"] = np.where(
    (by_client_fy["avgListRate"] > 0) & (by_client_fy["totalBilledHr"] > 0),
    100 * ((by_client_fy["revenue"] / by_client_fy["totalBilledHr"]) / by_client_fy["avgListRate"]),
    np.nan
)

by_client_fy["differenceExtRev"] = by_client_fy["extPrice"] - by_client_fy["revenue"]

by_client_fy["differenceExtRevPercentage"] = np.where(
    by_client_fy["extPrice"] > 0,
    100 * (by_client_fy["differenceExtRev"] / by_client_fy["extPrice"]),
    np.nan
)

# Fill only for display (do not overwrite business-significant NaNs elsewhere)
by_client_fy[["avgListRate", "billingEfficiencyPct", "effectiveRateVsListPct"]] = \
    by_client_fy[["avgListRate", "billingEfficiencyPct", "effectiveRateVsListPct"]].fillna(0)

# sort for downstream visuals
by_client_fy = by_client_fy.sort_values("revenue", ascending=False).reset_index(drop=True)

print("Sample of aggregated data:")
print(by_client_fy.head(10))

# -----------------------------
# 5) VISUALIZATION
# -----------------------------
plt.style.use("seaborn-v0_8-darkgrid")

# 5.1 Total Number of Projects by Fiscal Year
projects_by_fy = (
    by_client_fy.groupby("fiscalYear", dropna=False)["projects_total"]
    .sum()
    .reset_index(name="total_projects")
)
fig, ax = plt.subplots(figsize=(10, 6))
years = projects_by_fy["fiscalYear"].astype(str)
colors = ["#a8e6cf" if fy not in ["FY21", "FY24", "FY25"] else "#2ecc71" for fy in years]
edge_colors = ["#8fd9bf" if fy not in ["FY21", "FY22", "FY23"] else "#27ae60" for fy in years]
bars = ax.bar(years, projects_by_fy["total_projects"], color=colors, edgecolor=edge_colors, linewidth=1.5)
ax.set_title("Total Number of Projects by Fiscal Year", fontsize=16, fontweight="bold")
ax.set_xlabel("Fiscal Year", fontsize=12)
ax.set_ylabel("Total Number of Projects", fontsize=12)
ax.grid(axis="y", alpha=0.3)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{int(b.get_height())}",
            ha="center", va="bottom", fontsize=10)
plt.tight_layout(); plt.show()

# 5.2 Correlation Matrix of Key Metrics
corr_cols = [
    "projects_total", "projects_with_name", "projects_distinct",
    "totalBillableHr", "totalBilledHr", "revenue", "extPrice",
    "avgListRate", "billingEfficiencyPct", "effectiveRateVsListPct",
    "differenceExtRev", "differenceExtRevPercentage"
]
numeric_cols_for_corr = by_client_fy[[c for c in corr_cols if c in by_client_fy.columns]].select_dtypes(include=[np.number])
if numeric_cols_for_corr.shape[1] >= 2:
    corr = numeric_cols_for_corr.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix of Key Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("Not enough numeric columns to compute correlation matrix.")

# 5.3 Billing Efficiency (0 < % < 95) vs Client Name by Fiscal Year
filtered_eff_low = by_client_fy[(by_client_fy["billingEfficiencyPct"] > 0) & (by_client_fy["billingEfficiencyPct"] < 95)]
for fy in sorted(filtered_eff_low["fiscalYear"].dropna().astype(str).unique()):
    fy_data = filtered_eff_low[filtered_eff_low["fiscalYear"] == fy].sort_values("billingEfficiencyPct")
    if not fy_data.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Greens(np.clip(fy_data["billingEfficiencyPct"] / 100.0, 0, 1))
        ax.barh(fy_data["clientName"].astype(str), fy_data["billingEfficiencyPct"],
                color=colors, edgecolor="black", linewidth=0.2)
        ax.set_title(f"Billing Efficiency (0 < % < 95) vs Client Name - {fy}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Billing Efficiency (%)", fontsize=12)
        ax.set_ylabel("Client Name", fontsize=12)
        ax.set_xlim(0, 100)
        ax.axvline(x=80, color="#27ae60", linestyle="--", alpha=0.5, label="80% threshold")
        ax.legend(); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

# 5.4 Billing Efficiency (100 < % < 600) vs Client Name - FY22
mask_fy22 = (by_client_fy["fiscalYear"].astype(str) == "FY22")
filtered_eff_high = by_client_fy[mask_fy22 & (by_client_fy["billingEfficiencyPct"] > 100) & (by_client_fy["billingEfficiencyPct"] < 600)]
filtered_eff_high = filtered_eff_high.sort_values("billingEfficiencyPct")
if not filtered_eff_high.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Greens(np.clip((filtered_eff_high["billingEfficiencyPct"] - 100) / 500.0, 0, 1))
    ax.barh(filtered_eff_high["clientName"].astype(str), filtered_eff_high["billingEfficiencyPct"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Billing Efficiency (100 < % < 600) vs Client Name - FY22", fontsize=16, fontweight="bold")
    ax.set_xlabel("Billing Efficiency (%)", fontsize=12)
    ax.set_ylabel("Client Name", fontsize=12)
    ax.axvline(x=100, color="#27ae60", linestyle="--", alpha=0.5, label="100% baseline")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()

# 5.5 Difference % (>0) vs Client Name by Fiscal Year
filtered_discount = by_client_fy[by_client_fy["differenceExtRevPercentage"].fillna(0) > 0]
for fy in sorted(filtered_discount["fiscalYear"].dropna().astype(str).unique()):
    fy_data = filtered_discount[filtered_discount["fiscalYear"] == fy].sort_values("differenceExtRevPercentage")
    if not fy_data.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Greens(np.clip(fy_data["differenceExtRevPercentage"] / 100.0, 0, 1))
        ax.barh(fy_data["clientName"].astype(str), fy_data["differenceExtRevPercentage"],
                color=colors, edgecolor="black", linewidth=0.2)
        ax.set_title(f"Difference % (>0) vs Client Name - {fy}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Percentage Difference between Extended and Revenue Price (%)", fontsize=12)
        ax.set_ylabel("Client Name", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

# 5.6 Average Difference (Extended vs Revenue) by Fiscal Year
diff_by_fy = (
    by_client_fy.groupby("fiscalYear", dropna=False)["differenceExtRevPercentage"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 6))
years = diff_by_fy["fiscalYear"].astype(str)
colors = ["#2ecc71" if fy == "FY20" else "#a8e6cf" for fy in years]
edge_colors = ["#27ae60" if fy == "FY20" else "#8fd9bf" for fy in years]
bars = ax.bar(years, diff_by_fy["differenceExtRevPercentage"], color=colors, edgecolor=edge_colors, linewidth=1.5)
ax.set_title("Average Difference (Extended vs Revenue) by Fiscal Year", fontsize=16, fontweight="bold")
ax.set_xlabel("Fiscal Year", fontsize=12)
ax.set_ylabel("Average Percentage Difference", fontsize=12)
ax.set_ylim(0, max(0.0, diff_by_fy["differenceExtRevPercentage"].max()) * 1.1)
ax.grid(axis="y", alpha=0.3)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout(); plt.show()

# 5.7 Top 20 Clients by Revenue
top20 = by_client_fy.nlargest(20, "revenue").sort_values("revenue")
fig, ax = plt.subplots(figsize=(10, 12))
revenue_normalized = top20["revenue"] / top20["revenue"].max() if top20["revenue"].max() not in [0, np.nan] else 0
colors = plt.cm.Greens(0.3 + 0.7 * np.clip(revenue_normalized, 0, 1))
ax.barh(top20["clientName"].astype(str), top20["revenue"] / 1000,
        color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Top 20 Clients by Revenue", fontsize=16, fontweight="bold")
ax.set_xlabel("Revenue (in K)", fontsize=12)
ax.set_ylabel("Client Name", fontsize=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.show()