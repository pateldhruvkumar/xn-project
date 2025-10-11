## xn-project — Billing and Revenue Analysis

Python and R scripts for cleaning timesheet/billing data, computing client- and fiscal-year metrics, and producing visualizations.

### Overview
- Cleans the raw Excel export and derives a canonical dataset
- Computes metrics like billing efficiency, realized rates, discounts, and project counts
- Produces plots in Python (matplotlib/seaborn) and R (ggplot2)

### Repository structure
```text
dataset/                      # Excel source files and (recommended) cleaned file location
  FY19_to_FY23.xlsx           # Raw combined file used by xnFormatting.py
  FY19_to_FY23_Cleaned.xlsx   # Cleaned file (output; see notes below)
xnFormatting.py               # Cleans raw file → writes cleaned Excel
xn.py                         # Aggregations + multiple matplotlib charts
xnEda.py                      # EDA: nulls, correlations, utilization %, optional qgrid
xnEdaInR.R                    # R equivalent analysis and plots
```

### Requirements
Python (3.9+ recommended)
- pandas, openpyxl, matplotlib, seaborn
- Optional: qgrid (for interactive tables in notebooks)

R (4.x)
- readxl, dplyr, tidyr, ggplot2, scales, skimr, lubridate

Install Python deps (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas openpyxl matplotlib seaborn qgrid
```

Install R packages:
```r
install.packages(c("readxl","dplyr","tidyr","ggplot2","scales","skimr","lubridate"))
```

### Data inputs
- Place source Excel files in `dataset/`.
- Primary raw file used by the cleaner: `dataset/FY19_to_FY23.xlsx`.

### Usage
1) Create the cleaned dataset (from the raw file)
```powershell
python xnFormatting.py
```
Notes:
- The script currently writes `FY19_to_FY23_Cleaned.xlsx` to the project root. Move it into `dataset/` (recommended), or adjust the read paths in the analysis scripts accordingly.

2) Run the Python analyses and plots
```powershell
python xn.py       # Aggregated metrics + multiple charts
python xnEda.py    # EDA: null counts, correlations, utilization %
```

3) Run the R analysis and plots
```powershell
Rscript xnEdaInR.R
```

### File path configuration
- `xnEda.py` reads `dataset/FY19_to_FY23_Cleaned.xlsx` (relative). This will work if you move the cleaned file into `dataset/`.
- `xn.py` and `xnEdaInR.R` currently reference an absolute path like `D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx`.
  - If your local path differs, change those to the relative path `dataset/FY19_to_FY23_Cleaned.xlsx` for portability, or keep the absolute path if it matches your machine.

### Outputs
- Console summaries of grouped metrics
- Multiple charts: projects by FY, billing efficiency distributions, discount percentages, client revenue bars, etc.

### Troubleshooting
- File not found: verify the cleaned file exists and paths are correct (see File path configuration above).
- Excel engine errors: ensure `openpyxl` is installed for `.xlsx` IO.
- Plots not appearing: some environments require an interactive backend; on Windows, running `python` from PowerShell generally works as-is.
- qgrid import errors: it is optional; the EDA will proceed without it.

### Notes
- Scripts reference academic context (ALY6080). Adjust for production use as needed.


