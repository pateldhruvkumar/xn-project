## xn-project — Billing/Revenue Analysis + NLP

Python and R scripts for cleaning timesheet/billing data, computing client- and fiscal-year metrics, visualizations, and text mining (TF‑IDF, clustering, topics) on `Summary Notes`.

### Overview
- Cleans the raw Excel export and derives a canonical dataset
- Computes metrics like billing efficiency, realized rates, discounts, and project counts
- Produces plots in Python (matplotlib/seaborn) and R (ggplot2)

### Repository structure
```text
dataset/                      # Excel source files and (recommended) cleaned file location
  FY19_to_FY23.xlsx           # Raw combined file used by xnFormatting.py
  FY19_to_FY23_Cleaned.xlsx   # Cleaned file (output; see notes below)
data/                         # Pre-trained embeddings (GloVe 6B)
  glove.6B.50d.txt
  glove.6B.100d.txt
  glove.6B.200d.txt
  glove.6B.300d.txt
xnFormatting.py               # Cleans raw file → writes cleaned Excel
xn.py                         # Aggregations + charts (matplotlib/seaborn + Yellowbrick)
nlp.py                        # Python NLP on Summary Notes (EDA, TF‑IDF, LDA, clustering)
nlp.R                         # R NLP on Summary Notes (tidytext, LDA, GloVe demo)
xnEdaInR.R                    # R equivalent analysis and plots
```

### Requirements
Python (3.9+ recommended)
- pandas, openpyxl, matplotlib, seaborn
- scikit-learn, yellowbrick, nltk, scipy
- Optional: qgrid (for interactive tables in notebooks)

R (4.x)
- readxl, dplyr, tidyr, ggplot2, scales, skimr, lubridate
- For NLP: tidyverse, tidytext, tokenizers, textstem, text2vec, topicmodels, broom, data.table

Install Python deps (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas openpyxl matplotlib seaborn scikit-learn yellowbrick nltk scipy qgrid
# First run of NLP will download NLTK data (punkt, stopwords, wordnet)
```

Install R packages:
```r
install.packages(c(
  "readxl","dplyr","tidyr","ggplot2","scales","skimr","lubridate",
  "tidyverse","tidytext","tokenizers","textstem","text2vec","topicmodels","broom","data.table"
))
```

### Data inputs
- Place source Excel files in `dataset/`.
- Primary raw file used by the cleaner: `dataset/FY19_to_FY23.xlsx`.

Pre-trained embeddings (optional, for NLP):
- Place GloVe 6B files in `data/` (at least `glove.6B.50d.txt`).
- Download from `https://nlp.stanford.edu/projects/glove/`.

### Usage
1) Create the cleaned dataset (from the raw file)
```powershell
python xnFormatting.py
```
Notes:
- The script currently writes `FY19_to_FY23_Cleaned.xlsx` to the project root. Move it into `dataset/` (recommended), or adjust the read paths in the analysis scripts accordingly.

2) Run the Python analyses and plots
```powershell
python xn.py       # Aggregated metrics + multiple charts (incl. Yellowbrick visuals)
```

3) Run NLP (Python)
```powershell
python nlp.py      # Text EDA, TF‑IDF, clustering (KMeans), LDA topics, GloVe demo
```

4) Run the R analyses and plots
```powershell
Rscript xnEdaInR.R
Rscript nlp.R
```

### File path configuration
- `xn.py`, `xnEdaInR.R`, `nlp.py`, and `nlp.R` currently reference an absolute path like `D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx`.
  - For portability, change these to the relative path `dataset/FY19_to_FY23_Cleaned.xlsx` (recommended), or keep the absolute path if it matches your machine.

### Outputs
- Console summaries of grouped metrics
- Multiple charts: projects by FY, billing efficiency distributions, discount percentages, client revenue bars, etc.
- NLP outputs: token frequency bars, TF‑IDF top terms, Yellowbrick K‑Elbow/Silhouette/t‑SNE visuals, LDA topic term bar charts, nearest neighbors via GloVe (if embeddings present)

### Troubleshooting
- File not found: verify the cleaned file exists and paths are correct (see File path configuration above).
- Excel engine errors: ensure `openpyxl` is installed for `.xlsx` IO.
- Plots not appearing: some environments require an interactive backend; on Windows, running `python` from PowerShell generally works as-is.
- NLTK downloads: the first run may download tokenizers/lexicons; ensure internet access or pre-download.
- GloVe embeddings: place `glove.6B.50d.txt` under `data/` or skip the embeddings section; the scripts will warn if missing.
- qgrid import errors: it is optional; the EDA will proceed without it.

### Notes
- Scripts reference academic context (ALY6080). Adjust for production use as needed.


