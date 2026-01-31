## xn-project — Billing/Revenue Analysis + NLP

Python and R scripts for cleaning timesheet/billing data, computing client- and fiscal-year metrics, visualizations, and text mining (TF‑IDF, clustering, topics) on `Summary Notes`.

### Overview

- Cleans the raw Excel export and derives a canonical dataset
- Computes metrics like billing efficiency, realized rates, differences (extended vs revenue), and project counts
- Produces comprehensive visualizations in Python (matplotlib/seaborn/Yellowbrick) with consistent green color theming
- Generates 15+ independent graphs for fiscal year comparisons and client-level analysis
- Performs text mining and NLP on project summary notes
- Provides R-based analytics as an alternative implementation

### Repository structure

```text
dataset/                      # Excel source files and (recommended) cleaned file location
  FY19_to_FY25_Final.xlsx     # Primary dataset used by analysis scripts
  FY19_to_FY23.xlsx           # Historical raw file
data/                         # Pre-trained embeddings (GloVe 6B)
  glove.6B.50d.txt
  glove.6B.100d.txt
  glove.6B.200d.txt
  glove.6B.300d.txt
xnFormatting.py               # Cleans raw file → writes cleaned Excel
xn.py                         # Aggregations + charts (matplotlib/seaborn + Yellowbrick)
nlp.py                        # Python NLP on Summary Notes (EDA, TF‑IDF, K-Means Clustering, LDA Topic Modeling)
nlp.R                         # R NLP on Summary Notes (tidytext, LDA, GloVe demo)
xnEdaInR.R                    # R equivalent analysis and plots (Billing Efficiency, Revenue, Discount %)
openrouter_invoice_descriptions.py # Script to generate invoice descriptions using OpenRouter API
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
pip install pandas openpyxl matplotlib seaborn scikit-learn yellowbrick nltk scipy qgrid requests tqdm
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
- **Primary file used by analysis scripts:** `dataset/FY19_to_FY25_Final.xlsx`.

Pre-trained embeddings (optional, for NLP):

- Place GloVe 6B files in `data/` (at least `glove.6B.50d.txt`).
- Download from `https://nlp.stanford.edu/projects/glove/`.

### Usage

1. **Create/Clean Dataset** (if starting from raw files)

```powershell
python xnFormatting.py
```

2. **Run Python Analyses and Plots**

```powershell
python xn.py       # Generates 15+ visualizations including correlation matrix,
                   # fiscal year comparisons, billing efficiency analysis,
                   # and top client revenue charts
```

3. **Run NLP (Python)**

```powershell
python nlp.py
```
*   **EDA:** Token counts, top tokens visualization.
*   **Preprocessing:** Tokenization, stopword removal, lemmatization.
*   **Encoding:** Bag of Words and TF-IDF analysis.
*   **Clustering:** K-Means clustering with Elbow method and Silhouette plots to find optimal groups.
*   **Topic Modeling:** Latent Dirichlet Allocation (LDA) to identify 7 key topics in summary notes.

4. **Run R Analyses and Plots**

```powershell
Rscript xnEdaInR.R
```
*   Generates `ggplot2` visualizations for:
    *   Number of Projects by Fiscal Year.
    *   Billing Efficiency (< 95%) vs Client Name.
    *   Discount % (>0) vs Client Name by Fiscal Year.
    *   Clients by Revenue.

```powershell
Rscript nlp.R
```

5. **Generate Invoice Descriptions (OpenRouter)**

```powershell
# Set up .env with OPENROUTER_API_KEY
python openrouter_invoice_descriptions.py
```

### File path configuration

- Scripts (`xn.py`, `xnEdaInR.R`, `nlp.py`) currently reference the absolute path `D:/Projects/xn-project/dataset/FY19_to_FY25_Final.xlsx`.
- Adjust this path in the scripts if your environment differs.

### Visualization Features

- **Consistent Color Theme**: All graphs (except correlation matrix) use green color gradients
- **Independent Graphs**: Each fiscal year's billing efficiency and difference analysis displayed in separate windows
- **Interactive Viewing**: Graphs appear sequentially, allowing detailed examination of each visualization
- **Highlight Key Metrics**: Important fiscal years (FY20, FY21, FY22, FY23) highlighted in dark green vs light green for others
- **Comprehensive Metrics**: 7 distinct visualization types covering project counts, correlations, efficiency, differences, and revenue

### Notes

- Scripts reference academic context (ALY6080). Adjust for production use as needed.
- All visualizations use a professional green color scheme for consistency and clarity
- Graph outputs include threshold lines and baseline references for contextual analysis
