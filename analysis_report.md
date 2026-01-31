## Overview

This report summarizes the work done in the two R scripts:

- **`xnEdaInR.R`**: Exploratory data analysis and feature engineering on the fiscal-year billing dataset (`FY19_to_FY25_Final.xlsx`).
- **`nlp.R`**: Text mining and topic modeling on the `Summary Notes` text field from `FY19_to_FY23_Cleaned.xlsx`.

The two scripts complement each other: the first focuses on **quantitative billing and revenue metrics**, while the second focuses on **qualitative text analysis** of work descriptions.

---

## 1. `xnEdaInR.R` – Billing & Revenue Analysis

### 1.1 Purpose

- **Goal**: Perform exploratory analysis and derive core performance metrics at the **client × fiscal year** level using the main billing file `FY19_to_FY25_Final.xlsx`.
- **Key outputs**:
  - Cleaned and typed dataset.
  - Derived efficiency and pricing metrics.
  - Aggregated client–fiscal-year table (`byClientFy`).
  - Multiple visualizations for projects, billing efficiency, discounting, and revenue.

### 1.2 Data loading and cleaning

- **Data source**: `D:/Projects/xn-project/dataset/FY19_to_FY25_Final.xlsx`, Sheet 1.
- **Libraries used**: `readxl`, `dplyr`, `tidyr`, `ggplot2`, `scales`, `lubridate`.
- **Key steps**:
  - **Load Excel** into a data frame `df`.
  - **Parse `Worked Date`**:
    - Uses `lubridate::parse_date_time()` with multiple date formats (`ymd`, `mdy`, `d-b-Y`, `d-B-Y`, `d/m/Y`, `m/d/Y`) to robustly convert the `Worked Date` column from text to date.
  - **Numeric column enforcement**:
    - Identifies numeric fields: `Billable Hours`, `Billed Hours`, `Hourly Billing Rate`, `Extended Price`, `Amount Billed`.
    - Converts them to numeric with `as.numeric()` under `suppressWarnings()`.
    - Replaces `NA` values in these numeric columns with `0` using `tidyr::replace_na`.

### 1.3 Feature engineering

A new data frame `dfFeat` is created with several derived performance metrics:

- **billingEfficiencyPct**
  - **Definition**: \( 100 \times \frac{\text{Billed Hours}}{\text{Billable Hours}} \) when `Billable Hours > 0`.
  - **Interpretation**: How much of the billable time actually gets billed (billing efficiency in %).

- **revenuePerBilledHour**
  - **Definition**: \( \frac{\text{Amount Billed}}{\text{Billed Hours}} \) when `Billed Hours > 0`.
  - **Interpretation**: Realized revenue per billed hour.

- **revenuePerBillableHour**
  - **Definition**: \( \frac{\text{Amount Billed}}{\text{Billable Hours}} \) when `Billable Hours > 0`.
  - **Interpretation**: Realized revenue per billable hour (includes any inefficiency).

- **effectiveRateVsListPct**
  - **Definition**: \( 100 \times \frac{\text{revenuePerBilledHour}}{\text{Hourly Billing Rate}} \) when `Hourly Billing Rate > 0`.
  - **Interpretation**: How the realized rate compares to the list billing rate (in %).

- **differenceExtRev**
  - **Definition**: `Extended Price - Amount Billed`.
  - **Interpretation**: Absolute gap between theoretical extended price and actual billed revenue (potential discount/write-off).

- **discountPct**
  - **Definition**: \( 100 \times \frac{\text{differenceExtRev}}{\text{Extended Price}} \) when `Extended Price > 0`.
  - **Interpretation**: Percentage discount or write-off at the line level.

### 1.4 Aggregation by client and fiscal year

The script builds a summarized table `byClientFy`:

- **Grouping**:
  - `clientName` = `Client_Name`
  - `fiscalYear` = `Fiscal_Year`

- **Metrics computed per client × fiscal year**:
  - **projects**: Count of non-missing `Project Name` entries (proxy for project count).
  - **totalBillableHr**: Sum of `Billable Hours`.
  - **totalBilledHr**: Sum of `Billed Hours`.
  - **revenue**: Sum of `Amount Billed`.
  - **extPrice**: Sum of `Extended Price`.
  - **avgListRate**:
    - Weighted average hourly list rate:
    - \( \frac{\sum (\text{Hourly Billing Rate} \times \text{Billed Hours})}{\sum \text{Billed Hours}} \) when total billed hours > 0.
  - **billingEfficiencyPct** (aggregated):
    - Re-computed as \( 100 \times \frac{\text{totalBilledHr}}{\text{totalBillableHr}} \) when `totalBillableHr > 0`.
  - **effectiveRateVsListPct** (aggregated):
    - \( 100 \times \frac{\text{(revenue / totalBilledHr)}}{\text{avgListRate}} \) when `avgListRate > 0`.
  - **differenceExtRev**:
    - If `revenue == 0`, set to `0`; otherwise `extPrice - revenue`.
  - **differenceExtRevPercentage**:
    - If `revenue == 0`, set to `0`.
    - Else: \( 100 \times \frac{\text{extPrice - revenue}}{\text{extPrice}} \) when `extPrice > 0`.

- **Post-processing**:
  - Replace `NA` in `avgListRate`, `billingEfficiencyPct`, and `effectiveRateVsListPct` with `0`.
  - Sort by descending `revenue`.

### 1.5 Visualizations

The script outputs several ggplot visualizations:

- **Number of Projects by Fiscal Year**
  - **Plot**: Column chart of `projects` vs. `fiscalYear` using `byClientFy`.
  - **Insight**: Shows the distribution of project counts across fiscal years.

- **Billing Efficiency (< 95%) vs Client Name by Fiscal Year**
  - For each fiscal year:
    - Filters `byClientFy` to `0 < billingEfficiencyPct < 95`.
    - Plots horizontal bars with:
      - X-axis: `billingEfficiencyPct`.
      - Y-axis: clients ordered by efficiency.
      - Color gradient from red → yellow → light green (0–100%).
      - Vertical dashed red line at 80% as a reference threshold.
  - **Insight**: Identifies clients and years where billing efficiency is below target benchmarks.

- **Discount % (> 0) vs Client Name by Fiscal Year**
  - Filters for `differenceExtRevPercentage > 0`.
  - Plots horizontal bars faceted by `fiscalYear`.
  - **Insight**: Highlights clients and years with significant discounting or write-offs.

- **Difference (Extended vs Revenue) by Fiscal Year**
  - Column chart with:
    - X-axis: `fiscalYear`.
    - Y-axis: `differenceExtRevPercentage` (0–100% scale).
  - **Insight**: How the spread between extended price and revenue changes over time.

- **Clients by Revenue**
  - Horizontal bar chart with:
    - X-axis: `revenue` (formatted in thousands, “K”).
    - Y-axis: clients ordered by revenue.
  - **Insight**: Ranking of clients by total revenue.

---

## 2. `nlp.R` – Text Mining on Summary Notes

### 2.1 Purpose

- **Goal**: Apply modern text mining and topic modeling techniques to the `Summary Notes` field from the billing dataset (cleaned subset up to FY23), in order to understand common themes and terms used in work descriptions.
- **Data source**: `D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx`.

### 2.2 Data loading and preparation

- **Libraries used**:
  - `tidyverse`, `skimr`, `tidytext`, `tokenizers`, `textstem`,
    `text2vec`, `topicmodels`, `broom`, `ggplot2`, `readxl`, `data.table`.

- **Loading**:
  - Reads the Excel file into `raw_df`.

- **Document creation**:
  - Creates a `docs` tibble with:
    - **doc_id**: synthetic ID (`D1`, `D2`, …) using `row_number()`.
    - **text**: the `Summary Notes` field from the Excel file.
  - Filters out missing or empty text entries.

- **Initial EDA**:
  - Uses `glimpse(docs)` and `skim(docs)` to inspect structure and basic statistics.

### 2.3 Basic text EDA

- **Sentence / word / character counts**:
  - Creates `eda_counts` with:
    - `n_sent`: number of sentences per document (via `tokenizers::tokenize_sentences`).
    - `n_word`: number of words (using `str_count` and word boundaries).
    - `n_char`: number of characters (`nchar`).
  - Prints `eda_counts` for a quick profile of document lengths.

- **Top tokens (raw)**:
  - Converts text to lowercase.
  - Tokenizes to words (`unnest_tokens`).
  - Counts token frequencies and selects the top 10 tokens.
  - Plots a bar chart of the top tokens.
  - **Insight**: Shows the most frequently used words before any cleaning.

### 2.4 Preprocessing for modeling

- **Stopwords and lemmatization**:
  - Loads `tidytext::stop_words`.
  - Builds a `tokens` tibble:
    - `unnest_tokens(word, text)` to get one word per row.
    - Filters out:
      - Stopwords (e.g., “the”, “and”).
      - Tokens that do not contain alphabetic characters.
    - Applies `textstem::lemmatize_words` to convert inflected forms to their lemmas (e.g., “running” → “run”).

### 2.5 Bag-of-words and TF–IDF

- **Bag-of-words (BoW) representation**:
  - Constructs `bow` as counts of words per document: `count(doc_id, word, sort = TRUE)`.

- **TF–IDF calculation**:
  - Builds `tfidf` with `bind_tf_idf(term = word, document = doc_id, n = n)`.
  - Sorts by descending `tf_idf` to identify document-specific important words.
  - Prints, for each document, the top 5 TF–IDF terms.
  - Aggregates TF–IDF across documents and plots the top 10 terms by total TF–IDF.
  - **Insight**: Highlights distinctive terms that characterize different documents and overall themes.

### 2.6 Pre-trained word embeddings (GloVe)

- **Setup**:
  - Expects pre-trained `GloVe 6B, 50d` embeddings in `data/glove.6B.50d.txt`.
  - If the file is missing, the script stops with a clear error message.

- **Loading embeddings**:
  - Reads the embedding file with `data.table::fread`.
  - Sets row names to the word column and converts the rest to a numeric matrix `emb`.

- **Cosine similarity helper**:
  - Defines `cosine_sim(mat, term, topn = 5)`:
    - Computes cosine similarity between the embedding of `term` and all other words using `text2vec::sim2`.
    - Returns the top `topn` most similar words and similarity scores.
  - Example usage: finds nearest neighbors to the word `"email"`.
  - **Insight**: Explores semantic neighborhoods of key terms in the `Summary Notes` vocabulary.

### 2.7 Topic modeling with LDA

- **Document–term matrix (DTM)**:
  - Uses `tidytext::cast_dtm` to convert `bow` to a DTM (`dtm`) with documents (`doc_id`) and terms (`word`).

- **LDA model**:
  - Fits a Latent Dirichlet Allocation model with:
    - `k = 5` topics.
    - `control = list(seed = 42)` for reproducibility.

- **Top terms per topic**:
  - Uses `broom::tidy(lda, matrix = "beta")` to get topic–term probabilities.
  - For each topic:
    - Extracts the top 9 terms by `beta` (probability of word given topic).
  - Prints a table of top terms per topic.

- **Visualization**:
  - Creates a faceted bar chart of the top words per topic:
    - Uses `reorder_within` and `scale_x_reordered` for proper ordering within facets.
    - Each facet represents a topic, with bars showing term probabilities (`beta`).
  - **Insight**: Visual summary of the key topics/themes present in the `Summary Notes` narratives.

---

## 3. How the scripts relate

- **`xnEdaInR.R`** quantifies **how much** work is done and billed (hours, revenue, discounts, rates) across clients and years.
- **`nlp.R`** analyzes **what** is being done, by mining the narrative `Summary Notes` field for dominant words and latent topics.
- Together, these scripts support a richer analysis of:
  - Financial performance (efficiency, pricing, discounts).
  - Thematic content of work (kinds of activities, recurring themes, and areas of focus across projects and clients).


