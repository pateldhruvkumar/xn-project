###############################################
## Text Mining in R
## Dataset: FY19_to_FY23_Cleaned.xlsx (Summary Notes column)
###############################################

# -------------------------
# 0) PACKAGES & SETUP
# -------------------------
# install.packages(c("tidyverse","skimr","tidytext","tokenizers","textstem","text2vec","topicmodels","ggplot2","readxl","data.table"))
library(tidyverse)   # wrangling + ggplot2
library(skimr)       # skim() quick EDA
library(tidytext)    # tidy NLP (unnest_tokens, stop_words, tf-idf)
library(tokenizers)  # sentence/word tokenizers
library(textstem)    # lemmatize
library(text2vec)    # BoW/TF-IDF, GloVe
library(topicmodels) # LDA
library(broom)       # tidy
library(ggplot2)     # plotting
library(readxl)      # <-- added to read Excel
library(data.table)  # fread() for GloVe
theme_set(theme_minimal(base_size = 12))
set.seed(123)

# -------------------------
# 1) LOAD DATA  (CHANGED)
# -------------------------
# Reads your Excel, keeps only a doc id and the "Summary Notes" text for NLP.
# doc_id is a simple row-based id to keep the rest of the script identical.
raw_df <- readxl::read_excel("D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx")

docs <- raw_df %>%
  transmute(
    doc_id = paste0("D", row_number()),
    text   = `Summary Notes`
  ) %>%
  filter(!is.na(text), nzchar(trimws(text)))

glimpse(docs)
skim(docs)

# -------------------------
# 2) EDA (tiny text EDA)
# -------------------------
eda_counts <- docs %>%
  mutate(
    n_sent = lengths(tokenizers::tokenize_sentences(text)),
    n_word = str_count(text, boundary("word")),
    n_char = nchar(text)
  )
print(eda_counts)

# Top tokens 
top_tokens <- docs %>%
  mutate(text = str_to_lower(text)) %>%
  unnest_tokens(token, text, token = "words") %>%
  count(token, sort = TRUE) %>%
  slice_head(n = 10)
print(top_tokens)

ggplot(top_tokens, aes(reorder(token, n), n)) +
  geom_col() + coord_flip() +
  labs(title = "Top tokens (raw, lowercased)", x = NULL, y = "Count")

# -------------------------
# 3) PREPROCESSING (tokenize → stopwords → lemmatize)
# -------------------------
data(stop_words)

tokens <- docs %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word, str_detect(word, "[a-z]")) %>%
  mutate(word = textstem::lemmatize_words(word))

# -------------------------
# 4) ENCODING (BoW & TF-IDF)
# -------------------------
bow <- tokens %>%
  count(doc_id, word, sort = TRUE)

tfidf <- bow %>%
  bind_tf_idf(term = word, document = doc_id, n = n) %>%
  arrange(desc(tf_idf))

cat("\nTop TF-IDF terms per document:\n")
print(tfidf %>% group_by(doc_id) %>% slice_max(tf_idf, n = 5) %>% ungroup())

ggplot(
  tfidf %>% group_by(word) %>% summarise(tf_idf = sum(tf_idf)) %>% slice_max(tf_idf, n = 10),
  aes(reorder(word, tf_idf), tf_idf)
) +
  geom_col() + coord_flip() +
  labs(title = "Top terms by total TF-IDF", x = NULL, y = "TF-IDF")

# -------------------------
# 5) PRE-TRAINED WORD EMBEDDINGS (GloVe 6B, 50d) — manual setup version
# -------------------------
# (unchanged, but will stop with a clear message if file not present)
emb_path <- file.path("data", "glove.6B.50d.txt")
if (!file.exists(emb_path)) {
  stop("Missing 'data/glove.6B.50d.txt'. Please download GloVe 6B and place the 50d file in the 'data/' folder (see comments above).")
}

glove_df <- fread(emb_path, header = FALSE, quote = "", data.table = FALSE)
rownames(glove_df) <- glove_df[[1]]
glove_df[[1]] <- NULL
emb <- as.matrix(glove_df)
rm(glove_df); invisible(gc())

cat("Embeddings loaded: ", nrow(emb), " words × ", ncol(emb), " dims\n", sep = "")

cosine_sim <- function(mat, term, topn = 5) {
  q <- tolower(term)
  if (!q %in% rownames(mat)) return(tibble(term = character(), sim = numeric()))
  sims <- sim2(mat, mat[q, , drop = FALSE], method = "cosine", norm = "l2")[, 1]
  tibble(term = names(sims), sim = as.numeric(sims)) |>
    arrange(desc(sim)) |>
    filter(term != q) |>
    slice_head(n = topn)
}

cat("\nNearest neighbors (pre-trained GloVe 50d):\n")
print(cosine_sim(emb, "email", 5))

# -------------------------
# 6) TOPIC MODELING (LDA, k = 4)
# -------------------------
dtm <- bow %>% tidytext::cast_dtm(document = doc_id, term = word, value = n)
as.matrix(dtm[1:2, 1:3]) # preview a subset of dtm

set.seed(42)
lda <- topicmodels::LDA(dtm, k = 4, control = list(seed = 42))

terms_per_topic <- broom::tidy(lda, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  arrange(topic, desc(beta)) %>%
  ungroup()

cat("\nTop terms per topic (LDA k=2):\n")
print(terms_per_topic)

ggplot(
  terms_per_topic %>%
    mutate(term = tidytext::reorder_within(term, beta, topic)),
  aes(x = term, y = beta, fill = factor(topic))
) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  tidytext::scale_x_reordered() +
  facet_wrap(~ topic, scales = "free_y") +
  labs(
    title = "Top words per topic (LDA) Latent Dirichlet Allocation",
    x = NULL,
    y = expression(beta ~ "=" ~ P(word ~ "|" ~ topic))
  ) +
  theme_minimal(base_size = 12)

#............... End of Script ......................... 
