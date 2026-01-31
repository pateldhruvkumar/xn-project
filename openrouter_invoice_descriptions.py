"""
Batch-generate invoice descriptions from Excel using OpenRouter.

Reads rows from `dataset/FY19_to_FY25_Final.xlsx`, sends a prompt to
`https://openrouter.ai/api/v1/chat/completions` for each one, and writes
the generated description back to a new Excel file.

Usage (PowerShell, from project root):

    # 1) Activate your venv if you use one
    # .\.venv\Scripts\Activate.ps1
    #
    # 2) Install deps if needed
    # pip install pandas openpyxl requests
    #
    # 3) Provide your OpenRouter API key (DO NOT commit this)
    #    Option A: .env file in project root:
    #       OPENROUTER_API_KEY=sk-or-...
    #    Option B: environment variable in the shell:
    #       $env:OPENROUTER_API_KEY = "sk-or-..."
    #
    # 4) Run
    # python openrouter_invoice_descriptions.py
"""

import os
import time
import argparse
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


# -----------------------------
# CONFIGURATION
# -----------------------------

# Input / output Excel files
INPUT_EXCEL = "dataset/FY19_to_FY25_Final.xlsx"
# Script will resume from this file if it already exists
OUTPUT_EXCEL = "dataset/FY19_to_FY25_Final_with_descriptions.xlsx"

# Additional metadata file (for categories, etc.)
ADDITIONAL_EXCEL = "dataset/additional.xlsx"

# Toggle whether to merge category info from additional.xlsx.
# The user requested to ignore additional.xlsx for now, so this is False by default.
USE_ADDITIONAL = False

# Column names in the main input file
HOURS_COLUMN = "Billable Hours"   # adjust to "Billed Hours" or another column if needed
SUMMARY_COLUMN = "Summary Notes"
OUTPUT_COLUMN = "Invoice Description"

# Category configuration
# - CATEGORY_OUTPUT_COLUMN: name of the category column we expect in the working DataFrame
# - CATEGORY_ADDITIONAL_COLUMN: name of the category column inside additional.xlsx
# - MERGE_KEY_COLUMN: optional explicit join key. If None, a key will be auto-detected.
CATEGORY_OUTPUT_COLUMN = "Category"
CATEGORY_ADDITIONAL_COLUMN = "Category"
MERGE_KEY_COLUMN: Optional[str] = None  # e.g. "Billing Code Name" or "Project Name"

# OpenRouter configuration
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Choose the model you want to use on OpenRouter
# Example models: "openai/gpt-4o-mini", "openai/gpt-4o", etc.
MODEL_ID = "openai/gpt-oss-20b"

# Rate limiting / retry behavior
MAX_RETRIES = 50
INITIAL_BACKOFF_SECONDS = 1.0  # base backoff for 429s (will be multiplied)

# Per-row retry behavior (on top of MAX_RETRIES inside call_openrouter_api)
MAX_ROW_ATTEMPTS = 30
ROW_RETRY_DELAY_SECONDS = 10.0  # delay between row-level attempts when a row fails

# Concurrency / checkpointing
MAX_CONCURRENT_REQUESTS = 100  # maximum in-flight requests at a time (reduced for stability)
SAVE_EVERY_N = 500  # save Excel to disk after this many rows complete
REQUEST_DELAY_SECONDS = 0.1  # delay between starting new requests to avoid overwhelming API


logger = logging.getLogger("openrouter_invoice_descriptions")

# Global semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)


def build_user_prompt(billing_hours: float, summary_notes: str, category: Optional[str]) -> str:
    """
    Build the user message content using the pattern you provided,
    but with dynamic billing hours and summary notes.
    """
    return f"""**Role:**
You are a helpful assistant writing an invoice description.

**Task:**
Write **one detailed sentence** explaining why the task described below took the specific amount of billing hours.

**Instructions:**
1. Expand the "Summary Notes" into full, clear phrases (e.g., change "Arrange Privs" to "setting up user access").
2. Connect the tasks together logically to show why they required this much time.
3. Keep the language professional but easy to understand.

**Input Data:**
* **Billing Hours:** {billing_hours}
* **Summary Notes:** {summary_notes}
{"* **Category:** " + category if category else ""}

**Desired Output:**
"This task required {billing_hours} hours to complete because [explain the steps clearly so the client understands the effort]."
"""


def load_env_from_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader: populate os.environ from KEY=VALUE pairs in a file.
    Existing environment variables are not overwritten.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("Failed to read .env file at %s: %s", path, exc)


def call_openrouter_api(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call OpenRouter Chat Completions API with basic 429-aware retry logic.

    Returns the assistant message content on success, or None if all retries fail.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        # You can optionally customize these for ranking/attribution on OpenRouter:
        # "HTTP-Referer": "https://your-site.example.com",
        # "X-Title": "Your App Name",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=60)
        except requests.RequestException as exc:
            # Network-level error; backoff and retry
            logger.warning("Request exception on attempt %s/%s: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                logger.error("Unexpected response format from OpenRouter: %s", data)
                return None

        if resp.status_code == 429:
            # Rate limit exceeded; check Retry-After header if present
            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = backoff
            else:
                delay = backoff

            logger.warning(
                "429 rate limit received. Sleeping for %.1f seconds (attempt %s/%s)",
                delay,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                return None
            time.sleep(delay)
            # Exponential backoff for next time
            backoff *= 2
            continue

        if 500 <= resp.status_code < 600:
            # Server errors: retry with backoff
            logger.error(
                "Server error %s from OpenRouter: %s...",
                resp.status_code,
                resp.text[:200],
            )
            if attempt == max_retries:
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        # Other non-success status codes: do not retry by default
        logger.error(
            "Non-success status code %s from OpenRouter: %s...",
            resp.status_code,
            resp.text,
        )
        return None

    return None


def process_row(
    idx: int,
    hours: float,
    summary: str,
    category: Optional[str],
    api_key: str,
) -> Tuple[int, Optional[str]]:
    """
    Worker wrapper for a single row.

    - Builds the prompt
    - Calls the API with internal retries (call_openrouter_api)
    - If that still fails (returns None), waits and retries up to MAX_ROW_ATTEMPTS.
    - Uses semaphore to limit concurrent API calls

    Returns (row_index, content_or_none).
    """
    prompt = build_user_prompt(float(hours), str(summary), category)
    messages = [{"role": "user", "content": prompt}]

    attempt = 0
    while attempt < MAX_ROW_ATTEMPTS:
        attempt += 1
        
        # Acquire semaphore before making API call
        with api_semaphore:
            logger.debug(
                "Row %s: OpenRouter attempt %s/%s (semaphore acquired)",
                idx,
                attempt,
                MAX_ROW_ATTEMPTS,
            )
            content = call_openrouter_api(api_key, MODEL_ID, messages)
            
            # Small delay after each API call to avoid overwhelming the server
            time.sleep(REQUEST_DELAY_SECONDS)
        
        if content is not None:
            if not content.strip():
                logger.warning("Row %s: received empty content from API. Retrying...", idx)
                # treat empty string as failure if you want to retry
            else:
                logger.info("Row %s: successfully received description", idx)
                return idx, content

        if attempt < MAX_ROW_ATTEMPTS:
            logger.warning(
                "Row %s: attempt %s failed, waiting %.1f seconds before retry...",
                idx,
                attempt,
                ROW_RETRY_DELAY_SECONDS,
            )
            time.sleep(ROW_RETRY_DELAY_SECONDS)

    logger.error("Row %s: all %s attempts failed; giving up.", idx, MAX_ROW_ATTEMPTS)
    return idx, None


def enrich_with_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a Category column to df using dataset/additional.xlsx if available.
    - If CATEGORY_OUTPUT_COLUMN already exists in df, it is left as-is.
    - Otherwise, attempts to merge from ADDITIONAL_EXCEL using a join key.
    """
    if CATEGORY_OUTPUT_COLUMN in df.columns:
        # Already present (e.g., when resuming) – nothing to do
        return df

    if not os.path.exists(ADDITIONAL_EXCEL):
        print(
            f"[WARN] Additional workbook not found at {ADDITIONAL_EXCEL}. "
            f"No category information will be added."
        )
        return df

    add_df = pd.read_excel(ADDITIONAL_EXCEL)

    if CATEGORY_ADDITIONAL_COLUMN not in add_df.columns:
        raise KeyError(
            f"Category column '{CATEGORY_ADDITIONAL_COLUMN}' not found in {ADDITIONAL_EXCEL}. "
            f"Available columns: {list(add_df.columns)}"
        )

    # Determine join key
    if MERGE_KEY_COLUMN is not None:
        key = MERGE_KEY_COLUMN
        if key not in df.columns:
            raise KeyError(
                f"Configured MERGE_KEY_COLUMN '{key}' not found in main Excel. "
                f"Available columns: {list(df.columns)}"
            )
        if key not in add_df.columns:
            raise KeyError(
                f"Configured MERGE_KEY_COLUMN '{key}' not found in {ADDITIONAL_EXCEL}. "
                f"Available columns: {list(add_df.columns)}"
            )
    else:
        preferred_keys = [
            "Billing Code Name",
            "Billing Code",
            "Project Name",
            "Task or Ticket Title",
        ]
        common = [c for c in preferred_keys if c in df.columns and c in add_df.columns]
        if not common:
            raise KeyError(
                "Could not auto-detect a merge key between main Excel and additional.xlsx.\n"
                f"Common columns: {set(df.columns) & set(add_df.columns)}\n"
                "Please set MERGE_KEY_COLUMN at the top of openrouter_invoice_descriptions.py "
                "to a column that exists in both files (for example 'Billing Code Name')."
            )
        key = common[0]
        print(f"[INFO] Using '{key}' as merge key between main and additional workbooks.")

    add_small = (
        add_df[[key, CATEGORY_ADDITIONAL_COLUMN]]
        .dropna(subset=[key])
        .drop_duplicates(subset=[key])
    )

    merged = df.merge(
        add_small,
        on=key,
        how="left",
        suffixes=("", "_from_additional"),
    )

    # Ensure the output column name is consistent
    if CATEGORY_ADDITIONAL_COLUMN != CATEGORY_OUTPUT_COLUMN:
        merged.rename(
            columns={CATEGORY_ADDITIONAL_COLUMN: CATEGORY_OUTPUT_COLUMN},
            inplace=True,
        )

    return merged


def main(max_samples: Optional[int] = None) -> None:
    # Load variables from .env first (if present), then read from environment
    load_env_from_dotenv()
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set.\n"
            f"Provide it either in a .env file at the project root:\n"
            f"  {OPENROUTER_API_KEY_ENV}=sk-or-...\n"
            f"or as an environment variable in the shell, e.g. PowerShell:\n"
            f'  $env:{OPENROUTER_API_KEY_ENV} = "sk-or-..."'
        )

    # Always load the full input workbook as the master dataset
    if not os.path.exists(INPUT_EXCEL):
        raise FileNotFoundError(
            f"Input Excel file not found at: {INPUT_EXCEL}"
        )

    df = pd.read_excel(INPUT_EXCEL)
    logger.info("Loaded input workbook: %s (rows: %s)", INPUT_EXCEL, len(df))

    # If an output workbook already exists, merge any previously generated
    # descriptions back onto the master dataset so we can truly resume even
    # when the output file only contains processed rows (e.g., in limited mode).
    if os.path.exists(OUTPUT_EXCEL):
        try:
            prev_df = pd.read_excel(OUTPUT_EXCEL)
            if OUTPUT_COLUMN in prev_df.columns:
                # Create merge keys that handle null hours properly
                # Use fillna(-999999) as a sentinel value for null hours in the merge key
                prev_df['_merge_hours'] = prev_df[HOURS_COLUMN].fillna(-999999)
                prev_df['_merge_summary'] = prev_df[SUMMARY_COLUMN].fillna('')
                df['_merge_hours'] = df[HOURS_COLUMN].fillna(-999999)
                df['_merge_summary'] = df[SUMMARY_COLUMN].fillna('')
                
                prev_small = (
                    prev_df[['_merge_hours', '_merge_summary', OUTPUT_COLUMN]]
                    .dropna(subset=[OUTPUT_COLUMN])
                    .drop_duplicates(subset=['_merge_hours', '_merge_summary', OUTPUT_COLUMN])
                )
                logger.info(
                    "Merging %s existing descriptions from %s back into master dataset",
                    len(prev_small),
                    OUTPUT_EXCEL,
                )
                df = df.merge(
                    prev_small,
                    on=['_merge_hours', '_merge_summary'],
                    how="left",
                    suffixes=("", "_prev"),
                )
                # Clean up merge keys
                df.drop(columns=['_merge_hours', '_merge_summary'], inplace=True)
                
                # Prefer descriptions from the previous output if present
                if OUTPUT_COLUMN not in df.columns and f"{OUTPUT_COLUMN}_prev" in df.columns:
                    df.rename(columns={f"{OUTPUT_COLUMN}_prev": OUTPUT_COLUMN}, inplace=True)
                elif f"{OUTPUT_COLUMN}_prev" in df.columns:
                    df[OUTPUT_COLUMN] = df[OUTPUT_COLUMN].combine_first(df[f"{OUTPUT_COLUMN}_prev"])
                    df.drop(columns=[f"{OUTPUT_COLUMN}_prev"], inplace=True)
        except Exception as exc:
            logger.warning(
                "Failed to merge existing descriptions from %s: %s",
                OUTPUT_EXCEL,
                exc,
            )

    if HOURS_COLUMN not in df.columns:
        raise KeyError(f"Expected hours column '{HOURS_COLUMN}' not found in Excel. "
                       f"Available columns: {list(df.columns)}")

    if SUMMARY_COLUMN not in df.columns:
        raise KeyError(f"Expected summary column '{SUMMARY_COLUMN}' not found in Excel. "
                       f"Available columns: {list(df.columns)}")

    # Prepare the output column if missing
    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = pd.NA

    # Enrich with Category data from additional.xlsx only if enabled
    if USE_ADDITIONAL:
        df = enrich_with_category(df)

    total_rows = len(df)
    logger.info("Total rows in master sheet after merge (if any): %s", total_rows)

    # Collect rows that still need descriptions (supports resume)
    # Use a dict to deduplicate by (hours, summary) key
    pending_map: Dict[Tuple[float, str], Tuple[int, float, str, Optional[str], List[int]]] = {}

    for idx, row in df.iterrows():
        hours = row[HOURS_COLUMN]
        summary = row[SUMMARY_COLUMN]
        category_val = row.get(CATEGORY_OUTPUT_COLUMN, pd.NA)

        # Skip rows with missing summary
        if pd.isna(summary) or str(summary).strip() == "":
            continue

        # If description already exists, skip (resume behavior)
        existing = row.get(OUTPUT_COLUMN, pd.NA)
        if pd.notna(existing) and str(existing).strip() != "":
            continue

        # Handle missing hours by using 0.0 as default
        hours_val = 0.0 if pd.isna(hours) else float(hours)

        category = None
        if pd.notna(category_val) and str(category_val).strip() != "":
            category = str(category_val)

        key = (hours_val, str(summary).strip())
        
        if key not in pending_map:
            # First occurrence: store (first_idx, hours, summary, category, [all_indices])
            pending_map[key] = (idx, hours_val, str(summary), category, [idx])
        else:
            # Duplicate: just append this row index to the list
            pending_map[key][4].append(idx)

    # Convert to list for processing
    pending = [(first_idx, hours, summary, category, all_indices) 
               for first_idx, hours, summary, category, all_indices in pending_map.values()]

    total_unique = len(pending)
    total_rows_affected = sum(len(indices) for _, _, _, _, indices in pending)
    logger.info(
        "Found %s unique (hours, summary) combinations needing descriptions, "
        "covering %s total rows",
        total_unique,
        total_rows_affected,
    )

    # If a limit was provided, only process up to that many unique combinations
    if max_samples is not None and max_samples > 0:
        pending = pending[:max_samples]

    if not pending:
        logger.info("No rows need invoice descriptions. Nothing to do.")
        return

    # Use a larger thread pool but actual concurrency is controlled by semaphore
    max_workers = min(MAX_CONCURRENT_REQUESTS * 3, len(pending))
    logger.info(
        "Submitting %s pending rows to OpenRouter with max %s concurrent API calls (thread pool: %s)...",
        len(pending),
        MAX_CONCURRENT_REQUESTS,
        max_workers,
    )
    logger.info("Using Model ID: %s", MODEL_ID)
    logger.info("Request delay: %.2f seconds between API calls", REQUEST_DELAY_SECONDS)

    completed = 0
    successes = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data: Dict[Any, Tuple[int, List[int]]] = {}

        for i, (first_idx, hours, summary, category, all_indices) in enumerate(pending):
            # Stagger submissions slightly to avoid thundering herd
            if i > 0 and i % 10 == 0:
                time.sleep(0.1)
            
            future = executor.submit(process_row, first_idx, hours, summary, category, api_key)
            # Store the representative index and all duplicate indices
            future_to_data[future] = (first_idx, all_indices)

        with tqdm(total=len(future_to_data), desc="Rows", unit="row") as pbar:
            for future in as_completed(future_to_data):
                first_idx, all_indices = future_to_data[future]
                try:
                    row_idx, content = future.result()
                except Exception as exc:
                    logger.error("Worker for row %s raised: %s", first_idx, exc)
                    row_idx, content = first_idx, None

                completed += 1

                if content:
                    # Apply the same description to ALL rows with this (hours, summary) combination
                    for idx in all_indices:
                        df.at[idx, OUTPUT_COLUMN] = content.strip()
                    successes += 1
                    if len(all_indices) > 1:
                        logger.info(
                            "Applied description to %s duplicate rows (indices: %s)",
                            len(all_indices),
                            all_indices[:5] if len(all_indices) > 5 else all_indices,
                        )

                if completed % SAVE_EVERY_N == 0:
                    logger.info(
                        "Saving progress after %s rows processed (%s successful)...",
                        completed,
                        successes,
                    )
                    # In limited mode, only persist processed rows so far
                    if max_samples is not None and max_samples > 0:
                        mask = df[OUTPUT_COLUMN].notna() & df[OUTPUT_COLUMN].astype(str).str.strip().ne("")
                        save_df = df.loc[mask].copy()
                    else:
                        save_df = df
                    save_df.to_excel(OUTPUT_EXCEL, index=False)

                pbar.update(1)

    # Final output: in limited mode, only keep rows that have an invoice description
    if max_samples is not None and max_samples > 0:
        mask = df[OUTPUT_COLUMN].notna() & df[OUTPUT_COLUMN].astype(str).str.strip().ne("")
        output_df = df.loc[mask].copy()
        logger.info(
            "Final save (limited mode) to: %s with %s rows containing descriptions (before dedup)",
            OUTPUT_EXCEL,
            len(output_df),
        )
    else:
        output_df = df
        logger.info("Final save to: %s", OUTPUT_EXCEL)

    # Remove duplicates based on (Billable Hours, Summary Notes, Invoice Description)
    # Keep the first occurrence
    before_dedup = len(output_df)
    output_df = output_df.drop_duplicates(
        subset=[HOURS_COLUMN, SUMMARY_COLUMN, OUTPUT_COLUMN],
        keep='first'
    ).reset_index(drop=True)
    after_dedup = len(output_df)
    
    if before_dedup > after_dedup:
        logger.info(
            "Removed %s duplicate rows; keeping %s unique rows",
            before_dedup - after_dedup,
            after_dedup,
        )

    output_df.to_excel(OUTPUT_EXCEL, index=False)
    logger.info(
        "Done. Processed %s pending rows, successfully generated %s descriptions.",
        completed,
        successes,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate invoice descriptions from FY19_to_FY25_Final.xlsx using OpenRouter.\n"
            "If --limit/-n is provided, only that many pending rows will be processed; "
            "otherwise all pending rows are processed."
        )
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending rows to process in this run (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Create a file handler for logging
    file_handler = logging.FileHandler("openrouter_invoice_descriptions.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # Log to console
            file_handler,             # Log to file
        ],
    )
    print("Logging detailed output to openrouter_invoice_descriptions.log")

    args = parse_args()
    main(max_samples=args.limit)


