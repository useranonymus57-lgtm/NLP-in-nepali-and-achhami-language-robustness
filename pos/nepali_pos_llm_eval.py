# ============================================================
# Nepali + Achhami POS Tagging with LLMs via OpenRouter
# ============================================================

import pandas as pd
import numpy as np
import re
import requests
import json
import time
import os
import ast
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD .env
# ============================================================

load_dotenv()

# ============================================================
# 2. CONFIG CLASS
# ============================================================

@dataclass
class Config:
    # ── API ───────────────────────────────────────────────
    api_key        : str   = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url       : str   = "https://openrouter.ai/api/v1"
    referrer       : str   = field(default_factory=lambda: os.getenv("OPENROUTER_REFERRER", "http://localhost:3000"))

    # ── Models ────────────────────────────────────────────
    models         : Dict[str, str] = field(default_factory=lambda: {
        "Claude 3.5 Haiku" : "anthropic/claude-3.5-haiku",
        "GPT-4o Mini"      : "openai/gpt-4o-mini",
        "Llama 3.1 70B"    : "meta-llama/llama-3.1-70b-instruct",
    })

    # ── Data ──────────────────────────────────────────────
    nepali_path    : str   = field(default_factory=lambda: os.getenv("NEPALI_DATA_PATH",  "dataset_clean.csv"))
    achhami_path   : str   = field(default_factory=lambda: os.getenv("ACHHAMI_DATA_PATH", "achhami_clean.csv"))

    # ── Request settings ──────────────────────────────────
    timeout        : int   = 60
    max_retries    : int   = 3
    retry_delay    : float = 2.0
    request_delay  : float = 0.5

    # ── Output ────────────────────────────────────────────
    output_dir     : str   = "pos_results"

    # ── Valid POS tags ────────────────────────────────────
    valid_pos      : set   = field(default_factory=lambda: {
        "NOUN","PROPN","VERB","AUX","ADJ","ADV",
        "ADP","PRON","DET","NUM","CCONJ","SCONJ",
        "PART","PUNCT","INTJ","SYM","X"
    })

# ============================================================
# 3. VALIDATE CONFIG
# ============================================================

def validate_config(cfg: Config) -> List[str]:
    """
    Check API key, CSV paths, referrer URL.
    Returns list of error messages (empty = all OK).
    """
    errors = []

    # API key
    if not cfg.api_key:
        errors.append("❌ OPENROUTER_API_KEY is missing in .env")
    elif not cfg.api_key.startswith("sk-"):
        errors.append("⚠️  OPENROUTER_API_KEY may be invalid (should start with sk-)")

    # CSV files
    for label, path in [("Nepali",  cfg.nepali_path),
                         ("Achhami", cfg.achhami_path)]:
        if not path:
            errors.append(f"❌ {label} data path missing in .env")
        elif not os.path.exists(path):
            errors.append(f"❌ {label} file not found: {path}")

    # Referrer URL
    if not cfg.referrer.startswith("http"):
        errors.append("⚠️  OPENROUTER_REFERRER should be a valid URL")

    return errors


def print_config_check(cfg: Config):
    print("\n" + "="*60)
    print("🔧  CONFIGURATION CHECK")
    print("="*60)

    errors = validate_config(cfg)

    # API key
    if cfg.api_key:
        masked = cfg.api_key[:8] + "..." + cfg.api_key[-4:]
        print(f"   API Key        : ✅ Loaded  ({masked})")
    else:
        print(f"   API Key        : ❌ Missing")

    # Files
    for label, path in [("Nepali  file  ", cfg.nepali_path),
                         ("Achhami file  ", cfg.achhami_path)]:
        if path and os.path.exists(path):
            size = os.path.getsize(path) // 1024
            print(f"   {label}: ✅ Found   ({path}, {size}KB)")
        else:
            print(f"   {label}: ❌ Missing ({path})")

    # Referrer
    print(f"   Referrer       : {cfg.referrer}")
    print(f"   Output dir     : ./{cfg.output_dir}/")
    print(f"   Models         : {list(cfg.models.keys())}")

    if errors:
        print("\n   ERRORS:")
        for e in errors:
            print(f"   {e}")
        raise SystemExit("Fix config errors before running.")

    print("\n   ✅ All config checks passed!")
    print("="*60)

# ============================================================
# 4. FETCH AVAILABLE MODELS FROM OPENROUTER
#    Prevents 404 errors from invalid model IDs
# ============================================================

def fetch_available_models(cfg: Config) -> Dict[str, bool]:
    """
    Query OpenRouter /models endpoint to verify
    which model IDs are actually available.
    Returns {model_id: True/False}
    """
    print("\n" + "="*60)
    print("🌐  FETCHING AVAILABLE MODELS FROM OPENROUTER")
    print("="*60)

    available = {}
    try:
        resp = requests.get(
            f"{cfg.base_url}/models",
            headers={
                "Authorization" : f"Bearer {cfg.api_key}",
                "HTTP-Referer"  : cfg.referrer,
            },
            timeout=cfg.timeout
        )
        resp.raise_for_status()
        data        = resp.json()
        model_ids   = {m["id"] for m in data.get("data", [])}
        print(f"   Found {len(model_ids)} models on OpenRouter")

        for display_name, model_id in cfg.models.items():
            found = model_id in model_ids
            available[display_name] = found
            status = "✅ Available" if found else "❌ Not found"
            print(f"   {display_name:<22} {status}  ({model_id})")

    except Exception as e:
        print(f"   ⚠️  Could not fetch model list: {e}")
        print("   Proceeding with test calls to verify...")
        # Fall back to test calls
        available = test_models_with_call(cfg)

    print("="*60)
    return available


def test_models_with_call(cfg: Config) -> Dict[str, bool]:
    """
    Fallback: test each model with a tiny request.
    """
    print("\n🔌  TESTING MODELS WITH SAMPLE CALL")
    test_prompt = (
        'Tag these 3 tokens with UD POS tags: [नेपाल, राम्रो, छ]\n'
        'Return ONLY: ["PROPN","ADJ","AUX"]'
    )
    results = {}
    for name, model_id in cfg.models.items():
        print(f"\n   Testing: {name} ({model_id})")
        try:
            resp = requests.post(
                f"{cfg.base_url}/chat/completions",
                headers={
                    "Authorization" : f"Bearer {cfg.api_key}",
                    "HTTP-Referer"  : cfg.referrer,
                    "Content-Type"  : "application/json",
                },
                json={
                    "model"      : model_id,
                    "messages"   : [{"role":"user","content":test_prompt}],
                    "max_tokens" : 32,
                    "temperature": 0,
                },
                timeout=cfg.timeout
            )
            if resp.status_code == 200:
                out = resp.json()["choices"][0]["message"]["content"].strip()
                print(f"   Output : {out}")
                print(f"   Status : ✅ Working")
                results[name] = True
            else:
                print(f"   Status : ❌ HTTP {resp.status_code}")
                results[name] = False
        except Exception as e:
            print(f"   Status : ❌ {str(e)[:80]}")
            results[name] = False
    return results

# ============================================================
# 5. LOAD DATASETS
# ============================================================

def parse_list_col(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        value = str(value).strip("[]")
        return [x.strip().strip("'\"") for x in value.split(",") if x.strip()]


def load_dataset(filepath: str, dialect: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"\n✅ Loaded {dialect}: {df.shape[0]} rows")
    print(f"   Columns : {df.columns.tolist()}")

    df["parsed_tokens"] = df["parsed_tokens"].apply(parse_list_col)
    df["parsed_labels"] = df["parsed_labels"].apply(parse_list_col)

    if "sentence" not in df.columns:
        df["sentence"] = df["parsed_tokens"].apply(" ".join)

    # Keep only rows with matching token/label counts
    before = len(df)
    df = df[df.apply(
        lambda r: (len(r["parsed_tokens"]) > 0 and
                   len(r["parsed_labels"]) > 0 and
                   len(r["parsed_tokens"]) == len(r["parsed_labels"])),
        axis=1
    )].reset_index(drop=True)

    print(f"   Valid rows   : {len(df)} / {before}")
    print(f"   Total tokens : {df['parsed_tokens'].map(len).sum()}")
    print(f"   Avg tokens   : {df['parsed_tokens'].map(len).mean():.1f}")
    print(f"   Sample       : {df.iloc[0]['sentence'][:55]}")

    all_labels = [l for row in df["parsed_labels"] for l in row]
    dist       = pd.Series(all_labels).value_counts().head(6).to_dict()
    print(f"   Top POS tags : {dist}")
    return df

# ============================================================
# 6. OPENROUTER CLASS
# ============================================================

class OpenRouterPOS:
    """Handles all OpenRouter API calls for POS tagging."""

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.headers = {
            "Authorization" : f"Bearer {cfg.api_key}",
            "HTTP-Referer"  : cfg.referrer,
            "Content-Type"  : "application/json",
        }

    # ── Prompt ────────────────────────────────────────────
    def _build_prompt(self, tokens: List[str], dialect: str) -> str:
        token_list   = ", ".join(tokens)
        dialect_note = ""
        if dialect == "Achhami":
            dialect_note = (
                "\nNOTE: Tokens are in ACHHAMI dialect "
                "(Achham district, western Nepal). "
                "Apply UD POS rules for this dialect."
            )
        return (
            f"You are a Nepali linguistics expert for Universal Dependencies POS tagging."
            f"{dialect_note}\n\n"
            f"TOKENS ({len(tokens)}): [{token_list}]\n\n"
            f"RULES:\n"
            f"1. Return ONLY a JSON array of {len(tokens)} POS tags\n"
            f"2. Tags: NOUN PROPN VERB AUX ADJ ADV ADP PRON DET NUM "
            f"CCONJ SCONJ PART PUNCT INTJ SYM X\n"
            f"3. No explanation, no markdown\n"
            f'4. Format: ["TAG1","TAG2",...]\n\n'
            f"OUTPUT:"
        )

        # ── Parse response ────────────────────────────────────
    def _parse_response(self, text: str,
                        expected_len: int) -> Optional[List[str]]:
        if not text:
            return None
        text = text.strip()

        # ── Try 1: direct JSON ────────────────────────────────
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                tags = [str(t).strip().upper() for t in parsed]
                if len(tags) == expected_len:
                    return tags
        except Exception:
            pass

        # ── Try 2: fix truncated JSON by closing the array ────
        # e.g. '["PROPN","NOUN","N' → add missing closing chars
        if text.startswith("[") and not text.endswith("]"):
            # Find last complete quoted tag
            fixed = text.rstrip(', "\'')
            # Remove any partial tag at the end (e.g. "N or "PRO)
            fixed = re.sub(r',?\s*"[^"]*$', '', fixed)
            fixed = fixed + "]"
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, list):
                    tags = [str(t).strip().upper() for t in parsed]
                    # Pad with X if truncated
                    if len(tags) < expected_len:
                        print(f"   ⚠️  Truncated response: got {len(tags)}, "
                            f"expected {expected_len}. Padding with X.")
                        tags += ["X"] * (expected_len - len(tags))
                    return tags[:expected_len]
            except Exception:
                pass

        # ── Try 3: regex JSON array ───────────────────────────
        m = re.search(r'\[([^\[\]]+)\]', text, re.DOTALL)
        if m:
            try:
                tags = json.loads("[" + m.group(1) + "]")
                tags = [str(t).strip().upper() for t in tags]
                if len(tags) == expected_len:
                    return tags
            except Exception:
                pass

        # ── Try 4: extract all valid quoted tags ──────────────
        # Works even on truncated responses
        tags = re.findall(r'"([A-Z]{1,6})"', text)
        if not tags:
            tags = re.findall(r"'([A-Z]{1,6})'", text)
        if tags:
            tags = [t.upper() for t in tags]
            if len(tags) == expected_len:
                return tags
            if len(tags) > expected_len:
                return tags[:expected_len]
            if len(tags) > 0:
                print(f"   ⚠️  Partial tags: got {len(tags)}, "
                    f"expected {expected_len}. Padding with X.")
                tags += ["X"] * (expected_len - len(tags))
                return tags

        # ── Try 5: comma-separated plain text ─────────────────
        parts = [p.strip().strip('"\'[] ') for p in text.split(",")]
        tags  = [p.upper() for p in parts if p.upper() in self.cfg.valid_pos]
        if len(tags) >= expected_len:
            return tags[:expected_len]
        if len(tags) > 0:
            tags += ["X"] * (expected_len - len(tags))
            return tags

        return None

    # ── Normalize tags ────────────────────────────────────
    def _normalize_tag(self, tag: str) -> str:
        """Map non-standard tags to closest valid UD tag."""
        tag = str(tag).strip().upper()
        mapping = {
            "CONJ"   : "CCONJ",
            "SUBCONJ": "SCONJ",
            "PUNC"   : "PUNCT",
            "PROP"   : "PROPN",
            "VRB"    : "VERB",
        }
        return mapping.get(tag, tag if tag in self.cfg.valid_pos else "X")

    # ── Single API call with retries + backoff ────────────
    def call(self, tokens: List[str],
             model_id: str,
             dialect: str) -> Optional[List[str]]:

        prompt = self._build_prompt(tokens, dialect)
        payload = {
            "model"      : model_id,
            "messages"   : [
                {
                    "role"   : "system",
                    "content": (
                        "You are a Nepali/Achhami POS tagging expert. "
                        "Output ONLY JSON arrays. No extra text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens" : max(256, len(tokens) * 12),
            "temperature": 0,
        }

        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers = self.headers,
                    json    = payload,
                    timeout = self.cfg.timeout,
                )

                if resp.status_code == 429:
                    wait = self.cfg.retry_delay * (2 ** attempt)
                    print(f"   ⏳ Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    print(f"   ❌ Model not found: {model_id}")
                    return None

                resp.raise_for_status()
                text      = resp.json()["choices"][0]["message"]["content"].strip()
                pred_tags = self._parse_response(text, len(tokens))

                if pred_tags is not None:
                    return [self._normalize_tag(t) for t in pred_tags]

                print(f"   ⚠️  Parse failed (attempt {attempt+1}). "
                      f"Response: {text[:60]}")

            except requests.exceptions.Timeout:
                print(f"   ⏰ Timeout (attempt {attempt+1}/{self.cfg.max_retries})")
            except Exception as e:
                print(f"   ❌ Error (attempt {attempt+1}): {str(e)[:80]}")

            if attempt < self.cfg.max_retries - 1:
                time.sleep(self.cfg.retry_delay * (attempt + 1))

        return None

    # ── Run on full dataset with checkpointing ────────────
    def run_dataset(self,
                    model_name : str,
                    model_id   : str,
                    df         : pd.DataFrame,
                    dialect    : str,
                    output_dir : str) -> Tuple[List, List, int]:

        print(f"\n{'='*60}")
        print(f"🤖  {model_name}  |  📂  {dialect}")
        print(f"    Rows    : {len(df)}")
        print(f"    Tokens  : {df['parsed_tokens'].map(len).sum()}")
        print(f"{'='*60}")

        # ── Checkpoint file ───────────────────────────────
        ckpt_name = (f"checkpoint_{model_name.replace(' ','_')}"
                     f"_{dialect}.csv")
        ckpt_path = os.path.join(output_dir, ckpt_name)

        # Resume from checkpoint if exists
        done_idx = set()
        ckpt_rows = []
        if os.path.exists(ckpt_path):
            ckpt_df   = pd.read_csv(ckpt_path)
            done_idx  = set(ckpt_df["row_idx"].tolist())
            ckpt_rows = ckpt_df.to_dict("records")
            print(f"   ♻️  Resuming from checkpoint: "
                  f"{len(done_idx)} rows done")

        gold_all     = []
        pred_all     = []
        failed_count = 0
        new_rows     = []

        for i, row in enumerate(df.itertuples()):
            tokens = row.parsed_tokens
            labels = row.parsed_labels

            if not tokens or not labels:
                continue

            # Already done → use checkpoint
            if i in done_idx:
                existing = next(
                    (r for r in ckpt_rows if r["row_idx"] == i), None
                )
                if existing:
                    pred = parse_list_col(existing["pred_tags"])
                    gold_all.append(labels)
                    pred_all.append(pred)
                    continue

            # Call API
            pred_tags = self.call(tokens, model_id, dialect)

            if pred_tags is None:
                failed_count += 1
                pred_tags = ["X"] * len(tokens)

            # Validate
            pred_tags = [
                t if t in self.cfg.valid_pos else "X"
                for t in pred_tags
            ]

            gold_all.append(labels)
            pred_all.append(pred_tags)

            # Save to checkpoint
            new_rows.append({
                "row_idx"   : i,
                "sentence"  : row.sentence,
                "gold_tags" : str(labels),
                "pred_tags" : str(pred_tags),
            })

            # Write checkpoint every 25 rows
            if len(new_rows) % 25 == 0:
                all_rows = ckpt_rows + new_rows
                pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)

            # Progress log
            if (i + 1) % 25 == 0 or i == 0:
                correct = sum(g == p for g, p in zip(labels, pred_tags))
                pct     = correct / len(labels) if labels else 0
                print(f"   [{i+1:>3}/{len(df)}]  "
                      f"tokens={len(tokens):>3}  "
                      f"sent_acc={pct:.0%}  "
                      f"failed={failed_count}")

            time.sleep(self.cfg.request_delay)

        # Final checkpoint save
        all_rows = ckpt_rows + new_rows
        pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)
        print(f"\n   ✅ Done — Failed: {failed_count}/{len(df)}")
        print(f"   💾 Checkpoint: {ckpt_path}")

        return gold_all, pred_all, failed_count

# ============================================================
# 7. METRICS
# ============================================================

def compute_metrics(gold_all : List[List[str]],
                    pred_all : List[List[str]],
                    model_name: str,
                    dialect   : str) -> Dict:

    flat_gold = [t for s in gold_all for t in s]
    flat_pred = [t for s in pred_all for t in s]
    labels    = sorted(set(flat_gold + flat_pred))

    # ── Token-level counts ────────────────────────────────
    tp = sum(g == p for g, p in zip(flat_gold, flat_pred))
    fp = sum(g != p for g, p in zip(flat_gold, flat_pred))
    fn = fp   # symmetric for flat token classification
    tn = 0    # not meaningful for multi-class

    metrics = {
        "model"           : model_name,
        "dialect"         : dialect,
        "accuracy"        : accuracy_score(flat_gold, flat_pred),
        "macro_f1"        : f1_score(flat_gold, flat_pred,
                                     average="macro",    labels=labels, zero_division=0),
        "micro_f1"        : f1_score(flat_gold, flat_pred,
                                     average="micro",    labels=labels, zero_division=0),
        "weighted_f1"     : f1_score(flat_gold, flat_pred,
                                     average="weighted", labels=labels, zero_division=0),
        "macro_precision" : precision_score(flat_gold, flat_pred,
                                            average="macro", labels=labels, zero_division=0),
        "macro_recall"    : recall_score(flat_gold, flat_pred,
                                         average="macro", labels=labels, zero_division=0),
        "true_positives"  : tp,
        "false_positives" : fp,
        "false_negatives" : fn,
        "total_tokens"    : len(flat_gold),
        "total_sentences" : len(gold_all),
    }

    # ── Print ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"📊  {model_name}  |  {dialect}")
    print(f"{'='*55}")
    print(f"   {'Accuracy':<22} {metrics['accuracy']:.4f}")
    print(f"   {'Macro F1':<22} {metrics['macro_f1']:.4f}")
    print(f"   {'Micro F1':<22} {metrics['micro_f1']:.4f}")
    print(f"   {'Weighted F1':<22} {metrics['weighted_f1']:.4f}")
    print(f"   {'Macro Precision':<22} {metrics['macro_precision']:.4f}")
    print(f"   {'Macro Recall':<22} {metrics['macro_recall']:.4f}")
    print(f"   {'True Positives':<22} {tp}")
    print(f"   {'False Positives':<22} {fp}")
    print(f"   {'Total Tokens':<22} {metrics['total_tokens']}")
    print(f"\n   Per-class Report:")
    print(classification_report(flat_gold, flat_pred,
                                labels=labels, zero_division=0))
    return metrics

# ============================================================
# 8. SHOW SAMPLES
# ============================================================

def show_samples(df, gold_all, pred_all,
                 model_name, dialect, n=3):
    print(f"\n{'='*65}")
    print(f"🔍  Samples — {model_name} | {dialect}")
    print(f"{'='*65}")
    for i in range(min(n, len(df))):
        tokens = df.iloc[i]["parsed_tokens"]
        gold   = gold_all[i]
        pred   = pred_all[i]
        n_show = min(len(tokens), len(gold), len(pred))
        print(f"\n  #{i+1}: {df.iloc[i]['sentence'][:52]}...")
        print(f"  {'Token':<22} {'Gold':>8} {'Pred':>10} {'':>4}")
        print(f"  {'-'*46}")
        correct = 0
        for tok, g, p in zip(tokens[:n_show], gold[:n_show], pred[:n_show]):
            mark = "✅" if g == p else "❌"
            if g == p:
                correct += 1
            print(f"  {tok:<22} {g:>8} {p:>10} {mark:>4}")
        pct = correct / n_show if n_show > 0 else 0
        print(f"  Accuracy: {correct}/{n_show} = {pct:.1%}")

# ============================================================
# 9. VISUALIZATIONS
# ============================================================

def save_fig(output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   💾 Saved: {path}")


def plot_per_tag_f1(gold_all, pred_all,
                    model_name, dialect, output_dir):
    flat_gold = [t for s in gold_all for t in s]
    flat_pred = [t for s in pred_all for t in s]
    labels    = sorted(set(flat_gold))
    scores    = f1_score(flat_gold, flat_pred,
                         labels=labels, average=None, zero_division=0)
    colors    = ["#4CAF50" if s >= 0.5 else "#FF9800" if s >= 0.2
                 else "#F44336" for s in scores]

    plt.figure(figsize=(13, 5))
    bars = plt.bar(labels, scores, color=colors, edgecolor="white")
    for bar, sc in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{sc:.2f}", ha="center", va="bottom", fontsize=9)
    plt.title(f"Per-Tag F1 — {model_name} | {dialect}",
              fontweight="bold", fontsize=13)
    plt.xlabel("POS Tag")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.12)
    plt.xticks(rotation=45)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(output_dir,
             f"per_tag_f1_{model_name.replace(' ','_')}_{dialect}.png")
    plt.close()


def plot_confusion_matrix(gold_all, pred_all,
                           model_name, dialect, output_dir):
    flat_gold = [t for s in gold_all for t in s]
    flat_pred = [t for s in pred_all for t in s]
    labels    = sorted(set(flat_gold))
    cm        = confusion_matrix(flat_gold, flat_pred, labels=labels)

    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", linewidths=0.5, annot_kws={"size": 8})
    plt.title(f"Confusion Matrix — {model_name} | {dialect}",
              fontweight="bold", fontsize=13)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(output_dir,
             f"confusion_{model_name.replace(' ','_')}_{dialect}.png")
    plt.close()


def plot_multi_model_comparison(results_df, output_dir):
    """Multi-model metric comparison grouped by dialect."""
    metrics       = ["accuracy","macro_f1","weighted_f1",
                     "macro_precision","macro_recall"]
    metric_labels = ["Accuracy","Macro F1","Weighted F1",
                     "Precision","Recall"]
    model_names   = results_df["model"].unique()
    colors        = {"Nepali":"#2196F3","Achhami":"#FF5722"}

    fig, axes = plt.subplots(1, len(model_names),
                              figsize=(7*len(model_names), 6),
                              sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_names):
        x     = np.arange(len(metrics))
        width = 0.35
        for j, dialect in enumerate(["Nepali","Achhami"]):
            row = results_df[(results_df["model"]==model) &
                             (results_df["dialect"]==dialect)]
            if row.empty:
                continue
            vals = [row.iloc[0][m] for m in metrics]
            bars = ax.bar(x + j*width, vals, width,
                          label=dialect,
                          color=colors[dialect], alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(model, fontweight="bold", fontsize=11)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(metric_labels, rotation=30, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.legend(title="Dialect")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("LLM POS Tagging — Nepali vs Achhami",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(output_dir, "multi_model_comparison.png")
    plt.close()


def plot_performance_degradation(results_df, output_dir):
    """Show performance drop Nepali → Achhami per model."""
    model_names     = results_df["model"].unique()
    gaps_f1, gaps_acc, names = [], [], []

    for model in model_names:
        nep = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Nepali")]
        ach = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Achhami")]
        if nep.empty or ach.empty:
            continue
        gaps_f1.append(nep.iloc[0]["macro_f1"] - ach.iloc[0]["macro_f1"])
        gaps_acc.append(nep.iloc[0]["accuracy"] - ach.iloc[0]["accuracy"])
        names.append(model)

    x, width = np.arange(len(names)), 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, gaps_f1, width, label="Macro F1 Drop",
                color=["#F44336" if g > 0.05 else "#4CAF50" for g in gaps_f1],
                alpha=0.85)
    b2 = ax.bar(x + width/2, gaps_acc, width, label="Accuracy Drop",
                color=["#FF9800" if g > 0.05 else "#8BC34A" for g in gaps_acc],
                alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.002, f"{h:+.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(0,    color="black", linewidth=0.8)
    ax.axhline(0.05, color="red",   linestyle="--",
               alpha=0.5, label="5% threshold")
    ax.set_title("Performance Degradation: Nepali → Achhami",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Performance Drop")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(output_dir, "performance_degradation.png")
    plt.close()


def plot_f1_heatmap(results_dict: Dict, output_dir: str):
    data = {}
    for (model, dialect), (gold_all, pred_all) in results_dict.items():
        flat_gold = [t for s in gold_all for t in s]
        flat_pred = [t for s in pred_all for t in s]
        labels    = sorted(set(flat_gold))
        scores    = f1_score(flat_gold, flat_pred,
                             labels=labels, average=None, zero_division=0)
        data[f"{model} ({dialect})"] = dict(zip(labels, scores))

    df_heat = pd.DataFrame(data).T.fillna(0)
    plt.figure(figsize=(18, len(data)*1.2 + 2))
    sns.heatmap(df_heat, annot=True, fmt=".2f",
                cmap="RdYlGn", linewidths=0.5,
                vmin=0, vmax=1, annot_kws={"size": 9})
    plt.title("Per-Tag F1 — All Models × Both Dialects",
              fontweight="bold", fontsize=13)
    plt.xlabel("POS Tag")
    plt.ylabel("Model (Dialect)")
    plt.tight_layout()
    save_fig(output_dir, "f1_heatmap_all.png")
    plt.close()

# ============================================================
# 10. TEXT REPORT
# ============================================================

def save_text_report(results_df     : pd.DataFrame,
                     results_dict   : Dict,
                     output_dir     : str):
    path = os.path.join(output_dir, "report.txt")
    lines = []

    lines.append("=" * 65)
    lines.append("  NEPALI + ACHHAMI POS TAGGING — LLM EVALUATION REPORT")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    # ── Overall metrics table ─────────────────────────────
    lines.append("\n📋 OVERALL METRICS\n")
    cols = ["model","dialect","accuracy","macro_f1",
            "weighted_f1","macro_precision","macro_recall",
            "total_tokens"]
    lines.append(results_df[cols].to_string(index=False))

    # ── Per-class report ──────────────────────────────────
    lines.append("\n\n📊 PER-CLASS REPORTS\n")
    for (model, dialect), (gold_all, pred_all) in results_dict.items():
        flat_gold = [t for s in gold_all for t in s]
        flat_pred = [t for s in pred_all for t in s]
        labels    = sorted(set(flat_gold + flat_pred))
        lines.append(f"\n{'='*50}")
        lines.append(f"  {model}  |  {dialect}")
        lines.append(f"{'='*50}")
        lines.append(classification_report(
            flat_gold, flat_pred,
            labels=labels, zero_division=0
        ))

    # ── Degradation ───────────────────────────────────────
    lines.append("\n📉 PERFORMANCE DEGRADATION (Nepali → Achhami)\n")
    for model in results_df["model"].unique():
        nep = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Nepali")]
        ach = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Achhami")]
        if nep.empty or ach.empty:
            continue
        gap_f1  = nep.iloc[0]["macro_f1"] - ach.iloc[0]["macro_f1"]
        gap_acc = nep.iloc[0]["accuracy"] - ach.iloc[0]["accuracy"]
        flag    = "⚠️  Large gap" if gap_f1 > 0.05 else "✅ Robust"
        lines.append(
            f"  {model:<22}  "
            f"F1 drop: {gap_f1:+.4f}  "
            f"Acc drop: {gap_acc:+.4f}  {flag}"
        )

    # ── Best model ────────────────────────────────────────
    lines.append("\n\n🏆 BEST MODEL PER DIALECT\n")
    for dialect in ["Nepali","Achhami"]:
        sub = results_df[results_df["dialect"]==dialect]
        if sub.empty:
            continue
        best = sub.loc[sub["macro_f1"].idxmax()]
        lines.append(
            f"  {dialect}: {best['model']}  "
            f"(F1={best['macro_f1']:.4f}, "
            f"Acc={best['accuracy']:.4f})"
        )

    report_text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"   💾 Report saved: {path}")
    return report_text

# ============================================================
# 11. MAIN
# ============================================================

def main():

    # ── Init config ────────────────────────────────────────
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Step 1: Validate config ────────────────────────────
    print_config_check(cfg)

    # ── Step 2: Fetch & verify models ─────────────────────
    model_status = fetch_available_models(cfg)

    active_models = {
        name: mid for name, mid in cfg.models.items()
        if model_status.get(name, False)
    }
    if not active_models:
        raise SystemExit("❌ No models available. "
                         "Check API key / credits.")
    print(f"\n✅ Active models: {list(active_models.keys())}")

    # ── Step 3: Load datasets ──────────────────────────────
    print("\n" + "="*60)
    print("📂  LOADING DATASETS")
    print("="*60)
    df_nepali  = load_dataset(cfg.nepali_path,  "Nepali")
    df_achhami = load_dataset(cfg.achhami_path, "Achhami")

    # ── Step 4: Run models ─────────────────────────────────
    pos_runner   = OpenRouterPOS(cfg)
    all_results  = []
    results_dict = {}

    for model_name, model_id in active_models.items():
        for dialect, df in [("Nepali",  df_nepali),
                             ("Achhami", df_achhami)]:

            print(f"\n\n{'#'*60}")
            print(f"#  {model_name}  →  {dialect}")
            print(f"{'#'*60}")

            # Run
            gold_all, pred_all, failed = pos_runner.run_dataset(
                model_name, model_id, df,
                dialect, cfg.output_dir
            )
            results_dict[(model_name, dialect)] = (gold_all, pred_all)

            # Samples
            show_samples(df, gold_all, pred_all,
                         model_name, dialect, n=3)

            # Metrics
            metrics = compute_metrics(
                gold_all, pred_all, model_name, dialect
            )
            all_results.append(metrics)

            # Per-model/dialect plots
            plot_per_tag_f1(
                gold_all, pred_all,
                model_name, dialect, cfg.output_dir
            )
            plot_confusion_matrix(
                gold_all, pred_all,
                model_name, dialect, cfg.output_dir
            )

            # Save per-run results CSV
            run_df   = pd.DataFrame([metrics])
            run_path = os.path.join(
                cfg.output_dir,
                f"results_{model_name.replace(' ','_')}_{dialect}.csv"
            )
            run_df.to_csv(run_path, index=False)
            print(f"   💾 Results: {run_path}")

    # ── Step 5: Summary ────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n\n" + "="*65)
    print("📋  FINAL SUMMARY — ALL MODELS × BOTH DIALECTS")
    print("="*65)
    print(results_df[[
        "model","dialect","accuracy","macro_f1",
        "weighted_f1","macro_precision","macro_recall"
    ]].to_string(index=False))

    # ── Step 6: Summary plots ──────────────────────────────
    plot_multi_model_comparison(results_df, cfg.output_dir)
    plot_performance_degradation(results_df, cfg.output_dir)
    plot_f1_heatmap(results_dict, cfg.output_dir)

    # ── Step 7: Save summary CSV + report ─────────────────
    summary_path = os.path.join(cfg.output_dir, "results_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n💾 Summary CSV: {summary_path}")

    save_text_report(results_df, results_dict, cfg.output_dir)

    # ── Step 8: Best model + dialect gap ──────────────────
    print("\n" + "="*65)
    print("🏆  BEST MODEL PER DIALECT")
    print("="*65)
    for dialect in ["Nepali","Achhami"]:
        sub = results_df[results_df["dialect"]==dialect]
        if sub.empty:
            continue
        best = sub.loc[sub["macro_f1"].idxmax()]
        print(f"\n  {dialect}:")
        print(f"    Model    : {best['model']}")
        print(f"    Macro F1 : {best['macro_f1']:.4f}")
        print(f"    Accuracy : {best['accuracy']:.4f}")

    print("\n" + "="*65)
    print("📉  PERFORMANCE DEGRADATION (Nepali F1 − Achhami F1)")
    print("="*65)
    for model in results_df["model"].unique():
        nep = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Nepali")]["macro_f1"].values
        ach = results_df[(results_df["model"]==model) &
                         (results_df["dialect"]=="Achhami")]["macro_f1"].values
        if len(nep) and len(ach):
            gap  = nep[0] - ach[0]
            flag = "⚠️  Large gap" if gap > 0.05 else "✅ Robust"
            print(f"  {model:<22} {gap:+.4f}  {flag}")

    # ── Step 9: List all generated files ──────────────────
    print(f"\n📁 All files in ./{cfg.output_dir}/")
    print("="*65)
    for f in sorted(os.listdir(cfg.output_dir)):
        size = os.path.getsize(os.path.join(cfg.output_dir, f)) // 1024
        print(f"   {f:<55} {size:>5} KB")

    return results_df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()