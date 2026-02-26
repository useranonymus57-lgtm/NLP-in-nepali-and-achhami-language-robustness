import pandas as pd
import numpy as np
import re
import requests
import json
import time
import os
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION FROM .env
# ============================================

class Config:
    """Configuration loaded from .env file"""
    
    API_KEY = os.getenv('OPENROUTER_API_KEY')
    REFERRER = os.getenv('OPENROUTER_REFERRER', 'http://localhost')
    APP_TITLE = os.getenv('OPENROUTER_APP_TITLE', 'NER Dialect Eval')
    API_TIMEOUT = int(os.getenv('OPENROUTER_API_TIMEOUT', '120'))
    API_MAX_RETRIES = int(os.getenv('OPENROUTER_API_MAX_RETRIES', '3'))
    API_RETRY_BACKOFF = float(os.getenv('OPENROUTER_API_RETRY_BACKOFF', '2'))  # exponential base
    
    # Cost-efficient and capable models
    MODELS_TO_TEST = [
        "openai/gpt-4o-mini",              # Strong accuracy, moderate cost
        "anthropic/claude-3.5-haiku",           # Good quality, affordable tier
        "meta-llama/llama-3.1-70b-instruct"     # Better than 8B, still cost-efficient
    ]
    
    NEPALI_CSV = os.getenv('NEPALI_CSV_PATH', 'ner.csv')
    ACHHAMI_CSV = os.getenv('ACHHAMI_CSV_PATH', 'ner_achami - Sheet1.csv')
    
    API_DELAY = float(os.getenv('API_DELAY', '1'))
    
    @classmethod
    def validate(cls):
        """Validate that required environment variables are set"""
        if not cls.API_KEY:
            raise ValueError("❌ OPENROUTER_API_KEY not found in .env file!")
        if not os.path.exists(cls.NEPALI_CSV):
            raise FileNotFoundError(f"❌ Nepali CSV file not found: {cls.NEPALI_CSV}")
        if not os.path.exists(cls.ACHHAMI_CSV):
            raise FileNotFoundError(f"❌ Achhami CSV file not found: {cls.ACHHAMI_CSV}")
        if not cls.REFERRER.startswith(('http://', 'https://')):
            raise ValueError("❌ OPENROUTER_REFERRER must start with http:// or https://")
        
        print("✓ Configuration validated successfully")
        print(f"  - API Key: {cls.API_KEY[:10]}...{cls.API_KEY[-4:]}")
        print(f"  - Models to test: {len(cls.MODELS_TO_TEST)}")
        for i, model in enumerate(cls.MODELS_TO_TEST, 1):
            print(f"    {i}. {model}")
        print(f"  - Nepali CSV: {cls.NEPALI_CSV}")
        print(f"  - Achhami CSV: {cls.ACHHAMI_CSV}")
        print(f"  - Referrer: {cls.REFERRER}")


def fetch_available_models(api_key: str) -> List[str]:
    """Fetch available model IDs from OpenRouter to avoid 404s"""

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": Config.REFERRER,
        "X-Title": Config.APP_TITLE,
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'data' in data:
            return [m.get('id') for m in data['data'] if m.get('id')]
        return []
    except Exception as e:
        print(f"⚠️  Could not fetch model list: {e}")
        return []

# ============================================
# 1. DATA LOADING
# ============================================

def parse_entities_from_text(entities_text: str) -> Dict:
    """Parse entities from CSV format: 'entity1 (TYPE1), entity2 (TYPE2)'"""
    if pd.isna(entities_text) or entities_text.strip() == "":
        return {"entities": []}
    
    entities = []
    # Pattern to match: text (TYPE)
    pattern = r'([^(,]+)\s*\(([^)]+)\)'
    matches = re.findall(pattern, entities_text)
    
    for text, entity_type in matches:
        entities.append({
            "text": text.strip(),
            "type": entity_type.strip()
        })
    
    return {"entities": entities}


def normalize_entity_text(text: str) -> str:
    """Normalize entity surface form for matching"""
    if text is None:
        return ""
    cleaned = text.strip().lower()
    cleaned = cleaned.strip("।")
    cleaned = re.sub(r'^[\W_]+', '', cleaned)
    cleaned = re.sub(r'[\W_]+$', '', cleaned)
    return cleaned

def load_dataset(csv_file: str) -> Tuple[List[str], List[Dict]]:
    """Load dataset from CSV file"""
    print(f"📂 Loading {csv_file}...")
    
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Column A: Sentences
    sentences = df.iloc[:, 0].tolist()
    
    # Column B: Entities
    entities_raw = df.iloc[:, 1].tolist()
    ground_truth = [parse_entities_from_text(ent) for ent in entities_raw]
    
    print(f"  ✓ Loaded {len(sentences)} sentences")
    
    # Print sample
    print(f"  Sample: {sentences[0][:50]}...")
    print(f"  Entities: {ground_truth[0]}")
    
    return sentences, ground_truth

# ============================================
# 2. OPENROUTER NER
# ============================================

class OpenRouterNER:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def extract_entities(self, text: str, language: str = "Nepali") -> Dict:
        """Extract named entities using OpenRouter API"""
        
        few_shot_examples = """
Examples (follow exactly):
1) Text: प्रधानमन्त्री केपी शर्मा ओलीले सिंहदरबारमा नयाँ आर्थिक नीति प्रस्तुत गरे।
Expected JSON: {"entities": [{"text": "केपी शर्मा ओली", "type": "PER"}, {"text": "सिंहदरबार", "type": "LOC"}]}

2) Text: राम र सीता काठमाडौँस्थित त्रिभुवन अन्तर्राष्ट्रिय विमानस्थल गए।
Expected JSON: {"entities": [{"text": "राम", "type": "PER"}, {"text": "सीता", "type": "PER"}, {"text": "काठमाडौँ", "type": "LOC"}, {"text": "त्रिभुवन अन्तर्राष्ट्रिय विमानस्थल", "type": "LOC"}]}

3) Text: रमाइलाे मेलामा जनकपुर नगरपालिका र अञ्चल अस्पतालले स्टल राखे।
Expected JSON: {"entities": [{"text": "जनकपुर नगरपालिका", "type": "ORG"}, {"text": "अञ्चल अस्पताल", "type": "ORG"}]}
"""

        prompt = f"""You are a Named Entity Recognition expert for {language} language.

Extract ALL named entities from this text and classify them:
- PER: people
- LOC: places
- ORG: organizations/institutions
- DATE: dates/times
- MISC: other named entities

Rules:
- Only include entities that appear verbatim in the text; do not guess or hallucinate.
- If no entities, return {{"entities": []}}.
- Output must be valid JSON on a single line. No markdown, no code fences, no explanations.

{few_shot_examples}

Now process this text:
{text}
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.REFERRER,
            "X-Title": Config.APP_TITLE,
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        
        for attempt in range(1, Config.API_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=Config.API_TIMEOUT
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Clean JSON from markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                entities_data = json.loads(content)
                return entities_data
                
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 'N/A'
                body = e.response.text[:500] if e.response is not None else ''
                print(f"  ⚠️  API HTTP Error {status}: {e}")
                if status == 404:
                    print("  ↳ Check model name, endpoint, and OpenRouter access. See response snippet below.")
                print(f"  ↳ Response snippet: {body}")
                # Do not retry on 4xx except maybe 429; keep simple here.
                return {"entities": [], "error": f"HTTP {status}: {body}"}
            except requests.exceptions.RequestException as e:
                # Retry on network/timeout issues
                if attempt < Config.API_MAX_RETRIES:
                    sleep_for = Config.API_RETRY_BACKOFF ** (attempt - 1)
                    print(f"  ↻ Retry {attempt}/{Config.API_MAX_RETRIES} after error: {e}. Sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
                    continue
                print(f"  ⚠️  API Error after retries: {e}")
                return {"entities": [], "error": str(e)}
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON Parse Error: {e}")
                print(f"  Response content: {content[:200] if 'content' in locals() else 'N/A'}")
                return {"entities": [], "error": f"JSON parse error: {e}"}
            except Exception as e:
                print(f"  ⚠️  Unexpected Error: {e}")
                return {"entities": [], "error": str(e)}

# ============================================
# 3. PROCESS DATASETS
# ============================================

def process_dataset(
    sentences: List[str],
    ground_truth: List[Dict],
    ner_model: OpenRouterNER,
    language: str,
    output_file: str,
    api_delay: float = 1.0
) -> pd.DataFrame:
    """Process all sentences through NER model"""
    
    results = []
    total = len(sentences)
    
    print(f"\n🔄 Processing {total} sentences ({language})...")
    
    for idx, (sentence, gt) in enumerate(zip(sentences, ground_truth)):
        # Progress indicator
        print(f"  [{idx + 1}/{total}] {sentence[:60]}...", end='')
        
        # Get predictions
        predicted = ner_model.extract_entities(sentence, language)
        
        # Store results
        results.append({
            'sentence_id': idx + 1,
            'text': sentence,
            'language': language,
            'ground_truth': json.dumps(gt, ensure_ascii=False),
            'predicted': json.dumps(predicted, ensure_ascii=False),
            'gt_entities': gt,
            'pred_entities': predicted
        })
        
        print(f" ✓ ({len(predicted.get('entities', []))} entities)")
        
        # Save progress checkpoint every 10 sentences
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{output_file}.temp", index=False, encoding='utf-8')
            print(f"  💾 Checkpoint saved ({idx + 1}/{total})")
        
        # Rate limiting
        time.sleep(api_delay)
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Clean up temp file
    temp_file = f"{output_file}.temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"  ✅ Results saved to {output_file}")
    
    return df

# ============================================
# 4. EVALUATION METRICS
# ============================================

def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity types to standard format"""
    type_map = {
        'PERSON': 'PER',
        'LOCATION': 'LOC',
        'ORGANIZATION': 'ORG',
        'ORGANISATIONS': 'ORG',
    }
    return type_map.get(entity_type.upper(), entity_type.upper())

def calculate_entity_level_metrics(results_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive entity-level metrics"""
    
    all_true_labels = []
    all_pred_labels = []
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for idx, row in results_df.iterrows():
        gt_entities = row['gt_entities']['entities']
        pred_entities = row['pred_entities']['entities']
        
        # Normalize entity types
        gt_set = {(normalize_entity_text(e['text']), normalize_entity_type(e['type'])) for e in gt_entities}
        pred_set = {(normalize_entity_text(e['text']), normalize_entity_type(e['type'])) for e in pred_entities}
        
        # True positives: exact match (text + type)
        for gt_text, gt_type in gt_set:
            if (gt_text, gt_type) in pred_set:
                true_positives += 1
                all_true_labels.append(gt_type)
                all_pred_labels.append(gt_type)
            else:
                false_negatives += 1
                all_true_labels.append(gt_type)
                all_pred_labels.append('O')  # Missed
        
        # False positives: predicted but not in ground truth
        for pred_text, pred_type in pred_set:
            if (pred_text, pred_type) not in gt_set:
                false_positives += 1
                all_true_labels.append('O')
                all_pred_labels.append(pred_type)
    
    # Calculate metrics
    total = true_positives + false_positives + false_negatives
    accuracy = true_positives / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confusion matrix
    labels = sorted(list(set(all_true_labels + all_pred_labels)))
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels) if labels else np.array([])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'confusion_matrix': cm,
        'labels': labels,
        'all_true_labels': all_true_labels,
        'all_pred_labels': all_pred_labels
    }

# ============================================
# 5. VISUALIZATION FOR MULTIPLE MODELS
# ============================================

def plot_multi_model_comparison(all_model_metrics: Dict):
    """Create comparison chart for all models"""
    
    models = list(all_model_metrics.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Prepare data
    nepali_data = []
    achhami_data = []
    
    for model in models:
        nepali_metrics = all_model_metrics[model]['nepali']
        achhami_metrics = all_model_metrics[model]['achhami']
        
        nepali_data.append([
            nepali_metrics['accuracy'],
            nepali_metrics['precision'],
            nepali_metrics['recall'],
            nepali_metrics['f1_score']
        ])
        
        achhami_data.append([
            achhami_metrics['accuracy'],
            achhami_metrics['precision'],
            achhami_metrics['recall'],
            achhami_metrics['f1_score']
        ])
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NER Performance Comparison Across Models', fontsize=16, fontweight='bold')
    
    colors_nepali = ['#2E86AB', '#A23B72', '#F18F01']
    colors_achhami = ['#6A9FB5', '#C894B6', '#F5A962']
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        nepali_values = [data[idx] for data in nepali_data]
        achhami_values = [data[idx] for data in achhami_data]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, nepali_values, width, label='Standard Nepali', 
                      color=colors_nepali[:len(models)])
        bars2 = ax.bar(x + width/2, achhami_values, width, label='Achhami Dialect',
                      color=colors_achhami[:len(models)])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1][:15] for m in models], rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.15])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = 'multi_model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

def plot_degradation_comparison(all_model_metrics: Dict):
    """Plot performance degradation across models"""
    
    models = list(all_model_metrics.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    degradations = []
    
    for model in models:
        nepali = all_model_metrics[model]['nepali']
        achhami = all_model_metrics[model]['achhami']
        
        degradations.append([
            (nepali['accuracy'] - achhami['accuracy']) * 100,
            (nepali['precision'] - achhami['precision']) * 100,
            (nepali['recall'] - achhami['recall']) * 100,
            (nepali['f1_score'] - achhami['f1_score']) * 100
        ])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, (model, deg) in enumerate(zip(models, degradations)):
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, deg, width, label=model.split('/')[-1][:20])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Degradation: Standard Nepali → Achhami Dialect', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = 'performance_degradation_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, model_name: str):
    """Plot confusion matrix heatmap"""
    
    if len(labels) == 0 or cm.size == 0:
        print(f"  ⚠️  Skipping confusion matrix for {title} - no data")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    plt.title(f'{title}\n{model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    safe_model = model_name.replace('/', '_')
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    filename = f'{safe_title}_{safe_model}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()

def save_comprehensive_report(all_model_metrics: Dict):
    """Save comprehensive report for all models"""
    
    filename = "comprehensive_model_comparison_report.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE NEPALI NER DIALECTAL ROBUSTNESS EVALUATION\n")
        f.write("Comparing Multiple Models\n")
        f.write("="*80 + "\n\n")
        
        for model_name, metrics in all_model_metrics.items():
            nepali = metrics['nepali']
            achhami = metrics['achhami']
            
            f.write("="*80 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("STANDARD NEPALI RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {nepali['accuracy']:.4f} ({nepali['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {nepali['precision']:.4f} ({nepali['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {nepali['recall']:.4f} ({nepali['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {nepali['f1_score']:.4f} ({nepali['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {nepali['true_positives']}, FP: {nepali['false_positives']}, FN: {nepali['false_negatives']}\n\n")
            
            f.write("ACHHAMI DIALECT RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {achhami['accuracy']:.4f} ({achhami['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {achhami['precision']:.4f} ({achhami['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {achhami['recall']:.4f} ({achhami['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {achhami['f1_score']:.4f} ({achhami['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {achhami['true_positives']}, FP: {achhami['false_positives']}, FN: {achhami['false_negatives']}\n\n")
            
            f.write("PERFORMANCE DEGRADATION\n")
            f.write("-"*80 + "\n")
            acc_diff = (nepali['accuracy'] - achhami['accuracy']) * 100
            prec_diff = (nepali['precision'] - achhami['precision']) * 100
            rec_diff = (nepali['recall'] - achhami['recall']) * 100
            f1_diff = (nepali['f1_score'] - achhami['f1_score']) * 100
            
            f.write(f"Accuracy Decrease:  {acc_diff:+.2f}%\n")
            f.write(f"Precision Decrease: {prec_diff:+.2f}%\n")
            f.write(f"Recall Decrease:    {rec_diff:+.2f}%\n")
            f.write(f"F1-Score Decrease:  {f1_diff:+.2f}%\n\n")
        
        # Summary comparison
        f.write("="*80 + "\n")
        f.write("SUMMARY COMPARISON - STANDARD NEPALI\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<50} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}\n")
        f.write("-"*80 + "\n")
        for model_name, metrics in all_model_metrics.items():
            m = metrics['nepali']
            f.write(f"{model_name:<50} {m['accuracy']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}  {m['f1_score']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY COMPARISON - ACHHAMI DIALECT\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<50} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}\n")
        f.write("-"*80 + "\n")
        for model_name, metrics in all_model_metrics.items():
            m = metrics['achhami']
            f.write(f"{model_name:<50} {m['accuracy']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}  {m['f1_score']:.4f}\n")
    
    print(f"  ✓ Saved: {filename}")

# ============================================
# 6. MAIN EXECUTION FOR MULTIPLE MODELS
# ============================================

def main():
    """Main execution function for multiple models"""
    
    print("="*80)
    print("MULTI-MODEL NEPALI NER DIALECTAL ROBUSTNESS EVALUATION")
    print("="*80)
    print()
    
    # Validate configuration
    try:
        Config.validate()
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease check your .env file and ensure:")
        print("  1. .env file exists in the same directory")
        print("  2. OPENROUTER_API_KEY is set")
        print("  3. CSV files exist at specified paths")
        return
    
    print()

    # Preflight: check which models are currently available to avoid 404s
    print("Checking available OpenRouter models...")
    available_models = fetch_available_models(Config.API_KEY)
    if not available_models:
        print("⚠️  Could not retrieve available models; proceeding with configured list (may 404 if unavailable).")
    else:
        missing = [m for m in Config.MODELS_TO_TEST if m not in available_models]
        if missing:
            print("⚠️  These models are not currently available and will be skipped:")
            for m in missing:
                print(f"    - {m}")
        Config.MODELS_TO_TEST = [m for m in Config.MODELS_TO_TEST if m in available_models]
        if not Config.MODELS_TO_TEST:
            print("❌ No requested models are available on OpenRouter right now. Update Config.MODELS_TO_TEST or try later.")
            return
        print("✓ Models to test after availability check:")
        for m in Config.MODELS_TO_TEST:
            print(f"    - {m}")
    
    # Load datasets once (used for all models)
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    try:
        nepali_sentences, nepali_gt = load_dataset(Config.NEPALI_CSV)
        achhami_sentences, achhami_gt = load_dataset(Config.ACHHAMI_CSV)
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Store results for all models
    all_model_metrics = {}
    
    # Process each model
    for model_idx, model_name in enumerate(Config.MODELS_TO_TEST, 1):
        print("\n" + "="*80)
        print(f"MODEL {model_idx}/{len(Config.MODELS_TO_TEST)}: {model_name}")
        print("="*80)
        
        # Initialize NER model
        print(f"\n🤖 Initializing OpenRouter NER with {model_name}...")
        ner_model = OpenRouterNER(Config.API_KEY, model_name)
        
        # Create safe filename
        safe_model_name = model_name.replace('/', '_')
        
        # Process Standard Nepali
        print("\n" + "-"*80)
        print("PROCESSING STANDARD NEPALI")
        print("-"*80)
        nepali_results = process_dataset(
            nepali_sentences,
            nepali_gt,
            ner_model,
            "Standard Nepali",
            f"nepali_results_{safe_model_name}.csv",
            Config.API_DELAY
        )
        
        # Process Achhami
        print("\n" + "-"*80)
        print("PROCESSING ACHHAMI DIALECT")
        print("-"*80)
        achhami_results = process_dataset(
            achhami_sentences,
            achhami_gt,
            ner_model,
            "Achhami Dialect",
            f"achhami_results_{safe_model_name}.csv",
            Config.API_DELAY
        )
        
        # Calculate metrics
        print(f"\n📊 Calculating metrics for {model_name}...")
        nepali_metrics = calculate_entity_level_metrics(nepali_results)
        achhami_metrics = calculate_entity_level_metrics(achhami_results)
        
        # Store metrics
        all_model_metrics[model_name] = {
            'nepali': nepali_metrics,
            'achhami': achhami_metrics
        }
        
        # Print results for this model
        print("\n" + "-"*80)
        print(f"RESULTS FOR {model_name}")
        print("-"*80)
        print("\nStandard Nepali:")
        print(f"  Accuracy:  {nepali_metrics['accuracy']:.4f} ({nepali_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {nepali_metrics['precision']:.4f} ({nepali_metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {nepali_metrics['recall']:.4f} ({nepali_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {nepali_metrics['f1_score']:.4f} ({nepali_metrics['f1_score']*100:.2f}%)")
        
        print("\nAchhami Dialect:")
        print(f"  Accuracy:  {achhami_metrics['accuracy']:.4f} ({achhami_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {achhami_metrics['precision']:.4f} ({achhami_metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {achhami_metrics['recall']:.4f} ({achhami_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {achhami_metrics['f1_score']:.4f} ({achhami_metrics['f1_score']*100:.2f}%)")
        
        acc_diff = (nepali_metrics['accuracy'] - achhami_metrics['accuracy']) * 100
        print(f"\nPerformance Degradation: {acc_diff:+.2f}%")
        
        # Generate confusion matrices for this model
        print(f"\n📈 Generating confusion matrices for {model_name}...")
        plot_confusion_matrix(
            nepali_metrics['confusion_matrix'], 
            nepali_metrics['labels'],
            "Confusion Matrix - Standard Nepali",
            model_name
        )
        plot_confusion_matrix(
            achhami_metrics['confusion_matrix'],
            achhami_metrics['labels'],
            "Confusion Matrix - Achhami Dialect",
            model_name
        )
    
    # Generate comparative visualizations
    print("\n" + "="*80)
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print("="*80)
    
    plot_multi_model_comparison(all_model_metrics)
    plot_degradation_comparison(all_model_metrics)
    save_comprehensive_report(all_model_metrics)
    
    # Print final summary
    print("\n" + "="*80)
    print("✅ MULTI-MODEL ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📊 SUMMARY COMPARISON:")
    print("-"*80)
    print(f"{'Model':<35} {'Nepali Acc':<12} {'Achhami Acc':<12} {'Degradation':<12}")
    print("-"*80)
    
    for model_name, metrics in all_model_metrics.items():
        nepali_acc = metrics['nepali']['accuracy']
        achhami_acc = metrics['achhami']['accuracy']
        degradation = (nepali_acc - achhami_acc) * 100
        
        model_short = model_name.split('/')[-1][:30]
        print(f"{model_short:<35} {nepali_acc:.4f}       {achhami_acc:.4f}        {degradation:+.2f}%")
    
    print("\n📁 Generated files:")
    print("  Results CSVs:")
    for model_name in Config.MODELS_TO_TEST:
        safe_name = model_name.replace('/', '_')
        print(f"    - nepali_results_{safe_name}.csv")
        print(f"    - achhami_results_{safe_name}.csv")
    
    print("\n  Visualizations:")
    print("    - multi_model_comparison.png")
    print("    - performance_degradation_comparison.png")
    for model_name in Config.MODELS_TO_TEST:
        safe_name = model_name.replace('/', '_')
        print(f"    - confusion_matrix_standard_nepali_{safe_name}.png")
        print(f"    - confusion_matrix_achhami_dialect_{safe_name}.png")
    
    print("\n  Report:")
    print("    - comprehensive_model_comparison_report.txt")

if __name__ == "__main__":
    main()