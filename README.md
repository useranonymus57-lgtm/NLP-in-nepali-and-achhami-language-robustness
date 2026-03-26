# Dialectal Robustness of Nepali NLP Systems: NER and POS Tagging on the Achhami Dialect

> The first systematic benchmark evaluating Named Entity Recognition (NER) and Part-of-Speech (POS) tagging performance degradation from Standard Nepali to the Achhami dialect.

---

## 📌 Overview

This repository contains the dataset, annotation guidelines, evaluation scripts, and results for our study on dialectal robustness in Nepali NLP systems. We benchmark **5 NER models** and **4 POS tagging models** on a parallel corpus of **300 sentences** in both Standard Nepali and the Achhami dialect of the Far-Western region of Nepal.

```
ner_model/
├── dialect_dataset.csv
├── NER_BERT/
│   ├── Achhami_both.ipynb
│   ├── ner_nepaliNEPBERT.ipynb
│   └── ner_nepalimbert.ipynb
└── NER_LLM/
    ├── achhami_results_anthropic_...
    ├── achhami_results_meta-llama...
    ├── achhami_results_openai_gp...
    ├── complete_ner.py
    ├── comprehensive_model_com...
    ├── confusion_matrix__achha...
    ├── confusion_matrix__achha...
    ├── confusion_matrix__achha...
    ├── confusion_matrix__standar...
    ├── confusion_matrix__standar...
    ├── confusion_matrix__standar...
    ├── load_csv.py
    ├── multi_model_comparison.png
    ├── nepali_results_anthropic_cla...
    ├── nepali_results_meta-llama_ll...
    ├── nepali_results_openai_gpt-4...
    ├── nercsv
    ├── ner_achami - Sheet1.csv
    └── performance_degradation_...
```
```
pos/
├── ack_dataset_clean.csv
├── achham_dataset.csv
├── nep_dataset_clean.csv
├── nepali_dataset.csv
├── nepali_pos.ipynb
├── nepali_pos_llm_eval.py
└── pos_results/
    ├── checkpoint_Claude_3_5_Ha...
    ├── checkpoint_Claude_3_5_Ha...
    ├── checkpoint_GPT_4o_Mini_A...
    ├── checkpoint_GPT_4o_Mini_A...
    ├── checkpoint_Llama_3_1_70...
    ├── checkpoint_Llama_3_1_70...
    ├── confusion_Claude_3_5_Ha...
    ├── confusion_Claude_3_5_Ha...
    ├── confusion_GPT_4o_Mini_A...
    ├── confusion_GPT_4o_Mini_A...
    ├── confusion_Llama_3_1_70...
    ├── confusion_Llama_3_1_70...
    ├── f1_heatmap_all.png
    ├── multi_model_comparison...
    ├── per_tag_f1_Claude_3_5_Ha...
    ├── per_tag_f1_Claude_3_5_Ha...
    ├── per_tag_f1_GPT_4o_Mini_A...
    ├── per_tag_f1_GPT_4o_Mini_A...
    ├── per_tag_f1_Llama_3_1_70...
    ├── per_tag_f1_Llama_3_1_70...
    ├── performance_degradation...
    ├── report.txt
    ├── results_Claude_3_5_Haku...
    ├── results_Claude_3_5_Haku...
    ├── results_GPT_4o_Mini_Nepa...
    ├── results_GPT_4o_Mini_Nepa...
    ├── results_Llama_3_1_70B_Ne...
    ├── results_Llama_3_1_70B_Ne...
    └── results_summary.csv

 ```
## 📊 Dataset

| Property | Details |
|---|---|
| Sentences | 300 parallel sentences |
| Dialects | Standard Nepali, Achhami |
| NER Tags | PER, ORG, LOC, DATE, MISC |
| POS Tags | Universal Dependencies (17 tags) |

---

## 🤖 Models Evaluated

### NER Models
| Model | Type |
|---|---|
| NepBERTa | Monolingual BERT |
| mBERT-Nepali | Multilingual BERT |
| Claude 3.5 Haiku | LLM |
| GPT-4o Mini | LLM |
| Llama 3.1 70B | LLM |

### POS Tagging Models
| Model | Type |
|---|---|
| XLM-RoBERTa UD | Fine-tuned Transformer |
| Claude 3.5 Haiku | LLM |
| GPT-4o Mini | LLM |
| Llama 3.1 70B | LLM |

---

## 📈 Key Results

### NER — Precision, Recall, F1

| Model | Dialect | Precision | Recall | F1 |
|---|---|---|---|---|
| NepBERTa | Nepali | 0.685 | 0.605 | 0.642 |
| NepBERTa | Achhami | 0.652 | 0.584 | 0.616 |
| mBERT-Nepali | Nepali | 0.743 | 0.605 | 0.667 |
| mBERT-Nepali | Achhami | 0.695 | 0.571 | 0.627 |
| Claude 3.5 Haiku | Nepali | 0.881 | 0.901 | 0.891 |
| Claude 3.5 Haiku | Achhami | 0.856 | 0.884 | 0.870 |
| GPT-4o Mini | Nepali | 0.888 | 0.838 | 0.862 |
| GPT-4o Mini | Achhami | 0.839 | 0.815 | 0.827 |
| Llama 3.1 70B | Nepali | 0.860 | 0.811 | 0.835 |
| Llama 3.1 70B | Achhami | 0.832 | 0.788 | 0.810 |

### POS Tagging — Accuracy, Precision, Recall, F1

| Model | Dialect | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XLM-RoBERTa UD | Nepali | 0.790 | 0.653 | 0.682 | 0.642 |
| XLM-RoBERTa UD | Achhami | 0.726 | 0.511 | 0.517 | 0.497 |
| Claude 3.5 Haiku | Nepali | 0.909 | 0.759 | 0.809 | 0.751 |
| Claude 3.5 Haiku | Achhami | 0.904 | 0.720 | 0.735 | 0.722 |
| GPT-4o Mini | Nepali | 0.848 | 0.796 | 0.788 | 0.771 |
| GPT-4o Mini | Achhami | 0.874 | 0.701 | 0.711 | 0.701 |
| Llama 3.1 70B | Nepali | 0.818 | 0.581 | 0.636 | 0.570 |
| Llama 3.1 70B | Achhami | 0.766 | 0.552 | 0.514 | 0.518 |

---

## 🔑 Key Findings

- **Standard Nepali bias is confirmed** across all models and both tasks
- **POS tagging degrades more severely** (2.91%–14.49%) than NER (2.12%–3.97%)
- **Claude 3.5 Haiku** is the most robust model across both tasks
- **NepBERTa** outperforms multilingual mBERT in dialectal robustness despite being monolingual
- **XLM-RoBERTa UD** suffers the largest degradation (14.49% Macro F1 drop) across all models
- Multilingual pre-training **does not inherently** confer dialectal robustness

---



## ⚠️ Ethical Considerations

Our datasets were constructed from publicly accessible news articles. No private or sensitive user data were collected. Personal identifiers beyond named entities required for the NER task were removed during annotation. POS annotation was performed solely on linguistic structure and contains no personally identifiable information. Both datasets are intended solely for research and educational purposes. We acknowledge that automated NLP evaluation on dialectal text carries the risk of reinforcing existing biases against underrepresented linguistic communities if findings are not interpreted with appropriate cultural and sociolinguistic context.
