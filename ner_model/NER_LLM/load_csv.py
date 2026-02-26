# import pandas as pd
# import re
# from typing import List, Dict

# def parse_entities_from_text(entities_text: str) -> Dict:
#     """
#     Parse entities from your CSV format
#     Example: "राम (PER), काठमाडौं (LOC)" 
#     Returns: {"entities": [{"text": "राम", "type": "PER"}, ...]}
#     """
#     if pd.isna(entities_text) or entities_text.strip() == "":
#         return {"entities": []}
    
#     entities = []
#     # Pattern to match: text (TYPE)
#     pattern = r'([^(,]+)\s*\(([^)]+)\)'
#     matches = re.findall(pattern, entities_text)
    
#     for text, entity_type in matches:
#         entities.append({
#             "text": text.strip(),
#             "type": entity_type.strip()
#         })
    
#     return {"entities": entities}

# def load_nepali_dataset(csv_file: str) -> tuple:
#     """
#     Load Nepali dataset from CSV
    
#     Args:
#         csv_file: Path to CSV file
        
#     Returns:
#         Tuple of (sentences, ground_truth_annotations)
#     """
#     # Read CSV
#     df = pd.read_csv(csv_file)
    
#     # Extract sentences (Column A)
#     sentences = df.iloc[:, 0].tolist()  # First column
    
#     # Extract and parse entities (Column B)
#     entities_raw = df.iloc[:, 1].tolist()  # Second column
#     ground_truth = [parse_entities_from_text(ent) for ent in entities_raw]
    
#     return sentences, ground_truth

# # Load your datasets
# nepali_sentences, nepali_ground_truth = load_nepali_dataset('ner.csv')
# achhami_sentences, achhami_ground_truth = load_nepali_dataset('ner_achami - Sheet1.csv')

# print(f"Loaded {len(nepali_sentences)} Nepali sentences")
# print(f"Loaded {len(achhami_sentences)} Achhami sentences")

# # Example: Print first sentence and its entities
# print("\nExample from dataset:")
# print(f"Sentence: {nepali_sentences[0]}")
# print(f"Entities: {nepali_ground_truth[0]}")