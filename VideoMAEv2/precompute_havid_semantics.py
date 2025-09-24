import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModel


def load_action_mapping(mapping_file: Path) -> Dict[str, str]:
    action_mapping: Dict[str, str] = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '"' in line:
                parts = line.split(' "', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    description = parts[1].rstrip('"')
                    action_mapping[label] = description
    return action_mapping

def convert_to_structured_description(label: str, description: str) -> str:
        """Convert natural language description to structured format by parsing the description directly"""
        # Handle special cases first
        if label == 'null':
            return "The action is transitional with no active manipulation. No object is being manipulated. No target is involved. No tool is being used."
        elif label == 'w':
            return "The action is incorrect or wrong. The manipulated object is unclear. The target object is unclear. The tool usage is incorrect."
        
        # Parse the natural language description
        action_verb = extract_action_verb_from_desc(description)
        manipulated_obj = extract_manipulated_object_from_desc(description)
        target_obj = extract_target_object_from_desc(description)
        tool = extract_tool_from_desc(description)
        
        return f"The action is to {action_verb}. The manipulated object is {manipulated_obj}. The target object is {target_obj}. The tool used is {tool}."
    
def extract_action_verb_from_desc(description: str) -> str:
    """Extract action verb as the first word of the description"""
    words = description.strip().split()
    if len(words) > 0:
        return words[0].lower()
    return "none"
    
def extract_manipulated_object_from_desc(description: str) -> str:
    """Extract manipulated object between action verb and 'into'/'onto'"""
    description_lower = description.lower()
        
    # Find the action verb (first word)
    words = description_lower.split()
    if len(words) < 2:
        return "none"
        
    # Find the start position after the first word
    start_pos = len(words[0]) + 1  # +1 for the space
        
    # Find 'into' or 'onto'
    into_pos = description_lower.find(' into ')
    onto_pos = description_lower.find(' onto ')
        
    # Use whichever comes first (or exists)
    if into_pos != -1 and onto_pos != -1:
        end_pos = min(into_pos, onto_pos)
    elif into_pos != -1:
        end_pos = into_pos
    elif onto_pos != -1:
        end_pos = onto_pos
    else:
        # No 'into' or 'onto' found, take everything after the verb
        return remove_articles(description[start_pos:].strip())
        
    # Extract the manipulated object
    manipulated_part = description[start_pos:end_pos].strip()
    return remove_articles(manipulated_part)
    
def extract_target_object_from_desc(description: str) -> str:
    """Extract target object after 'into'/'onto' and before 'using'"""
    description_lower = description.lower()
        
    # Find 'into' or 'onto'
    into_pos = description_lower.find(' into ')
    onto_pos = description_lower.find(' onto ')
        
    # Determine which preposition is used
    if into_pos != -1 and onto_pos != -1:
        start_pos = min(into_pos, onto_pos)
        prep_len = 6 if (into_pos <= onto_pos) else 6  # ' into ' or ' onto '
    elif into_pos != -1:
        start_pos = into_pos
        prep_len = 6  # ' into '
    elif onto_pos != -1:
        start_pos = onto_pos  
        prep_len = 6  # ' onto '
    else:
        return "none"
        
    # Start after the preposition
    target_start = start_pos + prep_len
        
    # Find 'using' if it exists
    using_pos = description_lower.find(' using ', target_start)
        
    if using_pos != -1:
        # Extract between preposition and 'using'
        target_part = description[target_start:using_pos].strip()
    else:
        # Extract everything after the preposition
        target_part = description[target_start:].strip()
        
    return remove_articles(target_part)
    
def extract_tool_from_desc(description: str) -> str:
    """Extract tool after 'using'"""
    description_lower = description.lower()
        
    using_pos = description_lower.find(' using ')
    if using_pos != -1:
        # Extract everything after 'using '
        tool_part = description[using_pos + 7:].strip()  # +7 for ' using '
        return remove_articles(tool_part)
    else:
        # No tool mentioned, assume hand manipulation
        return "none"
    
def remove_articles(text: str) -> str:
    """Remove 'the', 'a', 'an' from the beginning of text"""
    words = text.split()
    if len(words) > 0 and words[0].lower() in ['the', 'a', 'an']:
        return ' '.join(words[1:])
    return text


def main():
    parser = argparse.ArgumentParser(description="Precompute and save HAVID label semantic embeddings")
    parser.add_argument('--data_root', type=str, default='/home/hao/Polyphony/data/havid', help='Path to data/havid')
    parser.add_argument('--semantic_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save .pt file with embeddings')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    mapping_file = data_root / 'havid_description.txt'
    assert mapping_file.exists(), f"Mapping file not found: {mapping_file}"

    output_path = args.output_path
    if output_path is None:
        out_dir = data_root / 'semantic_embeddings'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Derive filename from model name
        safe_name = args.semantic_model_name.replace('/', '_')
        output_path = out_dir / f'{safe_name}.pt'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading action mapping from {mapping_file}")
    action_mapping = load_action_mapping(mapping_file)

    print(f"Loading tokenizer and model: {args.semantic_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.semantic_model_name)
    model = AutoModel.from_pretrained(args.semantic_model_name)
    model.eval()

    labels_to_encode = set(action_mapping.keys())
    # Ensure special labels are included
    labels_to_encode.update(['null', 'w'])

    label_to_embedding: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for label in sorted(labels_to_encode):
            description = convert_to_structured_description(label, action_mapping[label])

            # if label in action_mapping:
            #     description = action_mapping[label]
            # elif label == 'null':
            #     description = 'no action or transition state'
            # elif label == 'w':
            #     description = 'wrong or incorrect action'
            # else:
            #     description = f'unknown action {label}'

            inputs = tokenizer(
                description,
                return_tensors='pt',
                max_length=64,
                truncation=True,
                padding=True
            )
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [D]
            label_to_embedding[label] = embedding.cpu()

    meta = {
        'semantic_model_name': args.semantic_model_name,
        'num_labels': len(label_to_embedding),
    }

    save_obj = {
        'embeddings': label_to_embedding,
        'meta': meta,
    }

    torch.save(save_obj, str(output_path))
    print(f"Saved {len(label_to_embedding)} label embeddings to {output_path}")


if __name__ == '__main__':
    main()


