"""
Precompute Semantic Embeddings for Action Labels

This script generates semantic embeddings for action labels using transformer models.
It parses action descriptions and converts them to structured format before encoding.

Supported models:
- sentence-transformers/all-mpnet-base-v2 (default, 768-dim)
- sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- BAAI/bge-large-en-v1.5 (1024-dim)

Output: PyTorch .pt file with label-to-embedding mapping
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================================
# Action Mapping Loading
# ============================================================================

def load_action_mapping(mapping_file: Path) -> Dict[str, str]:
    """
    Load action label-to-description mapping from file
    
    Expected format:
        label1 "description text 1"
        label2 "description text 2"
        ...
    
    Args:
        mapping_file: Path to mapping file
        
    Returns:
        Dictionary mapping labels to descriptions
    """
    action_mapping: Dict[str, str] = {}
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '"' in line:
                # Split on first occurrence of ' "'
                parts = line.split(' "', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    description = parts[1].rstrip('"')
                    action_mapping[label] = description
    
    return action_mapping


# ============================================================================
# Description Parsing Utilities
# ============================================================================

def remove_articles(text: str) -> str:
    """Remove articles ('the', 'a', 'an') from the beginning of text"""
    words = text.split()
    if len(words) > 0 and words[0].lower() in ['the', 'a', 'an']:
        return ' '.join(words[1:])
    return text


def extract_action_verb_from_desc(description: str) -> str:
    """
    Extract action verb (first word) from description
    
    Example: "pour liquid into cup" → "pour"
    """
    words = description.strip().split()
    if len(words) > 0:
        return words[0].lower()
    return "none"


def extract_manipulated_object_from_desc(description: str) -> str:
    """
    Extract manipulated object between action verb and 'into'/'onto'
    
    Example: "pour the liquid into cup" → "liquid"
    """
    description_lower = description.lower()
    
    # Find the action verb (first word)
    words = description_lower.split()
    if len(words) < 2:
        return "none"
    
    # Start position after the first word
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
    """
    Extract target object after 'into'/'onto' and before 'using'
    
    Example: "pour liquid into the cup using spoon" → "cup"
    """
    description_lower = description.lower()
    
    # Find 'into' or 'onto'
    into_pos = description_lower.find(' into ')
    onto_pos = description_lower.find(' onto ')
    
    # Determine which preposition is used
    if into_pos != -1 and onto_pos != -1:
        start_pos = min(into_pos, onto_pos)
        prep_len = 6  # Length of ' into ' or ' onto '
    elif into_pos != -1:
        start_pos = into_pos
        prep_len = 6
    elif onto_pos != -1:
        start_pos = onto_pos
        prep_len = 6
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
    """
    Extract tool after 'using'
    
    Example: "pour liquid into cup using the spoon" → "spoon"
    """
    description_lower = description.lower()
    
    using_pos = description_lower.find(' using ')
    if using_pos != -1:
        # Extract everything after 'using '
        tool_part = description[using_pos + 7:].strip()  # +7 for ' using '
        return remove_articles(tool_part)
    else:
        # No tool mentioned
        return "none"


# ============================================================================
# Structured Description Conversion
# ============================================================================

def convert_to_structured_description(label: str, description: str) -> str:
    """
    Convert natural language description to structured format
    
    Format: "The action is to {verb}. The manipulated object is {object}. 
             The target object is {target}. The tool used is {tool}."
    
    Args:
        label: Action label
        description: Natural language description
        
    Returns:
        Structured description string
    """
    # Handle special cases
    if label == 'null':
        return ("The action is transitional with no active manipulation. "
                "No object is being manipulated. No target is involved. "
                "No tool is being used.")
    
    elif label == 'w':
        return ("The action is incorrect or wrong. The manipulated object is unclear. "
                "The target object is unclear. The tool usage is incorrect.")
    
    # Parse the natural language description
    action_verb = extract_action_verb_from_desc(description)
    manipulated_obj = extract_manipulated_object_from_desc(description)
    target_obj = extract_target_object_from_desc(description)
    tool = extract_tool_from_desc(description)
    
    return (f"The action is to {action_verb}. "
            f"The manipulated object is {manipulated_obj}. "
            f"The target object is {target_obj}. "
            f"The tool used is {tool}.")


# ============================================================================
# Embedding Generation
# ============================================================================

def generate_embeddings(action_mapping: Dict[str, str],
                       semantic_model_name: str,
                       special_labels: list = ['null', 'w']) -> Dict[str, torch.Tensor]:
    """
    Generate semantic embeddings for all action labels
    
    Args:
        action_mapping: Dictionary mapping labels to descriptions
        semantic_model_name: HuggingFace model name
        special_labels: Additional special labels to include
        
    Returns:
        Dictionary mapping labels to embedding tensors
    """
    print(f"Loading tokenizer and model: {semantic_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    model = AutoModel.from_pretrained(semantic_model_name)
    model.eval()
    
    # Collect all labels to encode
    labels_to_encode = set(action_mapping.keys())
    labels_to_encode.update(special_labels)
    
    label_to_embedding: Dict[str, torch.Tensor] = {}
    
    print(f"Generating embeddings for {len(labels_to_encode)} labels...")
    
    with torch.no_grad():
        for label in sorted(labels_to_encode):
            # Convert to structured description
            if label in action_mapping:
                description = convert_to_structured_description(label, action_mapping[label])
            else:
                # Fallback for special labels not in mapping
                description = convert_to_structured_description(label, "")
            
            # Tokenize and encode
            inputs = tokenizer(
                description,
                return_tensors='pt',
                max_length=64,
                truncation=True,
                padding=True
            )
            
            # Generate embedding (mean pooling over hidden states)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [D]
            label_to_embedding[label] = embedding.cpu()
            
            print(f"  ✓ {label:<15} → {description[:60]}...")
    
    return label_to_embedding


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precompute semantic embeddings for action labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to dataset root directory (should contain action_mapping.txt)'
    )
    
    parser.add_argument(
        '--mapping_file',
        type=str,
        default='action_mapping.txt',
        help='Name of the mapping file (relative to data_root)'
    )
    
    parser.add_argument(
        '--semantic_model_name',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        choices=[
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'BAAI/bge-large-en-v1.5'
        ],
        help='HuggingFace model name for semantic encoding'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save .pt file (default: data_root/semantic_embeddings/{model_name}.pt)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    mapping_file = data_root / args.mapping_file
    
    assert data_root.exists(), f"Data root not found: {data_root}"
    assert mapping_file.exists(), f"Mapping file not found: {mapping_file}"
    
    # Determine output path
    if args.output_path is None:
        out_dir = data_root / 'semantic_embeddings'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Derive filename from model name
        safe_name = args.semantic_model_name.replace('/', '_')
        output_path = out_dir / f'{safe_name}.pt'
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Semantic Embedding Precomputation")
    print("=" * 70)
    print(f"Data root:     {data_root}")
    print(f"Mapping file:  {mapping_file}")
    print(f"Model:         {args.semantic_model_name}")
    print(f"Output:        {output_path}")
    print("=" * 70)
    print()
    
    # Load action mapping
    print(f"Loading action mapping from {mapping_file}...")
    action_mapping = load_action_mapping(mapping_file)
    print(f"✓ Loaded {len(action_mapping)} action labels\n")
    
    # Generate embeddings
    label_to_embedding = generate_embeddings(action_mapping, args.semantic_model_name)
    
    # Prepare metadata
    meta = {
        'semantic_model_name': args.semantic_model_name,
        'num_labels': len(label_to_embedding),
        'embedding_dim': label_to_embedding[list(label_to_embedding.keys())[0]].shape[0],
        'data_root': str(data_root),
        'mapping_file': str(mapping_file)
    }
    
    # Save to disk
    save_obj = {
        'embeddings': label_to_embedding,
        'meta': meta
    }
    
    torch.save(save_obj, str(output_path))
    
    print()
    print("=" * 70)
    print(f"✓ Saved {len(label_to_embedding)} embeddings to {output_path}")
    print(f"  Embedding dimension: {meta['embedding_dim']}")
    print("=" * 70)


if __name__ == '__main__':
    main()

