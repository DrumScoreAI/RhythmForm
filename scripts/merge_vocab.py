import json
import os

project_root = os.environ.get('RHYTHMFORM_HOME', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open(os.path.join(project_root, 'training_data/tokenizer_vocab.json'), 'r') as f:
    orig_vocab = json.load(f)
with open(os.path.join(project_root, 'training_data/fine_tuning/finetune_tokenizer_vocab.json'), 'r') as f:
    finetune_vocab = json.load(f)

# Use an ordered set approach
seen = set(orig_vocab)
merged_vocab = orig_vocab[:]
for token in finetune_vocab:
    if token not in seen:
        merged_vocab.append(token)
        seen.add(token)

with open(os.path.join(project_root, 'training_data/merged_tokenizer_vocab.json'), 'w') as f:
    json.dump(merged_vocab, f, indent=2)

print(f"Merged vocab size: {len(merged_vocab)}")
