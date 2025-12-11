#!/usr/bin/env python3
"""
Download datasets for the artificial token language research project.
"""

from datasets import load_dataset
import os

# Create datasets directory
os.makedirs("datasets", exist_ok=True)

print("=" * 80)
print("DOWNLOADING DATASETS FOR ARTIFICIAL TOKEN LANGUAGE RESEARCH")
print("=" * 80)

# 1. WikiText-2 (Small, good for quick experiments)
print("\n[1/4] Downloading WikiText-2 (word-level)...")
try:
    wikitext2 = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    wikitext2.save_to_disk("datasets/wikitext-2-v1")
    print(f"✓ WikiText-2 saved to datasets/wikitext-2-v1")
    print(f"  Train: {len(wikitext2['train'])} examples")
    print(f"  Valid: {len(wikitext2['validation'])} examples")
    print(f"  Test: {len(wikitext2['test'])} examples")
except Exception as e:
    print(f"✗ Error downloading WikiText-2: {e}")

# 2. WikiText-103 (Larger, better for training)
print("\n[2/4] Downloading WikiText-103 (word-level)...")
try:
    wikitext103 = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    wikitext103.save_to_disk("datasets/wikitext-103-v1")
    print(f"✓ WikiText-103 saved to datasets/wikitext-103-v1")
    print(f"  Train: {len(wikitext103['train'])} examples")
    print(f"  Valid: {len(wikitext103['validation'])} examples")
    print(f"  Test: {len(wikitext103['test'])} examples")
except Exception as e:
    print(f"✗ Error downloading WikiText-103: {e}")

# 3. LAMBADA (Language understanding with broad context)
print("\n[3/4] Downloading LAMBADA (EleutherAI version)...")
try:
    lambada = load_dataset("EleutherAI/lambada_openai", "en")
    lambada.save_to_disk("datasets/lambada_openai")
    print(f"✓ LAMBADA saved to datasets/lambada_openai")
    print(f"  Test: {len(lambada['test'])} examples")
except Exception as e:
    print(f"✗ Error downloading LAMBADA: {e}")

# 4. GLUE subset (for downstream task evaluation)
print("\n[4/4] Downloading GLUE MRPC (Microsoft Research Paraphrase Corpus)...")
try:
    glue_mrpc = load_dataset("nyu-mll/glue", "mrpc")
    glue_mrpc.save_to_disk("datasets/glue-mrpc")
    print(f"✓ GLUE MRPC saved to datasets/glue-mrpc")
    print(f"  Train: {len(glue_mrpc['train'])} examples")
    print(f"  Valid: {len(glue_mrpc['validation'])} examples")
    print(f"  Test: {len(glue_mrpc['test'])} examples")
except Exception as e:
    print(f"✗ Error downloading GLUE MRPC: {e}")

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

# Save sample data from each dataset
print("\nSaving sample data for documentation...")
os.makedirs("datasets/samples", exist_ok=True)

try:
    with open("datasets/samples/wikitext-2_sample.txt", "w") as f:
        f.write("WikiText-2 Sample (first 3 training examples):\n\n")
        for i in range(min(3, len(wikitext2['train']))):
            f.write(f"Example {i+1}:\n{wikitext2['train'][i]['text'][:500]}...\n\n")
    print("✓ Saved WikiText-2 sample")
except:
    pass

try:
    with open("datasets/samples/lambada_sample.txt", "w") as f:
        f.write("LAMBADA Sample (first 5 test examples):\n\n")
        for i in range(min(5, len(lambada['test']))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Text: {lambada['test'][i]['text']}\n")
            f.write(f"Target: {lambada['test'][i]['target_word']}\n\n")
    print("✓ Saved LAMBADA sample")
except:
    pass

try:
    with open("datasets/samples/glue_mrpc_sample.txt", "w") as f:
        f.write("GLUE MRPC Sample (first 3 training examples):\n\n")
        for i in range(min(3, len(glue_mrpc['train']))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Sentence 1: {glue_mrpc['train'][i]['sentence1']}\n")
            f.write(f"Sentence 2: {glue_mrpc['train'][i]['sentence2']}\n")
            f.write(f"Label: {glue_mrpc['train'][i]['label']}\n\n")
    print("✓ Saved GLUE MRPC sample")
except:
    pass

print("\nAll datasets ready for experimentation!")
