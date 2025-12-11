# Downloaded Datasets

This directory contains datasets for the artificial token language research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Quick Start

To download all datasets at once, run:
```bash
python download_datasets.py
```

This script will download all datasets listed below and save sample data for documentation.

---

## Dataset 1: WikiText-2 (word-level)

### Overview
- **Source**: https://huggingface.co/datasets/Salesforce/wikitext
- **Size**: ~36K training examples, 3.7K validation, 4.3K test
- **Format**: HuggingFace Dataset
- **Task**: Language modeling, perplexity evaluation
- **License**: Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)

### Description
WikiText-2 is extracted from verified Good and Featured articles on Wikipedia. It's a standard benchmark for language modeling with over 2 million tokens. The word-level version (v1) contains preprocessed tokens suitable for word-level language modeling.

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
dataset.save_to_disk("datasets/wikitext-2-v1")
```

**Alternative (manual download):**
Visit https://huggingface.co/datasets/Salesforce/wikitext and download the wikitext-2-v1 split.

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-2-v1")

# Access splits
train_data = dataset['train']
valid_data = dataset['validation']
test_data = dataset['test']

# Example usage
for example in train_data:
    text = example['text']
    # Process text...
```

### Sample Data

See `samples/wikitext-2_sample.txt` for example records.

### Notes
- Good for quick experiments due to smaller size
- Standard benchmark for language model perplexity evaluation
- Contains Wikipedia-style text with long-form content
- Perplexity on this dataset is a common evaluation metric

---

## Dataset 2: WikiText-103 (word-level)

### Overview
- **Source**: https://huggingface.co/datasets/Salesforce/wikitext
- **Size**: ~1.8M training examples, 3.7K validation, 4.3K test
- **Format**: HuggingFace Dataset
- **Task**: Language modeling, large-scale perplexity evaluation
- **License**: Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)

### Description
WikiText-103 is a larger version of WikiText-2, containing over 100 million tokens extracted from verified Wikipedia articles. It's more suitable for training larger language models and provides a more comprehensive benchmark.

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext-103-v1")
```

**Alternative (manual download):**
Visit https://huggingface.co/datasets/Salesforce/wikitext and download the wikitext-103-v1 split.

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-103-v1")

# Access splits
train_data = dataset['train']
valid_data = dataset['validation']
test_data = dataset['test']
```

### Sample Data

Sample format is identical to WikiText-2 but with more training data.

### Notes
- 50x more training data than WikiText-2
- Better for training larger models
- State-of-the-art perplexity: ~16.4 (as of 2024)
- Widely used benchmark in language modeling research

---

## Dataset 3: LAMBADA (EleutherAI version)

### Overview
- **Source**: https://huggingface.co/datasets/EleutherAI/lambada_openai
- **Size**: 5,153 test examples
- **Format**: HuggingFace Dataset
- **Task**: Word prediction requiring broad discourse context
- **License**: CC BY 4.0

### Description
LAMBADA evaluates the ability of language models to understand long-term contextual dependencies. Each example requires predicting a target word that is easy for humans when reading the full passage but nearly impossible with only the last sentence. Extracted from books to ensure long-term dependencies.

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/lambada_openai", "en")
dataset.save_to_disk("datasets/lambada_openai")
```

**Alternative (manual download):**
Visit https://huggingface.co/datasets/EleutherAI/lambada_openai

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/lambada_openai")

# Access test split
test_data = dataset['test']

# Example usage
for example in test_data:
    text = example['text']  # Full context
    target = example['target_word']  # Word to predict
    domain = example['domain']  # Source domain
```

### Sample Data

See `samples/lambada_sample.txt` for example records showing context and target words.

### Notes
- Test-only dataset (no training split)
- Focuses on long-range dependencies
- Target words are intentionally challenging without full context
- Good benchmark for evaluating contextual understanding
- Multilingual versions available for cross-lingual evaluation

---

## Dataset 4: GLUE MRPC (Microsoft Research Paraphrase Corpus)

### Overview
- **Source**: https://huggingface.co/datasets/nyu-mll/glue
- **Size**: 3,668 training, 408 validation, 1,725 test examples
- **Format**: HuggingFace Dataset
- **Task**: Paraphrase detection (binary classification)
- **License**: Varies by GLUE component

### Description
MRPC is part of the GLUE benchmark for natural language understanding. It consists of sentence pairs extracted from online news sources, with human annotations indicating whether the sentences are semantically equivalent (paraphrases).

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc")
dataset.save_to_disk("datasets/glue-mrpc")
```

**Alternative (manual download):**
Visit https://huggingface.co/datasets/nyu-mll/glue

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/glue-mrpc")

# Access splits
train_data = dataset['train']
valid_data = dataset['validation']
test_data = dataset['test']

# Example usage
for example in train_data:
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    label = example['label']  # 0 = not paraphrase, 1 = paraphrase
```

### Sample Data

See `samples/glue_mrpc_sample.txt` for example sentence pairs with labels.

### Notes
- Part of the widely-used GLUE benchmark
- Good for evaluating downstream task performance
- Binary classification task
- Relatively small dataset, good for fine-tuning evaluation
- Metrics: Accuracy and F1 score

---

## Relevance to Research Hypothesis

These datasets were selected to evaluate different aspects of the artificial token language hypothesis:

### 1. **Language Modeling Efficiency** (WikiText-2, WikiText-103)
- Measure perplexity with different tokenization schemes
- Compare training efficiency (tokens processed, time, memory)
- Evaluate compression rates and sequence lengths

### 2. **Long-Range Understanding** (LAMBADA)
- Test if compact token representations maintain contextual understanding
- Evaluate whether artificial tokens can capture long-term dependencies
- Compare reasoning quality across different tokenization approaches

### 3. **Downstream Task Performance** (GLUE MRPC)
- Verify that alternative tokenization doesn't hurt task-specific performance
- Test transfer learning capabilities
- Evaluate practical applicability of artificial token languages

---

## Dataset Statistics Summary

| Dataset | Train Examples | Valid Examples | Test Examples | Task Type |
|---------|----------------|----------------|---------------|-----------|
| WikiText-2 | 36,718 | 3,760 | 4,358 | Language Modeling |
| WikiText-103 | 1,801,350 | 3,760 | 4,358 | Language Modeling |
| LAMBADA | - | - | 5,153 | Context Prediction |
| GLUE MRPC | 3,668 | 408 | 1,725 | Paraphrase Detection |

---

## Additional Datasets (Not Downloaded)

For future expansion, consider these datasets:

### Large-Scale Pre-training:
- **C4 (Colossal Clean Crawled Corpus)**: 305GB English text from Common Crawl
  - `load_dataset("allenai/c4", "en")`
  - Better for full-scale language model pre-training

### Additional Benchmarks:
- **SuperGLUE**: More challenging successor to GLUE
  - `load_dataset("super_glue", task_name)`
- **Penn Treebank**: Classic language modeling benchmark
- **The Pile**: Diverse 825GB dataset for language modeling

---

## Storage Considerations

**Total downloaded size**: ~350 MB (compressed)

The datasets are excluded from git via `.gitignore` to keep the repository small. All datasets can be re-downloaded using the instructions above or by running `python download_datasets.py`.

---

## Citation Information

If you use these datasets in your research, please cite:

**WikiText:**
```
@misc{merity2016pointer,
    title={Pointer Sentinel Mixture Models},
    author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
    year={2016},
    eprint={1609.07843},
    archivePrefix={arXiv}
}
```

**LAMBADA:**
```
@inproceedings{paperno-etal-2016-lambada,
    title = "The {LAMBADA} dataset: Word prediction requiring a broad discourse context",
    author = "Paperno, Denis and others",
    booktitle = "Proceedings of ACL 2016",
    year = "2016"
}
```

**GLUE:**
```
@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex and others",
    booktitle = "Proceedings of the 2018 EMNLP Workshop BlackboxNLP",
    year = "2018"
}
```
