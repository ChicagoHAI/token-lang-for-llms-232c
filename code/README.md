# Cloned Repositories

This directory contains code repositories relevant to the artificial token language research project.

## Repository 1: Byte Latent Transformer (BLT)

- **URL**: https://github.com/facebookresearch/blt
- **Purpose**: Official implementation of the Byte Latent Transformer paper
- **Location**: code/blt/
- **Key files**:
  - `blt/models/blt.py` - Core BLT model implementation
  - `blt/configs/` - Model configurations
  - `pretrain_blt.py` - Pre-training script
  - `eval_blt.py` - Evaluation script

### Description
BLT is a tokenizer-free architecture that learns from raw byte data. It encodes bytes into dynamically sized patches based on entropy, allocating more compute where data complexity demands it. The repository includes pre-trained weights on HuggingFace for BLT 1B and 7B models.

### Key Features
- Dynamic patch segmentation based on byte entropy
- No fixed vocabulary - maps arbitrary byte groups to latent representations
- Matches tokenization-based model performance at scale
- Significant improvements in inference efficiency and robustness

### Usage Notes
- Requires PyTorch and standard ML libraries
- Pre-trained models available on HuggingFace
- Can be adapted for custom tokenizer-free architectures
- See `README.md` in the repo for detailed setup

### Relevance to Research
This is directly relevant for implementing and experimenting with tokenizer-free approaches. The BLT architecture provides a concrete alternative to traditional token-based language models and demonstrates that byte-level modeling can be competitive.

---

## Repository 2: Language Model Evaluation Harness

- **URL**: https://github.com/EleutherAI/lm-evaluation-harness
- **Purpose**: Unified framework for evaluating language models on multiple benchmarks
- **Location**: code/lm-evaluation-harness/
- **Key files**:
  - `lm_eval/` - Core evaluation framework
  - `lm_eval/tasks/` - Task definitions (WikiText, LAMBADA, GLUE, etc.)
  - `lm_eval/models/` - Model adapters for different architectures
  - `lm_eval/api/` - API for custom task creation

### Description
The LM Evaluation Harness is the backend for Hugging Face's Open LLM Leaderboard and is widely used by organizations like NVIDIA, Cohere, and BigScience. It provides a standardized way to evaluate language models across hundreds of tasks.

### Key Features
- Supports 400+ evaluation tasks out of the box
- Includes WikiText, LAMBADA, GLUE, SuperGLUE, and many more
- Few-shot and zero-shot evaluation capabilities
- Custom task creation framework
- Multi-model comparison
- Perplexity, accuracy, F1, and other metrics

### Usage Notes
- Install with: `cd lm-evaluation-harness && pip install -e .`
- Basic usage: `lm_eval --model hf --model_args pretrained=model_name --tasks task_name`
- Supports HuggingFace models, OpenAI API, and custom model implementations
- See documentation in `docs/` directory

### Relevance to Research
Essential for standardized evaluation of models trained with different tokenization schemes. This will allow us to:
- Compare perplexity across different tokenizers on WikiText
- Evaluate context understanding on LAMBADA
- Test downstream performance on GLUE tasks
- Ensure reproducible, standardized benchmarking

---

## Repository 3: SentencePiece Tokenizer

- **URL**: https://github.com/google/sentencepiece
- **Purpose**: Unsupervised text tokenizer supporting BPE, Unigram, and other algorithms
- **Location**: code/sentencepiece/
- **Key files**:
  - `src/` - C++ implementation of tokenization algorithms
  - `python/` - Python bindings
  - `doc/` - Documentation and tutorials
  - `doc/experiments.md` - Comparison of tokenization algorithms

### Description
SentencePiece is Google's implementation of unsupervised tokenization, supporting multiple algorithms including BPE (Byte Pair Encoding), Unigram language model, character-level, and word-level tokenization. It's designed to be language-agnostic and end-to-end trainable.

### Key Features
- Multiple tokenization algorithms: BPE, Unigram, Char, Word
- Language-agnostic (no language-specific preprocessing needed)
- Pure end-to-end system
- Subword regularization for better generalization
- Supports custom vocabulary sizes
- Direct training from raw text

### Usage Notes
- Install with: `cd sentencepiece && pip install .`
- Train tokenizer: `spm_train --input=data.txt --model_prefix=tokenizer --vocab_size=8000 --model_type=bpe`
- Use in Python:
  ```python
  import sentencepiece as spm
  sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
  tokens = sp.encode('hello world', out_type=str)
  ```

### Relevance to Research
Critical for implementing and comparing different tokenization baselines:
- Train custom tokenizers optimized for compression
- Compare BPE vs. Unigram vs. custom approaches
- Experiment with different vocabulary sizes
- Serve as baseline for comparing artificial token languages
- Analyze tokenization efficiency metrics

---

## Additional Useful Repositories (Not Cloned)

For future reference, these repositories may also be useful:

### Tokenization Research
- **HuggingFace Tokenizers**: https://github.com/huggingface/tokenizers
  - Fast implementations of BPE, WordPiece, and Unigram
  - Rust-based with Python bindings
  - Industry standard for transformer models

- **tiktoken**: https://github.com/openai/tiktoken
  - OpenAI's fast BPE tokenizer
  - Used in GPT-3.5 and GPT-4
  - Good reference for efficient implementation

### Language Model Training
- **nanoGPT**: https://github.com/karpathy/nanoGPT
  - Minimal, educational GPT implementation
  - Good starting point for custom tokenizer experiments
  - Clear, readable code for understanding transformers

- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
  - Large-scale transformer training
  - For serious pre-training experiments
  - Supports model parallelism

### Compression and Efficiency
- **Transformer Efficiency Toolkit**: Various repos focused on model compression
- **Knowledge Distillation**: DistilBERT, TinyBERT implementations

---

## Quick Setup Guide

To set up all repositories for development:

```bash
# Navigate to code directory
cd code

# Install BLT
cd blt
pip install -r requirements.txt
cd ..

# Install LM Evaluation Harness
cd lm-evaluation-harness
pip install -e .
cd ..

# Install SentencePiece
cd sentencepiece
pip install .
cd ..
```

## Recommended Workflow

For the artificial token language research:

1. **Baseline Tokenization** (SentencePiece):
   - Train BPE and Unigram tokenizers on datasets
   - Measure compression rates and vocabulary efficiency
   - Establish baseline performance

2. **Alternative Approaches** (BLT):
   - Experiment with byte-level, tokenizer-free models
   - Compare dynamic patching vs. fixed tokenization
   - Analyze computational efficiency

3. **Evaluation** (LM Evaluation Harness):
   - Standardized benchmarking across all approaches
   - Compare perplexity, accuracy, and task performance
   - Generate reproducible results

---

## Notes

- All repositories are cloned with `--depth 1` to save space (shallow clone)
- Full git history not available; re-clone without `--depth 1` if needed
- Check each repo's LICENSE file before using code in publications
- Star the repositories on GitHub to support the projects!

---

## Citation Information

**Byte Latent Transformer:**
```
@article{pagnoni2024byte,
  title={Byte Latent Transformer: Patches Scale Better Than Tokens},
  author={Pagnoni, Artidoro and others},
  journal={arXiv preprint arXiv:2412.09871},
  year={2024}
}
```

**LM Evaluation Harness:**
```
@software{eval-harness,
  author = {Gao, Leo and others},
  title = {Language Model Evaluation Harness},
  year = {2024},
  publisher = {Zenodo},
  url = {https://github.com/EleutherAI/lm-evaluation-harness}
}
```

**SentencePiece:**
```
@inproceedings{kudo-richardson-2018-sentencepiece,
  title = "{S}entence{P}iece: A simple and language independent approach to subword regularization",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of EMNLP 2018",
  year = "2018"
}
```
