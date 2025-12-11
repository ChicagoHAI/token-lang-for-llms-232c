# Resources Catalog

This document catalogs all resources gathered for the "Artificial Token Language for More Efficient LLMs" research project, including papers, datasets, and code repositories.

---

## Summary

**Resource Gathering Completed**: December 11, 2025

**Total Resources Collected**:
- **Papers**: 8 research papers
- **Datasets**: 4 datasets (WikiText-2, WikiText-103, LAMBADA, GLUE MRPC)
- **Code Repositories**: 3 repositories (BLT, LM Evaluation Harness, SentencePiece)

All resources are organized in their respective directories with comprehensive documentation.

---

## Papers

**Total Papers Downloaded**: 8

| Paper Title | Authors/Source | Year | File | Key Relevance |
|-------------|----------------|------|------|---------------|
| Byte Latent Transformer | Meta AI | 2024 | [2412.09871_byte_latent_transformer.pdf](papers/2412.09871_byte_latent_transformer.pdf) | Tokenizer-free architecture, 8B params tested |
| Tokenization Matters! | arXiv | 2024 | [2405.17067_tokenization_matters.pdf](papers/2405.17067_tokenization_matters.pdf) | Shows tokenization as inherent LLM limitation |
| ReTok | arXiv | 2024 | [2410.04335_retok.pdf](papers/2410.04335_retok.pdf) | High-compression tokenizers improve efficiency |
| Theory of Tokenization | arXiv | 2025 | [2404.08335_theory_tokenization.pdf](papers/2404.08335_theory_tokenization.pdf) | Renyi efficiency, theoretical framework |
| Language Modeling is Compression | arXiv | 2023 | [2309.10668_language_modeling_compression.pdf](papers/2309.10668_language_modeling_compression.pdf) | Compression-performance connection |
| Length-MAX Tokenizer | arXiv | 2024 | [2511.20849_length_max_tokenizer.pdf](papers/2511.20849_length_max_tokenizer.pdf) | 18.5% fewer training steps, 13.7% faster inference |
| ARC-Encoder | arXiv | 2024 | [2510.20535_arc_encoder.pdf](papers/2510.20535_arc_encoder.pdf) | 4-8x compression via learned representations |
| BPE is Suboptimal | EMNLP 2020 | 2020 | [2020.findings-emnlp.414_bpe_suboptimal.pdf](papers/2020.findings-emnlp.414_bpe_suboptimal.pdf) | Unigram LM outperforms BPE |

### Key Themes

1. **Tokenization Efficiency**: Multiple approaches to improving tokens/byte ratio
2. **Alternative Architectures**: Byte-level (BLT) and continuous (ARC) representations
3. **Theoretical Foundations**: Information theory, compression, Renyi efficiency
4. **Concrete Improvements**: 14-18% token reduction, 18.5% training speedup possible

**Detailed Information**: See [papers/README.md](papers/README.md)

---

## Datasets

**Total Datasets Downloaded**: 4

| Dataset Name | Source | Size | Task | Location | Download Status |
|--------------|--------|------|------|----------|-----------------|
| WikiText-2 | HuggingFace/Salesforce | 36K train, 4K test | Language Modeling | datasets/wikitext-2-v1/ | ✓ Downloaded |
| WikiText-103 | HuggingFace/Salesforce | 1.8M train, 4K test | Language Modeling | datasets/wikitext-103-v1/ | ✓ Downloaded |
| LAMBADA | HuggingFace/EleutherAI | 5K test | Context Prediction | datasets/lambada_openai/ | ✓ Downloaded |
| GLUE MRPC | HuggingFace/NYU | 3.6K train, 1.7K test | Paraphrase Detection | datasets/glue-mrpc/ | ✓ Downloaded |

### Dataset Purpose

- **WikiText-2**: Quick iteration, perplexity baseline
- **WikiText-103**: Standard benchmark, main evaluation
- **LAMBADA**: Long-range dependency testing
- **GLUE MRPC**: Downstream task performance

### Download Instructions

All datasets can be re-downloaded by running:
```bash
python download_datasets.py
```

Individual datasets can be loaded using HuggingFace datasets library as documented in [datasets/README.md](datasets/README.md).

**Important Note**: Dataset files are excluded from git via `.gitignore` to keep repository size small. Download instructions are provided for reproducibility.

**Detailed Information**: See [datasets/README.md](datasets/README.md)

---

## Code Repositories

**Total Repositories Cloned**: 3

| Repository Name | URL | Purpose | Location | Clone Status |
|-----------------|-----|---------|----------|--------------|
| BLT (Byte Latent Transformer) | [github.com/facebookresearch/blt](https://github.com/facebookresearch/blt) | Tokenizer-free architecture implementation | code/blt/ | ✓ Cloned |
| LM Evaluation Harness | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standardized LLM evaluation framework | code/lm-evaluation-harness/ | ✓ Cloned |
| SentencePiece | [github.com/google/sentencepiece](https://github.com/google/sentencepiece) | BPE/Unigram tokenizer implementation | code/sentencepiece/ | ✓ Cloned |

### Repository Details

#### 1. Byte Latent Transformer (BLT)
- **Primary Use**: Reference implementation for tokenizer-free approach
- **Key Features**: Dynamic patching, entropy-based segmentation, 8B scale
- **Relevance**: Baseline for comparing artificial token language against tokenizer-free
- **Installation**: `cd code/blt && pip install -r requirements.txt`

#### 2. LM Evaluation Harness
- **Primary Use**: Standardized benchmarking on WikiText, LAMBADA, GLUE
- **Key Features**: 400+ tasks, consistent evaluation protocol, widely used in research
- **Relevance**: Essential for reproducible, comparable results
- **Installation**: `cd code/lm-evaluation-harness && pip install -e .`

#### 3. SentencePiece
- **Primary Use**: Training BPE and Unigram baselines
- **Key Features**: Multiple algorithms (BPE, Unigram, Char, Word), language-agnostic
- **Relevance**: Standard baselines for comparison
- **Installation**: `cd code/sentencepiece && pip install .`

**Detailed Information**: See [code/README.md](code/README.md)

---

## Resource Gathering Process

### Search Strategy

**Literature Search**:
1. ArXiv search for: tokenization efficiency, byte latent transformer, compressed representations
2. Web search for: recent tokenization research (2023-2025)
3. Citation following from key papers (BLT, Language Modeling is Compression)
4. Focus on: high-impact venues (ICML, NeurIPS, EMNLP, ACL), recent work (2020+)

**Dataset Search**:
1. HuggingFace datasets hub for standard benchmarks
2. Papers with Code for dataset-benchmark associations
3. Literature review to identify commonly used datasets
4. Selection criteria: standard benchmarks, manageable size, diverse task types

**Code Repository Search**:
1. GitHub search for official implementations
2. Papers with Code implementation links
3. Priority: official repos > well-maintained forks > educational implementations

### Selection Criteria

**Papers Selected Based On**:
- Direct relevance to tokenization efficiency
- Theoretical or empirical contributions to understanding token representations
- Recent work (primarily 2023-2025) with established baselines (2020)
- High-quality venues or well-cited arXiv preprints
- Diversity: theory, architecture, optimization, evaluation

**Datasets Selected Based On**:
- Standard benchmarks used in tokenization research
- Size appropriate for research iteration (not requiring massive compute)
- Task diversity: language modeling, context prediction, downstream tasks
- Availability and ease of access (HuggingFace preferred)

**Code Selected Based On**:
- Official implementations when available
- Active maintenance and documentation
- Relevance to baseline comparisons
- Practical utility for experiments

---

## Challenges Encountered

### Challenge 1: Paper Volume
**Issue**: Many papers on tokenization, difficult to select most relevant ones

**Resolution**: Focused on papers directly addressing efficiency/compression, plus key foundational work. Limited to 8 papers for depth over breadth.

### Challenge 2: Dataset Size
**Issue**: C4 and The Pile are very large (300GB+), impractical for initial experiments

**Resolution**: Selected WikiText-103 as primary (100M tokens), which is standard benchmark but manageable size. Documented larger datasets for future work.

### Challenge 3: Repository Dependencies
**Issue**: Some repos have complex dependencies or require specific hardware

**Resolution**: Selected well-documented repos with standard dependencies. Documented setup instructions for each.

### Challenge 4: Reproducibility
**Issue**: Need to ensure experiments can be reproduced without committing large files

**Resolution**: Created comprehensive download instructions and scripts. Used `.gitignore` to exclude data while documenting exact versions and sources.

---

## Gaps and Workarounds

### Gap 1: Limited Artificial Language Research
**Gap**: No existing papers specifically on "artificial token languages for LLMs"

**Workaround**: Gathered related work on: tokenization optimization, tokenizer-free models, learned compression. These provide building blocks for artificial language design.

### Gap 2: Multilingual Tokenization
**Gap**: Research is primarily English-focused, limited multilingual analysis

**Workaround**: Noted BLT's language-agnostic byte-level approach. Documented multilingual datasets (LAMBADA multilingual) for future expansion.

### Gap 3: Human Interpretability
**Gap**: Little research on making tokens human-interpretable

**Workaround**: Identified this as research opportunity. Current focus on efficiency first, interpretability as secondary consideration.

### Gap 4: Large-Scale Validation
**Gap**: Largest experiments in papers use 7B-8B parameters, but SOTA models are 70B+

**Workaround**: Experiments will start at 124M-1B scale. Literature shows tokenization effects are consistent across scales, so findings should generalize.

---

## Recommendations for Experiment Design

Based on gathered resources and literature synthesis:

### 1. Primary Dataset: WikiText-103
**Rationale**:
- Standard benchmark with extensive baselines
- Size enables meaningful experiments without massive compute
- Direct comparison with published results possible
- Language modeling is core task for evaluating tokenization

**Alternative/Additional**:
- WikiText-2 for rapid iteration during development
- LAMBADA for testing context understanding
- GLUE MRPC for downstream task validation

### 2. Baseline Methods

**Must Implement**:
1. **BPE (32K vocabulary)** - Industry standard, used in GPT
   - Use SentencePiece implementation
   - Metrics: tokens/byte, perplexity, training speed

2. **Unigram LM (32K vocabulary)** - Better alternative to BPE
   - Use SentencePiece implementation
   - Expect slight improvement over BPE

**Stretch Goals**:
3. **Byte-level** - Vocabulary-free baseline
   - Simplest approach, longest sequences
   - Useful for compression comparison

4. **BLT** - State-of-the-art tokenizer-free
   - If computational resources allow
   - Represents current best alternative to tokenization

### 3. Evaluation Metrics

**Essential**:
1. **Tokenization Efficiency**: Tokens per byte (target: >5, current BPE: ~4-4.5)
2. **Training Efficiency**: Steps to target perplexity, wall-clock time
3. **Model Quality**: Perplexity on WikiText-103 (target: match or beat BPE)
4. **Downstream Performance**: Accuracy on LAMBADA, GLUE MRPC

**Advanced** (if time permits):
5. **Information Density**: Renyi efficiency (use framework from Theory of Tokenization paper)
6. **Inference Speed**: Tokens per second throughput
7. **Memory Usage**: Peak memory during training/inference

**Implementation**: Use LM Evaluation Harness for standardized, reproducible evaluation

### 4. Model Scale Recommendation

**Phase 1 - Rapid Iteration** (recommended starting point):
- Model size: 124M-350M parameters
- Training time: Hours to days on single/few GPUs
- Purpose: Test different artificial token language designs, iterate quickly

**Phase 2 - Validation** (if Phase 1 shows promise):
- Model size: 1B-3B parameters
- Training time: Days on multi-GPU
- Purpose: Validate that efficiency gains hold at larger scale

**Phase 3 - Final Validation** (if resources allow and results are strong):
- Model size: 7B parameters
- Training time: Weeks on cluster
- Purpose: Demonstrate competitiveness with published results

### 5. Experimental Design Principles

**Controls**:
- Same model architecture across all tokenization schemes
- Same training data (just different tokenization)
- Same compute budget (FLOP-controlled comparison)

**Reproducibility**:
- Multiple random seeds (≥3 runs per configuration)
- Report mean ± standard deviation
- Use LM Evaluation Harness for consistent evaluation
- Document all hyperparameters
- Publish tokenizer configurations

**Success Criteria** (from literature review):
- **Minimum**: ≥10% fewer tokens, equal perplexity, ≥5% faster training
- **Strong**: ≥15% fewer tokens, better perplexity, ≥10% faster training
- **Exceptional**: ≥20% fewer tokens, competitive with BLT, ≥15% faster training

---

## Resource Utilization Plan

### Immediate Next Steps (Experiment Runner Phase)

1. **Setup Environment**:
   - Install LM Evaluation Harness: `cd code/lm-evaluation-harness && pip install -e .`
   - Install SentencePiece: `cd code/sentencepiece && pip install .`
   - Verify datasets are accessible: `python -c "from datasets import load_from_disk; load_from_disk('datasets/wikitext-103-v1')"`

2. **Baseline Tokenizers**:
   - Train BPE tokenizer (32K vocab) on WikiText-103 train split
   - Train Unigram tokenizer (32K vocab) on WikiText-103 train split
   - Measure tokenization efficiency (tokens/byte) on test set

3. **Artificial Token Language Design**:
   - Use insights from papers (compression, Renyi efficiency, morphology)
   - Design initial candidate language
   - Implement encoding/decoding

4. **Small-Scale Validation**:
   - Train 124M parameter models with BPE, Unigram, and artificial language
   - Measure training efficiency and perplexity
   - Iterate on artificial language design based on results

5. **Evaluation**:
   - Use LM Evaluation Harness for WikiText perplexity
   - Test on LAMBADA for context understanding
   - Test on GLUE MRPC for downstream tasks

### Papers to Re-Read During Experimentation

- **Theory of Tokenization**: For Renyi efficiency calculation
- **Language Modeling is Compression**: For compression-based evaluation
- **Length-MAX Tokenizer**: For optimization strategies
- **BPE is Suboptimal**: For understanding Unigram advantages

### Code to Reference During Implementation

- **SentencePiece**: For tokenization baselines and implementation patterns
- **LM Evaluation Harness**: For evaluation protocol and metrics
- **BLT**: For understanding tokenizer-free alternatives (if implementing similar approach)

---

## Maintenance and Updates

### Resource Freshness

**Papers**: Current as of December 2025, includes latest work through arXiv

**Datasets**: Standard benchmarks, stable over time. WikiText versions are fixed.

**Code**: Cloned repositories are snapshots. For production use, check for updates:
- BLT: `cd code/blt && git pull` (if not using --depth 1)
- LM Evaluation Harness: Actively maintained, check for new features
- SentencePiece: Stable, infrequent updates

### Future Resource Additions

If research progresses beyond initial experiments:

**Datasets**:
- C4 or The Pile for large-scale pre-training validation
- Multilingual datasets for cross-lingual evaluation
- Domain-specific datasets for transfer learning tests

**Code**:
- HuggingFace Transformers for integration with existing ecosystems
- nanoGPT for minimal, educational implementation
- Custom implementations as research develops

**Papers**:
- Continue monitoring arXiv for new tokenization research
- Follow citations of key papers (BLT, Theory of Tokenization)
- Track ICML/NeurIPS/EMNLP proceedings for related work

---

## Citations and Acknowledgments

All resources are properly documented with citations in their respective README files:

- **Papers**: See [papers/README.md](papers/README.md) for full citations
- **Datasets**: See [datasets/README.md](datasets/README.md) for attribution
- **Code**: See [code/README.md](code/README.md) for repository citations

When publishing results from this research, ensure proper citation of:
1. All papers used for background and methodology
2. Datasets used for training and evaluation
3. Code repositories used or adapted
4. LM Evaluation Harness for standardized evaluation

---

## Resource Inventory Summary

```
workspace/
├── papers/                          # 8 PDF papers (16 MB total)
│   ├── README.md                    # Detailed paper descriptions
│   └── *.pdf                        # Downloaded papers
│
├── datasets/                        # 4 datasets (~350 MB total)
│   ├── README.md                    # Download instructions & documentation
│   ├── .gitignore                   # Excludes data from git
│   ├── samples/                     # Small sample files for docs
│   ├── wikitext-2-v1/              # 36K train examples
│   ├── wikitext-103-v1/            # 1.8M train examples
│   ├── lambada_openai/             # 5K test examples
│   └── glue-mrpc/                  # 3.6K train examples
│
├── code/                           # 3 repositories
│   ├── README.md                   # Repository documentation
│   ├── blt/                        # Byte Latent Transformer
│   ├── lm-evaluation-harness/      # Evaluation framework
│   └── sentencepiece/              # Tokenizer implementation
│
├── literature_review.md            # Comprehensive synthesis (this document)
├── resources.md                    # Resource catalog (this document)
├── download_datasets.py            # Dataset download script
└── .resource_finder_complete       # Completion marker (to be created)
```

**Total Storage**: ~400 MB (excluding git repositories)

**Git-Tracked**: Papers (16 MB), documentation, scripts
**Git-Ignored**: Datasets (350 MB), large code repos

---

## Contact and Support

For questions about resources:

**Resource Issues**:
- Broken download links: Check HuggingFace/arXiv directly
- Missing papers: arXiv IDs provided for direct access
- Dataset problems: See datasets/README.md for alternative download methods

**Research Questions**:
- See literature_review.md for detailed analysis
- Papers in papers/ directory for original sources
- Code repositories include issue trackers for implementation questions

---

**Resource Finding Phase Status**: ✓ COMPLETE

All required resources have been gathered, documented, and organized for the experiment runner phase to begin.
