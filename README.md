# Public Code Usage Guide

This is the source code for the paper "An Information-Theoretic Approach to Understanding Transformers' In-Context Learning of Variable-Order Markov Chains", by Zhou, Tian, and Diggavi, published in ATSTATS 2026. 

Abstract: We study transformers' in-context learning of variable-length Markov chains (VOMCs), focusing on the finite-sample accuracy as the number of in-context examples increases. Compared to fixed-order Markov chains (FOMCs), learning VOMCs is substantially more challenging due to the additional structural learning component. The problem is naturally suited to a Bayesian formulation, where the context-tree weighting (CTW) algorithm, originally developed in the information theory community for universal data compression, provides an optimal solution. Empirically, we find that single-layer transformers fail to learn VOMCs in context, whereas transformers with two or more layers can succeed, with additional layers yielding modest but noticeable improvements. In contrast to prior results on FOMCs, attention-only networks appear insufficient for VOMCs. To explain these findings, we provide explicit transformer constructions: one with $D+2$ layers that can exactly implement CTW for VOMCs of maximum order $D$, and a simplified two-layer construction that uses partial information for approximate blending, shedding light on why two-layer transformers can perform well.


## 1. Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the environment already exists, only run:

```bash
source .venv/bin/activate
```

## 2. Entry Point

Main entry point:

```bash
python main.py --mode <data_gen|train|evaluate> --config <ConfigClassName>
```

Examples:

```bash
python public/main.py --mode data_gen --config ConfigTest
python public/main.py --mode train --config Config20
python public/main.py --mode evaluate --config Config20
```

## 3. Choosing a Config

Config classes are defined in:

- `config.py`

Recommended quick smoke-test config:

- `ConfigTest`

For larger runs, use one of the numbered configs such as `Config20`, `Config60`, `Config150`, etc.

## 4. Outputs

By default, outputs are written (relative to project root) to:

- `data/LLMCTdata` (datasets)
- `data/LLMCTmodels` (checkpoints/models)
- `data/LLMCTresults` (evaluation results)

## Notes:
A plotting python script in included, which can be used after the corresponding evaluation results are obtained.

## Citation
If you find our work relevant to your research, please cite:
```bibtex
@inproceedings{zhou2026information,
  title={An Information-Theoretic Approach to Understanding Transformers' In-Context Learning of Variable-Order Markov Chains},
  author={Zhou, Ruida and Tian, Chao and Diggavi, Suhas},
  booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
  year={2026}
}
