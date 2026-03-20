# Public Code Usage Guide

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
