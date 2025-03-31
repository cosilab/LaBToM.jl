# Epistemic Language-of-Thought (ELoT) Translation

The `translation` directory contains code for translating natural language into an epistemic language-of-thought (ELoT) representation. The `elot_translate.py` script can invoked at the command line to translate natural language statements into ELoT statements.

## Setup

We use the [**GenParse**](https://github.com/ChiSym/genparse) library introduced by [Loula et al (2025)](https://openreview.net/forum?id=xoXn62FzD0) to perform sequential Monte Carlo (SMC) grammar-constrained decoding. Install genparse by following the instructions described in the [GenParse repository](https://github.com/ChiSym/genparse) into a Python environment of your choice.

To use GenParse with `llama-3.1` (8B) for translation, we recommmed using a GPU machine with sufficient VRAM. On the first run of `elot_translate.py`, genparse will download a model for `llama-3.1` via HuggingFace if not already downloaded. This requires specifying a `--hf_token` argument from the [HuggingFace Hub](https://huggingface.co/settings/tokens). You will also need to [request access](https://huggingface.co/meta-llama/Llama-3.1-8B) to the `llama-3.1` model series.

## Usage

Run `python elot_translate.py --help` for usage instructions.

To reproduce the LLaMa 3.1 translation experiments in our paper, run the following commands from this directory:

```bash
# Grammar-constrained translation to ELoT formulas
python elot_translate.py --input_tense current --output_format elot
python elot_translate.py --input_tense initial --output_format elot
# Grammar-constrained translation to lowered formulas
python elot_translate.py --input_tense current --output_format lowered
python elot_translate.py --input_tense initial --output_format lowered
# Unconstrained translation to ELoT formulas
python elot_translate.py --input_tense current --output_format elot --unconstrained
python elot_translate.py --input_tense initial --output_format elot --unconstrained
# Unconstrained translation to lowered formulas
python elot_translate.py --input_tense current --output_format lowered --unconstrained
python elot_translate.py --input_tense initial --output_format lowered --unconstrained
```
