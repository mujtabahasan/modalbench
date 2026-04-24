# ModalBench: Evaluating Modal and Deontic Logic Reasoning in Large Language Models

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026%20Workshop-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)

> **The first benchmark with explicit Kripke-semantic ground truth for evaluating modal and deontic logic reasoning in LLMs.**
>
> Accepted at the [**ICLR 2026 Workshop on Logical Reasoning of Large Language Models**](https://sites.google.com/view/iclr-2026-llmreasoning).

---

## Overview

When a doctor says a patient "must have an infection" or a contract states a party "may terminate upon notice," the reasoning is *modal* — it concerns not just what is true, but what is *necessarily* true, *possibly* true, *obligatory*, or *permitted*. Modal logic formalizes these notions with operators □ (necessity) and ◇ (possibility), evaluated over possible worlds connected by accessibility relations (Kripke semantics). Deontic logic extends this to obligation (OB), permission (PE), and prohibition (FO).

Despite the importance of modal reasoning in medicine, law, AI safety, and planning, **no existing benchmark tests whether LLMs can perform it.** FOLIO, LogiQA, ProofWriter, PrOntoQA, and LogicBench all evaluate only propositional and first-order logic. ModalBench fills this gap.

---

## Key Findings

From **27,000 inferences** across 3 LLMs × 3 prompting strategies × 3,000 problems (with 16k-token generation budget and robust answer extraction):

| Finding | Detail |
|---|---|
| **Overall accuracy** | 73–91% under CoT prompting (Gemini 2.5 Flash best at 90.8%) |
| **System K is hardest** | 65–75% across models — no structural constraints = harder reasoning |
| **NL advantage** | 2 of 3 models reason *better* in natural language than formal notation |
| | Qwen3-235B: **−18.4 pp** gap (NL better), p < 10⁻⁹⁹ |
| | Llama 3.3 70B: **−4.7 pp** gap (NL better), p < 10⁻⁶ |
| | Gemini 2.5 Flash: +1.0 pp (neutral, not significant) |
| **World-enum as scaffold** | Boosts weaker models +9–11 pp, unnecessary for strongest (95.9% zero-shot) |
| **Implicit S5 bias** | All models over-apply axiom 5 by up to **+56 pp** in system K |
| **Contrary-to-duty** | Gentle Murderer paradox hardest (<79% for all models) |

---

## Benchmark Design

| Dimension | Details |
|---|---|
| **Total problems** | 3,000 (1,500 unique × 2 presentation tracks) |
| **Modal systems** | K (none), T (reflexive), S4 (refl.+trans.), S5 (equivalence), D (serial/deontic) |
| **Difficulty tiers** | Tier 1: single operator (□p) · Tier 2: nested (□◇p) · Tier 3: axiom interactions + paradoxes |
| **Tracks** | Formal symbolic + Natural language narrative |
| **Frame size** | 2–7 worlds (mean 4.5) |
| **Propositions** | 2–5 per problem |
| **Axiom types** | 23 tagged formula categories in Tier 3 |
| **Deontic paradoxes** | 5: Ross, Chisholm, Gentle Murderer, Good Samaritan, FO/PE Contradiction |
| **Label balance** | Exactly 50/50 True/False per system×tier cell |
| **Heuristic resistance** | "□ implies True" baseline = 47.1% (near chance) |
| **Ground truth** | Computed by recursive Kripke evaluation — zero human annotation |

### Dual-Track Presentation

Every problem appears in two forms with identical Kripke semantics:

- **Formal track:** Explicit worlds, accessibility relations, valuations, and symbolic formulas (□, ◇, OB, PE, FO)
- **Natural language track:** Alice explores rooms with one-way observation windows (alethic) or jurisdictions with regulatory influence (deontic)

The gap between tracks reveals whether failures stem from modal reasoning itself or from notation comprehension.

### Deontic Paradoxes

Tier 3 includes five classical paradoxes that test the limits of standard deontic logic:

- **Ross's Paradox:** OB(p) → OB(p∨q) — "you must mail the letter" entails "you must mail or burn it" (100% solved by all models)
- **Chisholm's Paradox:** Contrary-to-duty norms that become inconsistent in SDL
- **Gentle Murderer:** "If you murder, murder gently" — hardest paradox (<79%)
- **FO/PE Contradiction:** FO(p) ∧ PE(p) should always be False (89–98%)
- **Good Samaritan:** Helping obligations that presuppose a crime

---

## Repository Structure

```
modalbench/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── modalbench_complete.py                 # Core pipeline: Kripke engine, generators, evaluator, diagnostics
├── modalbench_neurips_extensions.py        # Extensions: RobustParser, NaturalisticNL, probing, contamination check
├── build_unified_results.py               # Unify JSONL inference files → single CSV + full analysis + figures
├── analyze_real_results.py                # Standalone analysis on results CSV
├── reanalyze_with_robust_parser.py        # Re-analysis comparing original vs robust parser
│
├── data/
│   ├── bench_final.json                   # 3,000 problems with verified Kripke ground truth
│   └── README.md                          # Schema documentation
│
├── results/
│   ├── results_unified.csv                # 27,000-row evaluation table (re-run with 16k tokens, RobustParser)
│   └──  results.zio                       # zip file containing inference of 3 LLMs with 3 prompting strategies (9 files)
│
├── figures/                               # All paper figures (PNG + PDF)
│   ├── fig1_overview.{png,pdf}
│   ├── fig1_heatmap.{png,pdf}
│   ├── fig3_formal_vs_nl.{png,pdf}
│   ├── fig5_s5_bias.{png,pdf}
│   └── ...
│
├── notebooks/
│   └── reproduce_paper.ipynb              # End-to-end reproduction notebook (Colab)
│
└── paper/
    ├── modalbench_paper.tex               # Camera-ready LaTeX source
    ├── modalbench_paper.pdf               # Compiled PDF
    ├── iclr2026_conference.bib            # Bibliography
    └── PARSER_CORRECTION_NOTICE.md        # Documents parser bug found and fixed during review
```

---

## Quick Start

### Install

```bash
git clone https://github.com/mujtabahasan/modalbench.git
cd modalbench
pip install -r requirements.txt
```

### Reproduce paper figures from cached results (no API keys needed)

```bash
cd modalbench
unzip results/results.zip -d .
python build_unified_results.py \
  --input_dir results/ \
  --bench data/bench_final.json \
  --output_dir output/
```

This reads `results/results_unified.csv`, generates all 8 figures, 6 tables, and statistical tests in ~30 seconds.

### Generate the benchmark from scratch

```python
from modalbench_complete import generate_benchmark

problems = generate_benchmark()  # 3,000 problems
print(f"Generated {len(problems)} problems")
# Ground truth is computed automatically via Kripke evaluation
```

### Run inference on new models

Each model needs its own inference script that:
1. Loads `data/bench_final.json`
2. For each problem × strategy, queries the model with the problem description
3. Saves results as JSONL with fields: `id`, `response`, `ground_truth`

Then unify all JSONL files:

```bash
python build_unified_results.py \
  --input_dir /path/to/jsonl/files/ \
  --bench data/bench_final.json \
  --output_dir my_results/
```

The script auto-detects model names and strategies from filenames. Supported patterns:
- `*gemini*cot*.jsonl` → (gemini-2.5-flash, cot)
- `*llama*70b*zero_shot*.jsonl` → (llama-70b-groq, zero_shot)
- `*qwen*235*world_enum*.jsonl` → (qwen3-235b-cerebras, world_enum)
- `*llama*8b*`, `*qwen*70b*`, `*mistral*7b*` also recognized

### Prompting strategies

The three strategies tested in the paper:

1. **Zero-shot:** Problem description + "Answer True or False."
2. **Chain-of-thought (CoT):** Problem description + "Think step by step. Then answer True or False."
3. **World-enumeration CoT:** Problem description + explicit instructions to:
   - List all accessible worlds from the evaluation world
   - Tabulate sub-formula truth values at each
   - Aggregate with the correct quantifier (∀ for □, ∃ for ◇)
   - Recurse inside-out for nested operators

---

## Data Schema

### `data/bench_final.json`

JSON list of 3,000 problem objects:

```json
{
  "id": "MB-K-T1-QA-0042",
  "system": "K",
  "tier": 1,
  "presentation": "formal",
  "deontic": false,
  "modal_depth": 1,
  "num_worlds": 4,
  "worlds": ["w0", "w1", "w2", "w3"],
  "relations": [["w0", "w1"], ["w1", "w2"], ...],
  "propositions": ["p", "q"],
  "valuation": {"w0": {"p": true, "q": false}, ...},
  "evaluation_world": "w0",
  "formula_str": "□(p)",
  "formula_symbolic": "□(p)",
  "ground_truth": false,
  "axiom_tag": null,
  "paradox_name": null,
  "nl_description": "Alice is exploring a building...",
  "description": "Worlds: {w0, w1, w2, w3}..."
}
```

### `results/results_unified.csv`

27,000 rows (3 models × 3 strategies × 3,000 problems):

| Column | Type | Description |
|---|---|---|
| `problem_id` | str | FK to bench_final.json |
| `model` | str | gemini-2.5-flash, llama-70b-groq, qwen3-235b-cerebras |
| `prompt_strategy` | str | zero_shot, cot, world_enum |
| `system` | str | K, T, S4, S5, D |
| `tier` | int | 1, 2, 3 |
| `presentation` | str | formal, nl |
| `ground_truth` | bool | Verified correct answer |
| `predicted` | bool/NaN | Model's parsed answer |
| `correct` | float | 1.0 / 0.0 / NaN (parse failure) |
| `response` | str | Full model response (up to 16k tokens) |
| `axiom_tag` | str/null | Axiom name for Tier 3 problems |
| `paradox_name` | str/null | Paradox name if applicable |

---

## Models Evaluated

| Model | Type | Provider | Parse Fail | Best Strategy | Best Accuracy |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | Reasoning | Google AI Studio | 2.3% | Zero-shot | **95.9%** |
| Llama 3.3 70B | Standard | Groq | 7.8% | World-enum | **81.4%** |
| Qwen3-235B Instruct | MoE | Cerebras | 0.0% | World-enum | **81.5%** |

All accessed via API tiers with a 16k-token generation budget.

---

## Results at a Glance

### Main Results (CoT)

| Model | D | K | S4 | S5 | T | Overall |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 94.8 | 72.0 | 97.3 | 94.6 | 95.5 | **90.8** |
| Llama 3.3 70B | 71.7 | 65.6 | 73.8 | 75.8 | 77.3 | 72.8 |
| Qwen3-235B | 82.3 | 75.2 | 80.2 | 82.8 | 82.5 | 80.6 |

### Tier Difficulty (CoT)

| Model | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Gemini 2.5 Flash | 92.4 | 90.2 | 90.0 |
| Llama 3.3 70B | 79.2 | 71.6 | 67.7 |
| Qwen3-235B | 87.0 | 75.5 | 79.3 |

### Implicit S5 Bias (axiom 5 over-application)

| Model | System K | S4 | S5 | T |
|---|---|---|---|---|
| Gemini | **+0.44** | 0.00 | 0.00 | 0.00 |
| Llama | +0.22 | 0.00 | −0.27 | 0.00 |
| Qwen3 | **+0.56** | +0.22 | 0.00 | **+0.50** |

---

## Answer Extraction

### RobustParser (recommended)

Located in `modalbench_neurips_extensions.py`. A cascade-based parser that:

1. Strips leading markdown and checks prefix for True/False
2. Handles `**True**` / `**False**` markdown-bold answers
3. Matches "Answer: True/False" patterns
4. Extracts from final line
5. Checks boxed LaTeX answers (`\boxed{True}`)
6. Searches for conclusion markers ("therefore", "the answer is")
7. **Refuses to guess** — returns `None` if no clear answer found

```python
from modalbench_neurips_extensions import RobustParser

answer = RobustParser.parse("**False**\n\nStep 1: We evaluate...")
# Returns: False

answer = RobustParser.parse("Let me think step by step. The formula at world A...")
# Returns: None (no clear answer — truncated mid-reasoning)
```

### Original parser (deprecated)

Located in `modalbench_complete.py` as `parse_answer()`. Falls back to majority word-counting on the full response, which produces unreliable labels when responses discuss sub-formula truth values. **Use RobustParser instead.**

See `paper/PARSER_CORRECTION_NOTICE.md` for the full analysis of how this bug was discovered, its impact, and the manual validation (95% vs 5% accuracy on 30 stratified disagreement cases).

---

## Extensions

| Extension | Class/Function | Purpose |
|---|---|---|
| **Robust parsing** | `RobustParser` | Cascade parser, no majority-count fallback |
| **Naturalistic NL** | `NaturalisticNL` | Epistemic/deontic/dynamic modal language templates |
| **Hidden-state probing** | `AccessibilityProbe` | Linear probe for accessibility relation encoding |
| **Prompt sensitivity** | `prompt_sensitivity_analysis()` | 10 extended prompt variants |
| **Error analysis** | `cross_model_error_analysis()` | Universal failures and discriminative problems |
| **Contamination check** | `benchmark_contamination_check()` | Training data leakage detection |
| **Neuro-symbolic** | `NeuroSymbolicBaseline` | LLM extracts Kripke model + symbolic checker |

---

## Reproducing Paper Figures

Each paper figure maps to generated output:

| Paper Figure | Generated File | What It Shows |
|---|---|---|
| Figure 1 | `fig1_overview.pdf` | ModalBench overview (Kripke frame + dual track + stats) |
| Figure 2 | Side-by-side prompt box | Formal vs NL prompt comparison (in LaTeX) |
| Figure 3 | `fig3_formal_vs_nl.pdf` | NL advantage bar chart |
| Figure 4 | `fig4_s5_bias.pdf` | Implicit S5 bias heatmap |
| Figure 5 | `fig5_worlds.pdf` + `fig6_depth.pdf` | Scaling: frame size (flat) vs depth (declining) |
| App. Fig | `fig1_heatmap.pdf` | System×tier accuracy heatmap per model |

To regenerate:

```bash
python build_unified_results.py -i results/ -b data/bench_final.json -o output/
# Figures saved in output/figures/
```

---

## Verification

The benchmark has been independently verified:

- **Ground truth:** 3,000/3,000 problems verified by an independent Kripke evaluator (reimplemented from scratch, zero mismatches)
- **Frame properties:** 3,000/3,000 frames satisfy their system constraints (T: reflexive, S4: refl.+trans., S5: equivalence, D: serial)
- **Label balance:** Exactly 50/50 True/False per system×tier cell (15 cells verified)
- **Evaluation world validity:** All 3,000 evaluation worlds are members of their world set
- **Valuation completeness:** Every world × proposition pair has an assigned value
- **Formal/NL pairing:** Exactly 1,500 paired problems sharing the same underlying Kripke model
- **No duplicates:** All 3,000 problem descriptions are unique
- **Response integrity:** 95.3% of responses exceed 500 characters (no truncation with 16k budget)

---

## Citation

```bibtex
@inproceedings{hasan2026modalbench,
  title     = {ModalBench: Evaluating Modal and Deontic Logic Reasoning
               in Large Language Models},
  author    = {Hasan, Mujtaba},
  booktitle = {ICLR 2026 Workshop on Logical Reasoning of Large Language Models},
  year      = {2026},
  url       = {https://github.com/mujtabahasan/modalbench}
}
```

---

## Contributing

Contributions welcome! Particularly interested in:

- **More models** — especially open-weight models for hidden-state probing
- **Naturalistic modal language** — real-world "must"/"might"/"should" with context-dependent readings
- **Multi-agent epistemic logic** — "Alice knows that Bob believes P"
- **Temporal and dynamic logic** — □_future, ◇_past, action operators
- **Neuro-symbolic pipelines** — LLM extracts Kripke model, symbolic solver evaluates
- **Adaptive prompting** — select strategy based on model capability

Open an issue or PR.

---

## License

MIT License — see [LICENSE](LICENSE).

---
