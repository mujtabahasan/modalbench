# ModalBench Data

## `bench_final.json`

The complete ModalBench benchmark: **3,000 problems** (1,500 unique × 2 presentation tracks).

### Format
JSON list of problem objects.

### Schema

```json
{
  "id": "K_1_42",                    // Unique problem ID: {system}_{tier}_{idx}
  "system": "K",                     // Modal system: K | T | S4 | S5 | D
  "tier": 1,                         // Difficulty: 1 | 2 | 3
  "presentation": "formal",          // Track: formal | nl
  "deontic": false,                  // True for system D
  "modal_depth": 1,                  // Nesting depth of modal operators
  "num_worlds": 4,                   // Number of worlds in Kripke frame
  "formula": "□p",                   // Symbolic formula (Unicode)
  "axiom_tag": null,                 // Axiom name for Tier 3 (T, 4, 5, B, K, ...)
  "paradox_name": null,              // Deontic paradox name if applicable
  "ground_truth": true,              // Computationally verified answer
  "description": "...",              // Full problem text shown to LLM
  "frame": {                         // Underlying Kripke frame
    "worlds": ["w0", "w1", "w2", "w3"],
    "rels": [["w0","w1"], ["w1","w2"], ...],
    "valuation": {"w0": {"p": true}, ...}
  }
}
```

### Statistics

| Dimension | Value |
|---|---|
| Total problems | 3,000 |
| Unique problems | 1,500 |
| Per system × tier cell | 200 (100 formal + 100 NL) |
| Modal systems | K, T, S4, S5, D |
| Difficulty tiers | 3 |
| Tracks | formal, nl |
| Frame size range | 2–7 worlds (mean 4.5) |
| Axiom-tagged formulas (Tier 3) | 23 types |
| Deontic paradox types | 5 |
| Label distribution | 50% True / 50% False per cell |

### Heuristic Resistance

| Heuristic | Accuracy |
|---|---|
| Always answer True | 50.0% |
| `□` implies True | 47.1% |

Both baselines are at chance, confirming the benchmark resists shallow strategies.

### Loading

```python
import json
with open('data/bench_final.json') as f:
    problems = json.load(f)

print(f"Loaded {len(problems)} problems")

# Filter by system and tier
k_tier3 = [p for p in problems if p['system'] == 'K' and p['tier'] == 3]
print(f"K Tier 3: {len(k_tier3)} problems")
```

### Generation

The benchmark is fully algorithmic and reproducible:

```python
from modalbench_complete import generate_benchmark
problems = generate_benchmark(n_per_cell=100, seed=42)
```

This produces an identical benchmark given the same seed.
