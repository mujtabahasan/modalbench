#!/usr/bin/env python3
"""
================================================================================
ModalBench v2 — Unified Results Builder
================================================================================

Reads all 9 JSONL result files (3 models × 3 strategies), applies RobustParser,
and produces a single unified CSV + full paper analysis with figures.

Usage:
    1. Put all JSONL files in a folder (e.g., /content/drive/MyDrive/results/)
    2. Run:
       python build_unified_results.py --input_dir /path/to/jsonl/files

    Or on Colab, just set INPUT_DIR below and run the cell.
================================================================================
"""

import os, sys, json, re, glob, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════
# CONFIG — edit these if running as a Colab cell
# ═══════════════════════════════════════════════════════════════════════

INPUT_DIR = "."  # folder containing the 9 JSONL files
OUTPUT_DIR = "modalbench_v2_unified"
BENCH_PATH = "data/bench_final.json"  # path to benchmark JSON

# File-to-(model, strategy) mapping
# The script auto-detects from filenames, but you can override here
FILE_MAP = {
    "modalbench_gemini_2_5_flash_cot_results_full_REPAIRED_1.jsonl": ("gemini-2.5-flash", "cot"),
    "modalbench_gemini_2_5_flash_world_enum_results_full_REPAIRED.jsonl": ("gemini-2.5-flash", "world_enum"),
    "modalbench_gemini_2_5_flash_zero_shot_results_full_REPAIRED.jsonl": ("gemini-2.5-flash", "zero_shot"),
    "modalbench_results_groq_llama70b_cot.jsonl": ("llama-70b-groq", "cot"),
    "modalbench_results_groq_llama70b_world_enum.jsonl": ("llama-70b-groq", "world_enum"),
    "modalbench_results_groq_llama70b_zero_shot.jsonl": ("llama-70b-groq", "zero_shot"),
    "results_modalbench_cot_cerebras_qwen3_235b.jsonl": ("qwen3-235b-cerebras", "cot"),
    "results_modalbench_world_enum_cerebras_qwen3_235b.jsonl": ("qwen3-235b-cerebras", "world_enum"),
    "results_modalbench_zero_shot_cerebras_qwen3_235b.jsonl": ("qwen3-235b-cerebras", "zero_shot"),
}

# ═══════════════════════════════════════════════════════════════════════
# ROBUST PARSER (from modalbench_neurips_extensions.py — self-contained)
# ═══════════════════════════════════════════════════════════════════════

TRUE_WORDS = r'(?:true|True|TRUE|yes|Yes|YES|correct|Correct)'
FALSE_WORDS = r'(?:false|False|FALSE|no|No|NO|incorrect|Incorrect)'

def robust_parse(response):
    """Parse True/False from model response. Returns True, False, or None."""
    if not isinstance(response, str) or not response.strip():
        return None
    r = response.strip()

    # Stage 0: Strip leading markdown
    r_clean = re.sub(r'^[\*_#`\s]+', '', r)

    # Stage 1: Prefix
    prefix = r_clean[:20].lower()
    if prefix.startswith('true'): return True
    if prefix.startswith('false'): return False

    # Stage 1b: Markdown-bold at start
    if re.match(r'^[\*_]*(' + TRUE_WORDS + r')[\*_.]*', r, re.I): return True
    if re.match(r'^[\*_]*(' + FALSE_WORDS + r')[\*_.]*', r, re.I): return False

    # Stage 2: "Answer: X"
    m = re.search(rf'\banswer\s*[:=]?\s*\**\s*({TRUE_WORDS})\b', r, re.I)
    if m: return True
    m = re.search(rf'\banswer\s*[:=]?\s*\**\s*({FALSE_WORDS})\b', r, re.I)
    if m: return False

    # Stage 3: Final line
    lines = [ln.strip() for ln in r.strip().split('\n') if ln.strip()]
    if lines:
        last = re.sub(r'[*_`#]+', '', lines[-1]).strip().rstrip('.').lower()
        if last in ('true', 'yes', 'correct'): return True
        if last in ('false', 'no', 'incorrect'): return False

    # Stage 4: Boxed LaTeX
    m = re.search(r'\\boxed\{\s*(\w+)\s*\}', r)
    if m:
        val = m.group(1).lower()
        if val in ('true', 'yes'): return True
        if val in ('false', 'no'): return False

    # Stage 5: Conclusion markers
    for pat in [
        rf'\b(?:therefore|thus|so|hence|conclude|conclusion)\b[^.]*?\b({TRUE_WORDS}|{FALSE_WORDS})\b',
        rf'\bthe answer is\s+\**\s*({TRUE_WORDS}|{FALSE_WORDS})\b',
        rf'\b(?:statement|formula|expression)\s+is\s+({TRUE_WORDS}|{FALSE_WORDS})\b',
    ]:
        matches = re.findall(pat, r, re.I)
        if matches:
            last_match = matches[-1].lower()
            if re.match(TRUE_WORDS, last_match, re.I): return True
            if re.match(FALSE_WORDS, last_match, re.I): return False

    return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load benchmark metadata
# ═══════════════════════════════════════════════════════════════════════

def load_benchmark(bench_path):
    """Load bench_final.json and build lookup dict."""
    with open(bench_path) as f:
        bench = json.load(f)
    lookup = {p['id']: p for p in bench}
    print(f"✅ Loaded benchmark: {len(bench)} problems")
    return bench, lookup


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Auto-detect model/strategy from filename
# ═══════════════════════════════════════════════════════════════════════

def detect_model_strategy(filename):
    """Try to auto-detect (model, strategy) from filename."""
    fn = filename.lower()

    # Strategy detection
    if 'world_enum' in fn or 'worldenum' in fn:
        strategy = 'world_enum'
    elif 'zero_shot' in fn or 'zeroshot' in fn:
        strategy = 'zero_shot'
    elif 'cot' in fn:
        strategy = 'cot'
    else:
        strategy = 'unknown'

    # Model detection
    if 'gemini' in fn:
        model = 'gemini-2.5-flash'
    elif 'llama' in fn and '70b' in fn:
        model = 'llama-70b-groq'
    elif 'llama' in fn and '8b' in fn:
        model = 'llama-8b-groq'
    elif 'qwen' in fn and '235' in fn:
        model = 'qwen3-235b-cerebras'
    elif 'qwen' in fn and '70' in fn:
        model = 'qwen-70b'
    elif 'mistral' in fn and '7b' in fn:
        model = 'mistral-7b'
    else:
        model = 'unknown'

    return model, strategy


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Read JSONL files
# ═══════════════════════════════════════════════════════════════════════

def read_jsonl(filepath):
    """Read a JSONL file, return list of dicts."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def build_unified_df(input_dir, bench_lookup, file_map=None):
    """Read all JSONL files in input_dir and build a unified DataFrame."""
    all_rows = []
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))

    if not jsonl_files:
        print(f"❌ No JSONL files found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(jsonl_files)} JSONL files:")

    for filepath in jsonl_files:
        fname = os.path.basename(filepath)

        # Determine model and strategy
        if file_map and fname in file_map:
            model, strategy = file_map[fname]
        else:
            model, strategy = detect_model_strategy(fname)

        print(f"\n  📄 {fname}")
        print(f"     → model={model}, strategy={strategy}")

        records = read_jsonl(filepath)
        print(f"     → {len(records)} records")

        if not records:
            continue

        # Detect JSONL schema by inspecting first record
        sample = records[0]
        keys = set(sample.keys())

        # Common field name variations
        id_field = None
        for candidate in ['problem_id', 'id', 'question_id', 'qid', 'idx']:
            if candidate in keys:
                id_field = candidate
                break

        response_field = None
        for candidate in ['response', 'answer', 'output', 'model_response',
                          'model_output', 'completion', 'text', 'generated_text']:
            if candidate in keys:
                response_field = candidate
                break

        gt_field = None
        for candidate in ['ground_truth', 'gt', 'label', 'correct_answer',
                          'expected', 'true_answer', 'answer_label']:
            if candidate in keys:
                gt_field = candidate
                break

        print(f"     → Schema: id={id_field}, response={response_field}, gt={gt_field}")
        if id_field is None or response_field is None:
            print(f"     ⚠️ Cannot determine schema, dumping sample keys: {keys}")
            print(f"     Sample record: {json.dumps(sample, default=str)[:500]}")
            continue

        for rec in records:
            pid = str(rec.get(id_field, ''))
            response = rec.get(response_field, '')
            gt_raw = rec.get(gt_field)

            # Resolve ground truth: from record or from benchmark lookup
            if gt_raw is not None:
                if isinstance(gt_raw, bool):
                    gt = gt_raw
                elif isinstance(gt_raw, str):
                    gt = gt_raw.lower() in ('true', '1', 'yes')
                else:
                    gt = bool(gt_raw)
            elif pid in bench_lookup:
                gt = bench_lookup[pid]['ground_truth']
            else:
                gt = None

            # Parse answer with robust parser
            predicted = robust_parse(response)

            # Compute correctness
            if predicted is not None and gt is not None:
                correct = 1.0 if predicted == gt else 0.0
            else:
                correct = np.nan

            # Get problem metadata from benchmark
            meta = bench_lookup.get(pid, {})

            all_rows.append({
                'problem_id': pid,
                'model': model,
                'prompt_strategy': strategy,
                'system': meta.get('system', ''),
                'tier': meta.get('tier', 0),
                'presentation': meta.get('presentation', ''),
                'deontic': meta.get('deontic', False),
                'modal_depth': meta.get('modal_depth', 0),
                'num_worlds': meta.get('num_worlds', 0),
                'formula': meta.get('formula_symbolic', ''),
                'axiom_tag': meta.get('axiom_tag'),
                'paradox_name': meta.get('paradox_name'),
                'ground_truth': gt,
                'predicted': predicted,
                'correct': correct,
                'response': response,
            })

    df = pd.DataFrame(all_rows)
    print(f"\n✅ Unified DataFrame: {df.shape[0]} rows, {df.model.nunique()} models, "
          f"{df.prompt_strategy.nunique()} strategies")
    print(f"   Models: {sorted(df.model.unique())}")
    print(f"   Strategies: {sorted(df.prompt_strategy.unique())}")
    print(f"   Parse failures: {df.correct.isna().sum()} ({df.correct.isna().mean():.1%})")
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Full analysis
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(df, out_dir):
    """Run complete paper analysis and generate figures."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    sns.set_theme(style="whitegrid")

    def _save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(f"{fig_dir}/{name}.{ext}")
        plt.close(fig)
        print(f"  ✅ {name}")

    cot = df[df.prompt_strategy == "cot"]
    models = sorted(df.model.unique())

    # ── TABLE 1: Main results ──
    print("\n" + "=" * 72)
    print("TABLE 1: Main Results (CoT)")
    print("=" * 72)
    t1 = cot.groupby(["model", "system"])["correct"].mean().unstack()
    t1["Overall"] = cot.groupby("model")["correct"].mean()
    print(t1.round(3).to_string())
    t1.round(3).to_csv(f"{fig_dir}/table1_main.csv")

    # ── TABLE 2: Strategies ──
    print("\n" + "=" * 72)
    print("TABLE 2: Prompting Strategy Comparison")
    print("=" * 72)
    t2 = df.groupby(["model", "prompt_strategy"])["correct"].mean().unstack()
    print(t2.round(3).to_string())
    t2.round(3).to_csv(f"{fig_dir}/table2_strategies.csv")

    # ── TABLE 3: Formal vs NL ──
    print("\n" + "=" * 72)
    print("TABLE 3: Formal vs NL Gap")
    print("=" * 72)
    t3 = df.groupby(["model", "presentation"])["correct"].mean().unstack()
    if "formal" in t3.columns and "nl" in t3.columns:
        t3["gap_pp"] = (t3["formal"] - t3["nl"]) * 100
    print(t3.round(3).to_string())
    t3.round(3).to_csv(f"{fig_dir}/table3_formal_nl.csv")

    # ── TABLE 4: Per-axiom ──
    print("\n" + "=" * 72)
    print("TABLE 4: Per-Axiom Accuracy (CoT)")
    print("=" * 72)
    axiom_data = cot[cot.axiom_tag.notna()]
    if not axiom_data.empty:
        t4 = axiom_data.groupby(["model", "axiom_tag"])["correct"].mean().unstack()
        print(t4.round(3).to_string())
        t4.round(3).to_csv(f"{fig_dir}/table4_axiom.csv")

    # ── TABLE 5: Paradoxes ──
    print("\n" + "=" * 72)
    print("TABLE 5: Deontic Paradox Accuracy (CoT)")
    print("=" * 72)
    para_data = cot[cot.paradox_name.notna()]
    if not para_data.empty:
        t5 = para_data.groupby(["model", "paradox_name"])["correct"].mean().unstack()
        print(t5.round(3).to_string())
        t5.round(3).to_csv(f"{fig_dir}/table5_paradox.csv")

    # ── TABLE 6: Tiers ──
    print("\n" + "=" * 72)
    print("TABLE 6: Accuracy by Tier (CoT)")
    print("=" * 72)
    t6 = cot.groupby(["model", "tier"])["correct"].mean().unstack()
    print(t6.round(3).to_string())
    t6.round(3).to_csv(f"{fig_dir}/table6_tiers.csv")

    # ── Parse failure rates ──
    print("\n" + "=" * 72)
    print("PARSE FAILURE RATES")
    print("=" * 72)
    for m in models:
        pf = df[df.model == m]['correct'].isna().mean()
        print(f"  {m:30s}: {pf:.1%}")

    # ── Heuristic resistance ──
    print("\n" + "=" * 72)
    print("HEURISTIC RESISTANCE")
    print("=" * 72)
    gt_rate = df['ground_truth'].mean()
    print(f"  H1 (always True): {gt_rate:.1%} {'✅' if 0.4 < gt_rate < 0.6 else '⚠️'}")

    # ── FIGURES ──
    print("\n📈 Generating figures...")

    # Fig 1: Heatmap (system × tier per model), CoT only
    n = len(models)
    fig, axes = plt.subplots(1, min(n, 6), figsize=(5 * min(n, 6), 5))
    if n == 1: axes = [axes]
    for i, mn in enumerate(models[:6]):
        md = cot[cot.model == mn]
        pv = md.groupby(["system", "tier"])["correct"].mean().unstack().astype(float)
        sns.heatmap(pv, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.3, vmax=1.0, ax=axes[i], cbar_kws={'shrink': 0.8})
        axes[i].set_title(mn, fontsize=10, fontweight='bold')
    fig.suptitle("Accuracy by system × tier (CoT)", fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_heatmap")

    # Fig 2: Strategy comparison
    fig, ax = plt.subplots(figsize=(max(12, 3 * n), 6))
    t2_plot = df.groupby(["model", "prompt_strategy"])["correct"].mean().unstack().astype(float)
    t2_plot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Prompting strategy comparison", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.legend(title="Strategy")
    fig.tight_layout()
    _save(fig, "fig2_strategies")

    # Fig 3: Formal vs NL
    fig, ax = plt.subplots(figsize=(max(10, 3 * n), 6))
    x = np.arange(len(models)); w = 0.35
    fa = [df[(df.model == m) & (df.presentation == "formal")]["correct"].mean() for m in models]
    na = [df[(df.model == m) & (df.presentation == "nl")]["correct"].mean() for m in models]
    ax.bar(x - w / 2, fa, w, label='Formal', color='#3498db', edgecolor='black')
    ax.bar(x + w / 2, na, w, label='Natural Language', color='#e74c3c', edgecolor='black')
    for i in range(len(models)):
        gap = (fa[i] - na[i]) * 100
        ax.annotate(f'{gap:+.1f}pp', xy=(x[i], max(fa[i], na[i]) + 0.01),
                    ha='center', fontsize=9, color='purple', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Formal vs Natural Language", fontweight='bold')
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig3_formal_vs_nl")

    # Fig 4: S5 bias
    ax5_data = cot[(cot.axiom_tag == "5") & (cot.presentation == "formal")]
    if not ax5_data.empty:
        rows = []
        for mn in models:
            for sys in ["K", "T", "S4", "S5"]:
                sub = ax5_data[(ax5_data.model == mn) & (ax5_data.system == sys)]
                if sub.empty or sub.predicted.isna().all():
                    continue
                pred_true = (sub.predicted == True).mean()
                gt_true = sub.ground_truth.mean()
                rows.append(dict(model=mn, system=sys, bias=pred_true - gt_true))
        if rows:
            bdf = pd.DataFrame(rows)
            pv = bdf.pivot(index="model", columns="system", values="bias").astype(float)
            fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.8)))
            sns.heatmap(pv, annot=True, fmt='+.2f', cmap='RdBu_r', center=0,
                        ax=ax, vmin=-0.5, vmax=0.5)
            ax.set_title("Implicit S5 bias: axiom-5 over-application", fontweight='bold')
            fig.tight_layout()
            _save(fig, "fig4_s5_bias")

    # Fig 5: Accuracy vs num_worlds
    d = cot[cot.presentation == "formal"]
    if not d.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mn in models:
            md = d[d.model == mn]
            bw = md.groupby("num_worlds")["correct"].mean()
            ax.plot(bw.index, bw.values, 'o-', label=mn, lw=2, ms=6)
        ax.set_xlabel("Number of worlds"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs frame size", fontweight='bold')
        ax.legend(fontsize=8); ax.set_ylim(0.3, 1.0)
        fig.tight_layout()
        _save(fig, "fig5_worlds")

    # Fig 6: Accuracy vs modal_depth
    if not d.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mn in models:
            md = d[d.model == mn]
            bd = md.groupby("modal_depth")["correct"].mean()
            ax.plot(bd.index, bd.values, 'o-', label=mn, lw=2, ms=6)
        ax.set_xlabel("Modal nesting depth"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs nesting depth", fontweight='bold')
        ax.legend(fontsize=8); ax.set_ylim(0.3, 1.0)
        fig.tight_layout()
        _save(fig, "fig6_depth")

    # Fig 7: Tiers
    fig, ax = plt.subplots(figsize=(max(12, 3 * n), 6))
    pv = cot.groupby(["model", "tier"])["correct"].mean().unstack().astype(float)
    pv.columns = [f"Tier {c}" for c in pv.columns]
    pv.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Accuracy by difficulty tier", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    fig.tight_layout()
    _save(fig, "fig7_tiers")

    # Fig 8: Paradoxes
    if not para_data.empty:
        fig, ax = plt.subplots(figsize=(max(12, 3 * n), 5))
        pv = para_data.groupby(["model", "paradox_name"])["correct"].mean().unstack().astype(float)
        pv.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_ylabel("Accuracy"); ax.set_ylim(0.0, 1.1)
        ax.set_title("Deontic paradox accuracy (CoT)", fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        ax.axhline(0.5, color='red', ls='--', alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig8_paradoxes")

    # ── STATISTICAL TESTS ──
    print("\n" + "=" * 72)
    print("STATISTICAL TESTS")
    print("=" * 72)

    print("\n--- Formal vs NL (Welch t-test) ---")
    for mn in models:
        fa = df[(df.model == mn) & (df.presentation == "formal")]["correct"].dropna()
        na = df[(df.model == mn) & (df.presentation == "nl")]["correct"].dropna()
        if len(fa) < 10 or len(na) < 10: continue
        t, p = stats.ttest_ind(fa.values, na.values, equal_var=False)
        d = (fa.mean() - na.mean()) / np.sqrt((fa.var() + na.var()) / 2)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {mn}: formal={fa.mean():.3f} nl={na.mean():.3f} "
              f"Δ={fa.mean() - na.mean():+.3f} t={t:.2f} p={p:.2e} {sig} d={d:+.2f}")

    print("\n--- Strategy effect (one-way ANOVA) ---")
    for mn in models:
        md = df[df.model == mn]
        groups = [md[md.prompt_strategy == s]["correct"].dropna().values
                  for s in sorted(df.prompt_strategy.unique())]
        groups = [g for g in groups if len(g) > 10]
        if len(groups) < 2: continue
        f_stat, p = stats.f_oneway(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {mn}: F={f_stat:.2f} p={p:.2e} {sig}")

    print("\n--- Tier difficulty (one-way ANOVA, CoT) ---")
    for mn in models:
        md = cot[cot.model == mn]
        groups = [md[md.tier == t]["correct"].dropna().values for t in [1, 2, 3]]
        if not all(len(g) > 10 for g in groups): continue
        f_stat, p = stats.f_oneway(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {mn}: T1={groups[0].mean():.3f} T2={groups[1].mean():.3f} "
              f"T3={groups[2].mean():.3f} F={f_stat:.2f} p={p:.2e} {sig}")

    # ── KEY STATS ──
    print("\n" + "=" * 72)
    print("📝 KEY STATISTICS FOR PAPER")
    print("=" * 72)

    overall_cot = cot.groupby("model")["correct"].mean()
    print(f"\nBest (CoT):  {overall_cot.idxmax()} = {overall_cot.max():.1%}")
    print(f"Worst (CoT): {overall_cot.idxmin()} = {overall_cot.min():.1%}")

    t1_acc = cot[cot.tier == 1]["correct"].mean()
    t3_acc = cot[cot.tier == 3]["correct"].mean()
    print(f"Tier drop: {t1_acc:.1%} → {t3_acc:.1%} ({(t1_acc - t3_acc) * 100:.1f}pp)")

    fa_all = df[df.presentation == "formal"]["correct"].mean()
    na_all = df[df.presentation == "nl"]["correct"].mean()
    print(f"Formal vs NL: {fa_all:.1%} vs {na_all:.1%} ({(fa_all - na_all) * 100:.1f}pp)")

    strat = df.groupby("prompt_strategy")["correct"].mean()
    print(f"Strategies: {strat.round(3).to_dict()}")

    print(f"\nParse failure: {df.correct.isna().mean():.1%}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build unified ModalBench results")
    parser.add_argument("--input_dir", "-i", default=INPUT_DIR,
                        help="Directory containing JSONL result files")
    parser.add_argument("--output_dir", "-o", default=OUTPUT_DIR,
                        help="Output directory for CSV and figures")
    parser.add_argument("--bench", "-b", default=BENCH_PATH,
                        help="Path to bench_final.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load benchmark
    bench, lookup = load_benchmark(args.bench)

    # Build unified DataFrame
    df = build_unified_df(args.input_dir, lookup, FILE_MAP)

    # Save unified CSV
    csv_path = os.path.join(args.output_dir, "results_unified.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")

    # Run full analysis
    df = run_analysis(df, args.output_dir)

    print(f"\n🎉 Done! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
