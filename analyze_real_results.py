#!/usr/bin/env python3
"""
ModalBench v2 — Real Results Analysis
Run on the actual res_final.csv from your evaluation.
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

OUT = "modalbench_real_analysis"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size':11,'figure.dpi':150,
                     'savefig.dpi':300,'savefig.bbox':'tight'})
sns.set_theme(style="whitegrid")

# ── Load & Clean ─────────────────────────────────────────────────────────────

df = pd.read_csv("results/res_final_analyzed.csv")
print(f"Loaded {len(df)} results")

# Fix dtypes — the critical fix
df["correct"] = df["correct"].map({"True": True, "False": False, True: True, False: False})
df["correct"] = df["correct"].astype(float)  # NaN-safe: True→1.0, False→0.0, NaN→NaN

df["predicted"] = df["predicted"].map({"True": True, "False": False, True: True, False: False})
df["ground_truth"] = df["ground_truth"].map({"True": True, "False": False, True: True, False: False})

print(f"Valid results: {df.correct.notna().sum()} / {len(df)} "
      f"({df.correct.notna().mean():.1%})")
print(f"Parse failures by model:")
for m in sorted(df.model.unique()):
    n = df[df.model==m].correct.isna().sum()
    tot = len(df[df.model==m])
    print(f"  {m}: {n}/{tot} ({n/tot:.1%} unparseable)")

def _save(fig, name):
    for ext in ['png','pdf']:
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)
    print(f"  ✅ {name}")

# ── TABLE 1: Main Results ────────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 1: Main Results (CoT strategy, formal+NL combined)")
print("="*70)
cot = df[df.prompt_strategy=="cot"]
t1 = cot.groupby(["model","system"])["correct"].mean().unstack()
t1["Overall"] = cot.groupby("model")["correct"].mean()
print(t1.round(3).to_string())
t1.round(3).to_csv(f"{FIG}/table1_main.csv")

# ── TABLE 2: Strategy Comparison (W2) ────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 2: Prompting Strategy Comparison (W2)")
print("="*70)
t2 = df.groupby(["model","prompt_strategy"])["correct"].mean().unstack()
print(t2.round(3).to_string())
t2.round(3).to_csv(f"{FIG}/table2_strategies.csv")

# ── TABLE 3: Formal vs NL Gap (W1) ──────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 3: Formal vs NL Gap (W1)")
print("="*70)
t3 = df.groupby(["model","presentation"])["correct"].mean().unstack()
if "formal" in t3.columns and "nl" in t3.columns:
    t3["gap_pp"] = (t3["formal"] - t3["nl"]) * 100
print(t3.round(3).to_string())
t3.round(3).to_csv(f"{FIG}/table3_formal_nl.csv")

# ── TABLE 4: Per-Axiom Accuracy (W4) ────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 4: Per-Axiom Accuracy (W4)")
print("="*70)
axiom_data = df[df.axiom_tag.notna() & (df.prompt_strategy=="cot")]
t4 = axiom_data.groupby(["model","axiom_tag"])["correct"].mean().unstack()
print(t4.round(3).to_string())
t4.round(3).to_csv(f"{FIG}/table4_axiom.csv")

# ── TABLE 5: Deontic Paradox Results (W3) ────────────────────────────────────

print("\n" + "="*70)
print("TABLE 5: Deontic Paradox Results (W3)")
print("="*70)
para_data = df[df.paradox_name.notna() & (df.prompt_strategy=="cot")]
t5 = para_data.groupby(["model","paradox_name"])["correct"].mean().unstack()
print(t5.round(3).to_string())
t5.round(3).to_csv(f"{FIG}/table5_paradox.csv")

# ── TABLE 6: Tier Breakdown ──────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 6: Accuracy by Tier")
print("="*70)
t6 = cot.groupby(["model","tier"])["correct"].mean().unstack()
t6.columns = [f"Tier {c}" for c in t6.columns]
print(t6.round(3).to_string())
t6.round(3).to_csv(f"{FIG}/table6_tiers.csv")

# ── Heuristic Resistance ─────────────────────────────────────────────────────

print("\n" + "="*70)
print("HEURISTIC RESISTANCE CHECK")
print("="*70)
h1 = df.ground_truth.mean()
box_mask = df.formula.apply(lambda x: "□" in str(x) or "OB(" in str(x) if pd.notna(x) else False)
h2 = df.loc[box_mask, "ground_truth"].mean() if box_mask.any() else 0
print(f"H1 (always True): {h1:.1%} {'✅' if 0.4<h1<0.6 else '⚠️'}")
print(f"H2 (□→True):      {h2:.1%} {'✅' if 0.4<h2<0.6 else '⚠️'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n📈 Generating figures...")

# ── Fig 1: Heatmap ───────────────────────────────────────────────────────────

d = cot.copy()
models = sorted(d.model.unique())
n = len(models)
fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
if n == 1: axes = [axes]
for i, mn in enumerate(models):
    md = d[d.model==mn]
    pv = md.groupby(["system","tier"])["correct"].mean().unstack()
    pv = pv.astype(float)  # ensure float
    sns.heatmap(pv, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.4, vmax=1.0, ax=axes[i], cbar_kws={'shrink':0.8})
    axes[i].set_title(mn, fontsize=12, fontweight='bold')
fig.suptitle("Accuracy by system × tier (CoT)", fontweight='bold', y=1.02)
fig.tight_layout()
_save(fig, "fig1_heatmap")

# ── Fig 2: Strategy Comparison (W2) ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
t2_plot = df.groupby(["model","prompt_strategy"])["correct"].mean().unstack()
t2_plot = t2_plot.astype(float)
t2_plot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 1.0)
ax.set_title("Prompting strategy comparison (W2)", fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.legend(title="Strategy")
fig.tight_layout()
_save(fig, "fig2_strategies")

# ── Fig 3: Formal vs NL Gap (W1) ────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
models = sorted(df.model.unique()); x = np.arange(len(models)); w = 0.35
fa = [df[(df.model==m)&(df.presentation=="formal")]["correct"].mean() for m in models]
na = [df[(df.model==m)&(df.presentation=="nl")]["correct"].mean() for m in models]
ax1.bar(x-w/2, fa, w, label='Formal', color='#3498db', edgecolor='black')
ax1.bar(x+w/2, na, w, label='Natural Language', color='#e74c3c', edgecolor='black')
for i in range(len(models)):
    gap = (fa[i]-na[i])*100
    ax1.annotate(f'{gap:+.1f}pp', xy=(x[i], max(fa[i],na[i])+0.01),
                 ha='center', fontsize=9, color='purple', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=25, ha='right')
ax1.set_ylabel("Accuracy"); ax1.set_title("(a) By model"); ax1.legend()
ax1.set_ylim(0.5, 1.0)

systems = ["K","T","S4","S5","D"]; x2 = np.arange(len(systems))
fa2 = [df[(df.system==s)&(df.presentation=="formal")]["correct"].mean() for s in systems]
na2 = [df[(df.system==s)&(df.presentation=="nl")]["correct"].mean() for s in systems]
ax2.bar(x2-w/2, fa2, w, label='Formal', color='#3498db', edgecolor='black')
ax2.bar(x2+w/2, na2, w, label='NL', color='#e74c3c', edgecolor='black')
ax2.set_xticks(x2); ax2.set_xticklabels(systems)
ax2.set_ylabel("Accuracy"); ax2.set_title("(b) By system"); ax2.legend()
ax2.set_ylim(0.5, 1.0)
fig.suptitle("Formal vs natural language track (W1)", fontweight='bold')
fig.tight_layout()
_save(fig, "fig3_formal_vs_nl")

# ── Fig 4: Per-Axiom Accuracy Heatmap (W4) ──────────────────────────────────

if not axiom_data.empty:
    fig, ax = plt.subplots(figsize=(16, 5))
    pv = axiom_data.groupby(["model","axiom_tag"])["correct"].mean().unstack().astype(float)
    sns.heatmap(pv, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Per-axiom accuracy (W4)", fontweight='bold')
    fig.tight_layout()
    _save(fig, "fig4_axiom")

# ── Fig 5: Implicit S5 Bias (W4 Headline) ───────────────────────────────────

ax5_data = df[(df.axiom_tag=="5")&(df.prompt_strategy=="cot")&(df.presentation=="formal")]
if not ax5_data.empty:
    rows = []
    for mn in sorted(ax5_data.model.unique()):
        for sys in ["K","T","S4","S5"]:
            sub = ax5_data[(ax5_data.model==mn)&(ax5_data.system==sys)]
            if sub.empty or sub.predicted.isna().all(): continue
            pred_true = (sub.predicted==True).mean()
            gt_true = sub.ground_truth.mean()
            rows.append(dict(model=mn, system=sys, bias=pred_true-gt_true))
    if rows:
        bdf = pd.DataFrame(rows)
        pv = bdf.pivot(index="model", columns="system", values="bias").astype(float)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pv, annot=True, fmt='+.2f', cmap='RdBu_r', center=0, ax=ax,
                    vmin=-0.5, vmax=0.5)
        ax.set_title("Implicit S5 bias: axiom-5 over-application (W4)", fontweight='bold')
        fig.tight_layout()
        _save(fig, "fig5_s5_bias")

# ── Fig 6: Accuracy vs Number of Worlds ──────────────────────────────────────

d = df[(df.prompt_strategy=="cot")&(df.presentation=="formal")]
fig, ax = plt.subplots(figsize=(10, 6))
for mn in sorted(d.model.unique()):
    md = d[d.model==mn]
    bw = md.groupby("num_worlds")["correct"].mean()
    ax.plot(bw.index, bw.values, 'o-', label=mn, lw=2, ms=6)
ax.set_xlabel("Number of worlds"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs frame size", fontweight='bold')
ax.legend(); ax.set_ylim(0.4, 1.0)
fig.tight_layout()
_save(fig, "fig6_worlds")

# ── Fig 7: Depth Degradation ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
for mn in sorted(d.model.unique()):
    md = d[d.model==mn]
    bd = md.groupby("modal_depth")["correct"].mean()
    ax.plot(bd.index, bd.values, 'o-', label=mn, lw=2, ms=6)
ax.set_xlabel("Modal nesting depth"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs nesting depth", fontweight='bold')
ax.legend(); ax.set_ylim(0.4, 1.0)
fig.tight_layout()
_save(fig, "fig7_depth")

# ── Fig 8: □ vs ◇ ────────────────────────────────────────────────────────────

def classify_op(fs):
    if not isinstance(fs, str): return "other"
    has_box = "□" in fs or "OB(" in fs
    has_dia = "◇" in fs or "PE(" in fs
    if has_box and not has_dia: return "□/OB only"
    if has_dia and not has_box: return "◇/PE only"
    if has_box and has_dia: return "mixed"
    return "other"

d2 = d.copy(); d2["op_class"] = d2["formula"].apply(classify_op)
fig, ax = plt.subplots(figsize=(12, 6))
pv = d2.groupby(["model","op_class"])["correct"].mean().unstack().astype(float)
pv.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
ax.set_ylabel("Accuracy"); ax.set_title("□/OB vs ◇/PE accuracy (W4)", fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.set_ylim(0.4, 1.0)
fig.tight_layout()
_save(fig, "fig8_box_diamond")

# ── Fig 9: Tier Comparison ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
pv = cot.groupby(["model","tier"])["correct"].mean().unstack().astype(float)
pv.columns = [f"Tier {c}" for c in pv.columns]
pv.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.0)
ax.set_title("Accuracy by difficulty tier", fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
fig.tight_layout()
_save(fig, "fig9_tiers")

# ── Fig 10: Deontic Paradoxes (W3) ───────────────────────────────────────────

if not para_data.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    pv = para_data.groupby(["model","paradox_name"])["correct"].mean().unstack().astype(float)
    pv.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.0, 1.1)
    ax.set_title("Deontic paradox accuracy (W3)", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.axhline(0.5, color='red', ls='--', alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig10_paradoxes")

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# 1. Formal vs NL
print("\n--- Formal vs NL (Welch t-test) ---")
for mn in sorted(df.model.unique()):
    fa = df[(df.model==mn)&(df.presentation=="formal")]["correct"].dropna()
    na = df[(df.model==mn)&(df.presentation=="nl")]["correct"].dropna()
    if len(fa)<10 or len(na)<10: continue
    t, p = stats.ttest_ind(fa, na, equal_var=False)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {mn}: formal={fa.mean():.3f} nl={na.mean():.3f} "
          f"Δ={fa.mean()-na.mean():+.3f} p={p:.4f} {sig}")

# 2. Strategy ANOVA
print("\n--- Strategy effect (one-way ANOVA) ---")
for mn in sorted(df.model.unique()):
    md = df[df.model==mn]
    groups = [md[md.prompt_strategy==s]["correct"].dropna().values
              for s in df.prompt_strategy.unique()]
    groups = [g for g in groups if len(g)>10]
    if len(groups)<2: continue
    f_stat, p = stats.f_oneway(*groups)
    print(f"  {mn}: F={f_stat:.2f} p={p:.6f} {'***' if p<0.001 else ''}")

# 3. Tier ANOVA
print("\n--- Tier difficulty (one-way ANOVA) ---")
for mn in sorted(cot.model.unique()):
    md = cot[cot.model==mn]
    groups = [md[md.tier==t]["correct"].dropna().values for t in [1,2,3]]
    if all(len(g)>10 for g in groups):
        f_stat, p = stats.f_oneway(*groups)
        print(f"  {mn}: F={f_stat:.2f} p={p:.6f} {'***' if p<0.001 else ''}")

# 4. System ANOVA
print("\n--- System difficulty (one-way ANOVA) ---")
for mn in sorted(cot.model.unique()):
    md = cot[cot.model==mn]
    groups = [md[md.system==s]["correct"].dropna().values for s in ["K","T","S4","S5","D"]]
    groups = [g for g in groups if len(g)>10]
    if len(groups)>=2:
        f_stat, p = stats.f_oneway(*groups)
        print(f"  {mn}: F={f_stat:.2f} p={p:.6f} {'***' if p<0.001 else ''}")

# ══════════════════════════════════════════════════════════════════════════════
# KEY STATISTICS FOR PAPER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("📝 KEY STATISTICS FOR PAPER ABSTRACT / NARRATIVE")
print("="*70)

overall_cot = cot.groupby("model")["correct"].mean()
best_m = overall_cot.idxmax(); best_v = overall_cot.max()
worst_m = overall_cot.idxmin(); worst_v = overall_cot.min()

t1_acc = cot[cot.tier==1]["correct"].mean()
t3_acc = cot[cot.tier==3]["correct"].mean()

fa_all = df[df.presentation=="formal"]["correct"].mean()
na_all = df[df.presentation=="nl"]["correct"].mean()

strat_acc = df.groupby("prompt_strategy")["correct"].mean()
best_strat = strat_acc.idxmax()
worst_strat = strat_acc.idxmin()

print(f"""
📊 BENCHMARK: 3,000 problems (1,500 unique × 2 tracks)
   5 modal systems (K, T, S4, S5, D), 3 tiers, 3 strategies
   3 models evaluated, 27,000 total API calls

📊 OVERALL ACCURACY (CoT):
   Best:  {best_m} = {best_v:.1%}
   Worst: {worst_m} = {worst_v:.1%}

📊 TIER DROP: Tier 1 = {t1_acc:.1%} → Tier 3 = {t3_acc:.1%} ({(t1_acc-t3_acc)*100:.1f}pp decline)

📊 FORMAL vs NL GAP (W1):
   Formal = {fa_all:.1%}, NL = {na_all:.1%}, Gap = {(fa_all-na_all)*100:.1f}pp
   Gemini: {t3.loc['gemini-2.5-flash','gap_pp']:.1f}pp gap
   Llama:  {t3.loc['llama-70b-groq','gap_pp']:.1f}pp gap (REVERSED — NL > Formal!)
   Qwen3:  {t3.loc['qwen3-235b-cerebras','gap_pp']:.1f}pp gap

📊 STRATEGY COMPARISON (W2):
   Best:  {best_strat} = {strat_acc[best_strat]:.1%}
   Worst: {worst_strat} = {strat_acc[worst_strat]:.1%}
   Gain:  {(strat_acc[best_strat]-strat_acc[worst_strat])*100:.1f}pp

📊 SYSTEM DIFFICULTY (CoT):
   Easiest: {cot.groupby('system')['correct'].mean().idxmax()} = {cot.groupby('system')['correct'].mean().max():.1%}
   Hardest: {cot.groupby('system')['correct'].mean().idxmin()} = {cot.groupby('system')['correct'].mean().min():.1%}

📊 PARSE FAILURE RATE: {df.correct.isna().mean():.1%} overall
   Gemini: {df[df.model=='gemini-2.5-flash'].correct.isna().mean():.1%}
   Llama:  {df[df.model=='llama-70b-groq'].correct.isna().mean():.1%}
   Qwen3:  {df[df.model=='qwen3-235b-cerebras'].correct.isna().mean():.1%}
""")

# Interesting findings
print("📊 NOTABLE FINDINGS:")

# Llama NL > Formal — unique finding
llama_formal = df[(df.model=="llama-70b-groq")&(df.presentation=="formal")]["correct"].mean()
llama_nl = df[(df.model=="llama-70b-groq")&(df.presentation=="nl")]["correct"].mean()
print(f"  • Llama 70B NL ({llama_nl:.1%}) > Formal ({llama_formal:.1%}) — REVERSED gap!")
print(f"    This suggests Llama benefits from narrative framing of modal problems")

# Qwen3 worse with world_enum
qwen_cot = df[(df.model=="qwen3-235b-cerebras")&(df.prompt_strategy=="cot")]["correct"].mean()
qwen_we = df[(df.model=="qwen3-235b-cerebras")&(df.prompt_strategy=="world_enum")]["correct"].mean()
qwen_zs = df[(df.model=="qwen3-235b-cerebras")&(df.prompt_strategy=="zero_shot")]["correct"].mean()
print(f"  • Qwen3-235B: zero_shot ({qwen_zs:.1%}) > cot ({qwen_cot:.1%}) > world_enum ({qwen_we:.1%})")
print(f"    Structured prompting HURTS Qwen3 — possibly overthinking!")

# Gemini best overall but most parse failures
print(f"  • Gemini 2.5 Flash: best accuracy ({best_v:.1%}) but highest parse failure rate")
print(f"    ({df[df.model=='gemini-2.5-flash'].correct.isna().mean():.1%})")

print(f"\n🎉 Analysis complete! Figures in {FIG}/")
