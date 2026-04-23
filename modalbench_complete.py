# %load ModalBench_v2_Complete.py
#!/usr/bin/env python3
"""
================================================================================
ModalBench v2 — Complete Codebase (Post-Review)
================================================================================
ICLR 2026 Workshop: Logical Reasoning of Large Language Models

Single-file pipeline. Run top-to-bottom in Google Colab.

Addresses all review feedback:
  W1: Dual-track presentation (formal + natural language)
  W2: 4 prompting strategies compared
  W3: Deontic paradoxes (Ross, Chisholm, Good Samaritan, Gentle Murderer)
  W4: Deep diagnostic analysis (per-axiom, □ vs ◇, implicit S5 bias, etc.)
  W5: Related work positioning (paper text, not code)
  + Free model evaluation (Gemini, Groq, Cerebras, OpenRouter)
  + Google Drive save/restore
  + Stratified sampling for efficiency
================================================================================
"""

# ── Cell 1: Install & Import ─────────────────────────────────────────────────
# !pip install -q openai google-generativeai matplotlib seaborn scipy tqdm
# !pip install -q google-genai openai
import os, sys, json, re, random, time, math, hashlib, itertools
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas as pd

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
# Add this to your imports in Cell 1
import asyncio
from openai import AsyncOpenAI

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it

SEED = 42
random.seed(SEED); np.random.seed(SEED)
print("✅ Imports done")





# ── Configuration ────────────────────────────────────────────────────────────
class C:
    """Central configuration — edit freely or override via env vars."""
    # API keys (set via os.environ before importing this module)
    GOOGLE_KEY     = os.environ.get("GOOGLE_API_KEY", "")
    GROQ_KEY       = os.environ.get("GROQ_API_KEY", "")
    CEREBRAS_KEY   = os.environ.get("CEREBRAS_API_KEY", "")
    OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

    # Benchmark
    WORLDS_RANGE = (2, 7)
    PROPS_RANGE  = (2, 5)
    PROBLEMS_PER_CELL = 100   # per system×tier — total ≈ 1500

    # Evaluation
    MAX_RETRIES = 8
    RETRY_DELAY = 2
    TEMPERATURE = 0.0
    MAX_TOKENS  = 1024

    # Paths
    OUT = "modalbench_v2"
    FIG = "figures"

os.makedirs(C.OUT, exist_ok=True)
os.makedirs(f"{C.OUT}/{C.FIG}", exist_ok=True)

# ── Cell 3: Kripke Semantics Engine ──────────────────────────────────────────

class Sys(Enum):
    K="K"; T="T"; S4="S4"; S5="S5"; D="D"

@dataclass
class Frame:
    worlds: List[str]
    rels: List[Tuple[str,str]]
    system: Sys
    def acc(self, w): return {b for a,b in self.rels if a==w}
    def is_reflexive(self):
        s=set(self.rels); return all((w,w) in s for w in self.worlds)
    def is_transitive(self):
        s=set(self.rels)
        for a,b in self.rels:
            for b2,c in self.rels:
                if b==b2 and (a,c) not in s: return False
        return True
    def is_serial(self):
        src={a for a,_ in self.rels}; return all(w in src for w in self.worlds)
    def is_euclidean(self):
        s=set(self.rels)
        for w in self.worlds:
            a=self.acc(w)
            for x,y in itertools.product(a,a):
                if (x,y) not in s: return False
        return True
    def density(self):
        n=len(self.worlds); return len(self.rels)/(n*n) if n else 0

@dataclass
class Model:
    frame: Frame
    val: Dict[str,Dict[str,bool]]
    props: List[str]

class FT(Enum):
    ATOM="atom"; NOT="not"; AND="and"; OR="or"; IMP="imp"
    BOX="box"; DIA="dia"; OB="ob"; PE="pe"; FO="fo"

@dataclass
class F:
    t: FT
    atom: Optional[str]=None
    sub: Optional['F']=None
    l: Optional['F']=None
    r: Optional['F']=None

    def ev(self, m: Model, w: str) -> bool:
        if self.t==FT.ATOM: return m.val[w].get(self.atom, False)
        if self.t==FT.NOT:  return not self.sub.ev(m,w)
        if self.t==FT.AND:  return self.l.ev(m,w) and self.r.ev(m,w)
        if self.t==FT.OR:   return self.l.ev(m,w) or self.r.ev(m,w)
        if self.t==FT.IMP:  return (not self.l.ev(m,w)) or self.r.ev(m,w)
        if self.t in (FT.BOX, FT.OB):
            a=m.frame.acc(w); return all(self.sub.ev(m,x) for x in a) if a else True
        if self.t in (FT.DIA, FT.PE):
            a=m.frame.acc(w); return any(self.sub.ev(m,x) for x in a)
        if self.t==FT.FO:
            return F(FT.BOX, sub=F(FT.NOT, sub=self.sub)).ev(m,w)
        raise ValueError(self.t)

    def s(self, d=False) -> str:
        if self.t==FT.ATOM: return self.atom
        if self.t==FT.NOT:
            i=self.sub.s(d)
            return f"¬{i}" if self.sub.t==FT.ATOM else f"¬({i})"
        if self.t==FT.AND:  return f"({self.l.s(d)} ∧ {self.r.s(d)})"
        if self.t==FT.OR:   return f"({self.l.s(d)} ∨ {self.r.s(d)})"
        if self.t==FT.IMP:  return f"({self.l.s(d)} → {self.r.s(d)})"
        if self.t==FT.BOX:  return f"{'OB' if d else '□'}({self.sub.s(d)})"
        if self.t==FT.DIA:  return f"{'PE' if d else '◇'}({self.sub.s(d)})"
        if self.t==FT.OB:   return f"OB({self.sub.s(True)})"
        if self.t==FT.PE:   return f"PE({self.sub.s(True)})"
        if self.t==FT.FO:   return f"FO({self.sub.s(True)})"
        return "?"

    def depth(self) -> int:
        if self.t==FT.ATOM: return 0
        if self.sub is not None:
            d = 1 if self.t in (FT.BOX,FT.DIA,FT.OB,FT.PE,FT.FO) else 0
            return d + self.sub.depth()
        return max(self.l.depth(), self.r.depth())

    def complexity(self) -> int:
        if self.t==FT.ATOM: return 1
        if self.sub is not None: return 1+self.sub.complexity()
        return 1+self.l.complexity()+self.r.complexity()

# Constructors
def A(n):       return F(FT.ATOM, atom=n)
def N(f):       return F(FT.NOT, sub=f)
def An(a,b):    return F(FT.AND, l=a, r=b)
def Or(a,b):    return F(FT.OR, l=a, r=b)
def Imp(a,b):   return F(FT.IMP, l=a, r=b)
def Bx(f):      return F(FT.BOX, sub=f)
def Di(f):      return F(FT.DIA, sub=f)
def Ob(f):      return F(FT.OB, sub=f)
def Pe(f):      return F(FT.PE, sub=f)
def Fo(f):      return F(FT.FO, sub=f)

print("✅ Kripke engine loaded")



# ── Cell 4: Frame & Model Generators ─────────────────────────────────────────

class FG:
    """Frame generators per modal system."""
    @staticmethod
    def _w(n): return [f"w{i}" for i in range(n)]

    @staticmethod
    def K(n):
        w=FG._w(n); r=[(a,b) for a in w for b in w if random.random()<0.4]
        if not r: r=[(random.choice(w), random.choice(w))]
        return Frame(w,r,Sys.K)

    @staticmethod
    def T(n):
        w=FG._w(n); r=[(x,x) for x in w]
        r+=[(a,b) for a in w for b in w if a!=b and random.random()<0.35]
        return Frame(w,list(set(r)),Sys.T)

    @staticmethod
    def S4(n):
        w=FG._w(n); s={(x,x) for x in w}
        for a in w:
            for b in w:
                if a!=b and random.random()<0.3: s.add((a,b))
        changed=True
        while changed:
            changed=False; new=set()
            for a,b in s:
                for b2,c in s:
                    if b==b2 and (a,c) not in s: new.add((a,c)); changed=True
            s|=new
        return Frame(w,list(s),Sys.S4)

    @staticmethod
    def S5(n):
        w=FG._w(n); nc=random.randint(1,min(3,n)); random.shuffle(w)
        cls=[[] for _ in range(nc)]
        for i,x in enumerate(w): cls[i%nc].append(x)
        s=set()
        for c in cls:
            for a in c:
                for b in c: s.add((a,b))
        return Frame(w,list(s),Sys.S5)

    @staticmethod
    def D(n):
        w=FG._w(n); s=set()
        for x in w:
            for t in random.sample(w, random.randint(1,max(1,n//2))):
                s.add((x,t))
        for a in w:
            for b in w:
                if random.random()<0.2: s.add((a,b))
        return Frame(w,list(s),Sys.D)

    @staticmethod
    def make(sys, n):
        return {Sys.K:FG.K, Sys.T:FG.T, Sys.S4:FG.S4, Sys.S5:FG.S5, Sys.D:FG.D}[sys](n)

    @staticmethod
    def validate(frame):
        s=frame.system
        if s==Sys.T and not frame.is_reflexive(): return False
        if s==Sys.S4 and not (frame.is_reflexive() and frame.is_transitive()): return False
        if s==Sys.S5 and not (frame.is_reflexive() and frame.is_transitive() and frame.is_euclidean()): return False
        if s==Sys.D and not frame.is_serial(): return False
        return True


ALETHIC_PROPS = ["p","q","r","s","t"]
DEONTIC_PROPS = ["paying_taxes","keeping_promises","telling_truth",
                 "helping_others","following_rules","respecting_privacy",
                 "reporting_crimes","recycling","voting","being_honest"]

def make_model(frame, n_props, deontic=False):
    pool = DEONTIC_PROPS if deontic else ALETHIC_PROPS
    props = random.sample(pool, min(n_props, len(pool)))
    val = {w: {p: random.random()<0.5 for p in props} for w in frame.worlds}
    return Model(frame, val, props)

print("✅ Generators loaded")



# ── Cell 5: Formula Pools (3 Tiers) + Deontic Paradoxes (W3) ────────────────

def tier1(props, deontic=False):
    fs=[]
    for pn in props:
        p=A(pn)
        if deontic:
            fs+=[Ob(p), Pe(p), Fo(p), Ob(N(p)), Pe(N(p))]
        else:
            fs+=[Bx(p), Di(p), Bx(N(p)), Di(N(p)), N(Bx(p)), N(Di(p))]
    if len(props)>=2:
        p,q=A(props[0]),A(props[1])
        if deontic: fs+=[Ob(An(p,q)), Pe(Or(p,q))]
        else: fs+=[Bx(An(p,q)), Di(An(p,q)), Bx(Or(p,q)), Di(Or(p,q)), Bx(Imp(p,q))]
    return [{"f":f, "tag":None, "key":f"T1:{f.s(deontic)}", "para":None} for f in fs if f.depth()<=1]

def tier2(props, deontic=False):
    fs=[]
    for pn in props:
        p=A(pn)
        if deontic:
            fs+=[Ob(Pe(p)), Pe(Ob(p)), Ob(Ob(p)), Fo(Pe(p))]
        else:
            fs+=[Bx(Bx(p)), Di(Di(p)), Bx(Di(p)), Di(Bx(p)), Bx(Di(N(p))), N(Bx(Di(p)))]
    if len(props)>=2:
        p,q=A(props[0]),A(props[1])
        if deontic: fs+=[Ob(An(Pe(p),Pe(q))), Imp(Ob(p),Pe(q))]
        else: fs+=[Bx(An(Di(p),Di(q))), Di(An(Bx(p),Di(q))), Imp(Bx(p),Di(q)), Bx(Imp(p,Di(q)))]
    return [{"f":f, "tag":None, "key":f"T2:{f.s(deontic)}", "para":None} for f in fs if f.depth()>=2]

def tier3_alethic(props, sys):
    p=A(props[0]); q=A(props[1]) if len(props)>=2 else A(props[0])
    items = [
        (Imp(Bx(p),p),                    "T",  sys in (Sys.T,Sys.S4,Sys.S5)),
        (Imp(Bx(p),Bx(Bx(p))),           "4",  sys in (Sys.S4,Sys.S5)),
        (Imp(Di(p),Bx(Di(p))),            "5",  sys==Sys.S5),
        (Imp(p, Bx(Di(p))),               "B",  sys==Sys.S5),
        (Imp(Bx(Imp(p,q)),Imp(Bx(p),Bx(q))), "K", True),
        (Imp(Bx(p),N(Di(N(p)))),          "dual", True),
        (Imp(N(Di(N(p))),Bx(p)),          "dual_rev", True),
        (Imp(Bx(An(p,q)),An(Bx(p),Bx(q))), "dist", True),
        (Imp(Di(Bx(p)),Bx(Di(p))),        "Barcan", False),
        (Imp(Bx(Bx(Bx(p))),Bx(p)),        "triple", sys in (Sys.S4,Sys.S5)),
        (Imp(Bx(Di(p)),Di(Bx(p))),        "McKinsey", False),
        # Non-axiom formulas for label balance
        (An(Bx(p),Di(N(p))),              None, False),
        (Or(Bx(p),Bx(N(p))),             None, False),
        (Imp(Di(p),p),                    "conv_T", False),  # converse of T — NOT valid
    ]
    return [{"f":f,"tag":tag,"key":f"T3:{tag or f.s()}","para":None} for f,tag,_ in items]

def tier3_deontic(props, sys):
    p=A(props[0]); q=A(props[1]) if len(props)>=2 else A(props[0])
    items = [
        (Imp(Ob(p),Pe(p)),                           "D",       None),
        (Imp(Ob(p),Di(p)),                            "ought_can",None),
        (N(An(Ob(p),Ob(N(p)))),                      "no_conflict",None),
        (Imp(Ob(p),Ob(Or(p,q))),                     "Ross",     "Ross"),
        (Imp(Fo(p),N(Pe(p))),                         "FO_PE",   None),
        (Imp(Ob(Ob(p)),Ob(p)),                        "OB_4",    None),
        (Imp(Pe(An(p,q)),An(Pe(p),Pe(q))),            "PE_dist", None),
        # Deontic paradoxes (W3)
        (Imp(Ob(An(p,N(q))),An(Ob(p),Ob(N(q)))),     "GoodSam", "Good_Samaritan"),
        (Imp(An(Ob(p),N(p)),Ob(N(q))),                "Chisholm","Chisholm"),
        (An(Ob(p),Imp(N(p),Ob(q))),                   "Gentle",  "Gentle_Murderer"),
        (An(Fo(p),Pe(p)),                              "FO_PE_contra","FO_PE_Contradiction"),
    ]
    return [{"f":f,"tag":tag,"key":f"T3D:{tag}","para":para} for f,tag,para in items]

def get_pool(props, sys, tier, deontic):
    if tier==1: return tier1(props, deontic)
    if tier==2: return tier2(props, deontic)
    return tier3_deontic(props,sys) if deontic else tier3_alethic(props,sys)

print("✅ Formula pools loaded (incl. deontic paradoxes)")



# ── Cell 6: Dual-Track Description Generators (W1) ──────────────────────────

WNAMES = {f"w{i}": n for i,n in enumerate(
    ["World Alpha","World Beta","World Gamma","World Delta",
     "World Epsilon","World Zeta","World Eta"])}
DSCEN = dict(paying_taxes="paying taxes", keeping_promises="keeping promises",
    telling_truth="telling the truth", helping_others="helping others",
    following_rules="following rules", respecting_privacy="respecting privacy",
    reporting_crimes="reporting crimes", recycling="recycling",
    voting="voting", being_honest="being honest")

def formal_desc(model, formula, world, deontic=False):
    """FORMAL track: explicit worlds, relations, valuations."""
    lines = []
    fr = model.frame
    lines.append(f"There are {len(fr.worlds)} possible worlds: " +
                 ", ".join(WNAMES.get(w,w) for w in fr.worlds) + ".")
    lines.append("\nAccessibility:")
    for w in fr.worlds:
        a = fr.acc(w)
        wn = WNAMES.get(w,w)
        if a:
            lines.append(f"  {wn} → {', '.join(WNAMES.get(x,x) for x in sorted(a))}")
        else:
            lines.append(f"  {wn} → (none)")
    lines.append("\nValuations:")
    for w in fr.worlds:
        wn=WNAMES.get(w,w)
        ts=[p for p,v in model.val[w].items() if v]
        fs=[p for p,v in model.val[w].items() if not v]
        if deontic:
            parts=[]
            if ts: parts.append(", ".join(DSCEN.get(p,p) for p in ts)+" practiced")
            if fs: parts.append(", ".join(DSCEN.get(p,p) for p in fs)+" not practiced")
            lines.append(f"  {wn}: {'; '.join(parts)}")
        else:
            parts=[]
            if ts: parts.append(", ".join(ts)+"=True")
            if fs: parts.append(", ".join(fs)+"=False")
            lines.append(f"  {wn}: {'; '.join(parts)}")
    wn=WNAMES.get(world,world)
    fs=formula.s(deontic)
    lines.append(f"\nEvaluating at {wn}, is this true or false?\n  {fs}")
    if deontic:
        lines.append("\nOB='obligatory', PE='permitted', FO='forbidden'.")
    else:
        lines.append("\n□='necessarily' (ALL accessible), ◇='possibly' (SOME accessible).")
    lines.append("Answer exactly 'True' or 'False', then explain.")
    return "\n".join(lines)


PROP_NL = {
    "p": "the door is locked", "q": "the window is open",
    "r": "the alarm is active", "s": "the safe contains documents",
    "t": "the lights are on",
}

def _formula_nl(f, ctx="epistemic"):
    """Recursively translate formula to NL question."""
    if f.t==FT.ATOM:
        return PROP_NL.get(f.atom, DSCEN.get(f.atom, f.atom))
    if f.t==FT.NOT:
        return f"it is NOT the case that {_formula_nl(f.sub,ctx)}"
    if f.t in (FT.BOX,FT.OB):
        inner=_formula_nl(f.sub,ctx)
        if ctx=="epistemic": return f"Alice necessarily knows that {inner}"
        return f"it is obligatory that {inner}"
    if f.t in (FT.DIA,FT.PE):
        inner=_formula_nl(f.sub,ctx)
        if ctx=="epistemic": return f"Alice considers it possible that {inner}"
        return f"it is permitted that {inner}"
    if f.t==FT.FO:
        return f"it is forbidden that {_formula_nl(f.sub,ctx)}"
    if f.t==FT.IMP:
        return f"IF {_formula_nl(f.l,ctx)} THEN {_formula_nl(f.r,ctx)}"
    if f.t==FT.AND:
        return f"BOTH ({_formula_nl(f.l,ctx)}) AND ({_formula_nl(f.r,ctx)})"
    if f.t==FT.OR:
        return f"EITHER ({_formula_nl(f.l,ctx)}) OR ({_formula_nl(f.r,ctx)})"
    return f.s()

ROOMS = ["Room A","Room B","Room C","Room D","Room E","Room F","Room G"]
DISTS = ["District Alpha","District Beta","District Gamma","District Delta",
         "District Epsilon","District Zeta","District Eta"]

def nl_desc(model, formula, world, deontic=False):
    """NATURAL LANGUAGE track: narrative scenario (W1)."""
    fr=model.frame; lines=[]
    if deontic:
        nm={w:DISTS[i] for i,w in enumerate(fr.worlds)}
        lines.append("Consider a legal system with several jurisdictions.")
        lines.append("Each jurisdiction's regulations constrain what is "
                     "obligatory/permitted in jurisdictions it influences.\n")
        for w in fr.worlds:
            a=fr.acc(w); others=[nm[x] for x in sorted(a) if x!=w]
            if w in a and others:
                lines.append(f"{nm[w]} regulates itself and influences {', '.join(others)}.")
            elif w in a:
                lines.append(f"{nm[w]} only regulates itself.")
            elif others:
                lines.append(f"{nm[w]} influences {', '.join(others)} (not itself).")
            else:
                lines.append(f"{nm[w]} has no regulatory influence.")
        lines.append("\nCurrent practices:")
        for w in fr.worlds:
            ts=[DSCEN.get(p,p) for p,v in model.val[w].items() if v]
            fs=[DSCEN.get(p,p) for p,v in model.val[w].items() if not v]
            parts=[]
            if ts: parts.append(", ".join(ts)+" is practiced")
            if fs: parts.append(", ".join(fs)+" is not practiced")
            lines.append(f"  {nm[w]}: {'; '.join(parts)}")
        lines.append(f"\nEvaluating from {nm[world]}:")
        lines.append(f"Question: Is it true that {_formula_nl(formula,'legal')}?")
        lines.append("'Obligatory' = true in ALL influenced jurisdictions. "
                     "'Permitted' = true in SOME.")
    else:
        nm={w:ROOMS[i] for i,w in enumerate(fr.worlds)}
        lines.append("Alice is exploring a building. Some rooms have one-way "
                     "observation windows into other rooms.\n")
        for w in fr.worlds:
            a=fr.acc(w); others=[nm[x] for x in sorted(a) if x!=w]
            if w in a and others:
                lines.append(f"From {nm[w]}, Alice can see {nm[w]} itself and {', '.join(others)}.")
            elif w in a:
                lines.append(f"From {nm[w]}, Alice can only see {nm[w]} itself (mirror).")
            elif others:
                lines.append(f"From {nm[w]}, Alice can see into {', '.join(others)}.")
            else:
                lines.append(f"From {nm[w]}, Alice cannot see any room.")
        lines.append("\nFacts in each room:")
        for w in fr.worlds:
            facts=[]
            for p in model.props:
                v=model.val[w].get(p,False)
                pnl=PROP_NL.get(p,p)
                facts.append(pnl if v else f"it is NOT the case that {pnl}")
            lines.append(f"  {nm[w]}: {'; '.join(facts)}.")
        lines.append(f"\nAlice is in {nm[world]}.")
        lines.append(f"Question: Is it true that {_formula_nl(formula,'epistemic')}?")
        lines.append("'Necessarily'=true in ALL observable rooms. "
                     "'Possibly'=true in SOME.")
    lines.append("\nAnswer exactly 'True' or 'False' and explain.")
    return "\n".join(lines)

print("✅ Dual-track generators loaded (W1)")



# ── Cell 7: BenchmarkProblem + Generation Pipeline ───────────────────────────

@dataclass
class BP:
    """Benchmark Problem."""
    id: str; system: str; tier: int; task_type: str
    presentation: str = "formal"       # "formal" or "nl"
    num_worlds: int = 0
    worlds: List[str] = field(default_factory=list)
    relations: List[Tuple[str,str]] = field(default_factory=list)
    propositions: List[str] = field(default_factory=list)
    valuation: Dict[str,Dict[str,bool]] = field(default_factory=dict)
    formula_str: str = ""
    formula_symbolic: str = ""
    evaluation_world: str = ""
    ground_truth: bool = False
    related_formulas: Optional[List[Dict]] = None
    modal_depth: int = 0
    formula_complexity: int = 0
    deontic: bool = False
    nl_description: str = ""
    axiom_tag: Optional[str] = None
    paradox_name: Optional[str] = None
    frame_density: float = 0.0
    always_valid: bool = False

_counter = 0
def _mid(sys,tier,tt):
    global _counter; _counter+=1
    return f"MB-{sys.value}-T{tier}-{tt[:2].upper()}-{_counter:04d}"

def generate_benchmark():
    """Generate full ModalBench benchmark with dual tracks."""
    problems = []
    systems_al = [Sys.K, Sys.T, Sys.S4, Sys.S5]
    systems_de = [Sys.D]

    print("=" * 60)
    print("GENERATING MODALBENCH v2")
    print("=" * 60)

    for sys_list, deontic in [(systems_al, False), (systems_de, True)]:
        for sys in sys_list:
            for tier in [1, 2, 3]:
                target = C.PROBLEMS_PER_CELL
                count_t, count_f = 0, 0
                half = target // 2
                attempts = 0

                while (count_t + count_f) < target and attempts < target * 20:
                    attempts += 1
                    nw = random.randint(*C.WORLDS_RANGE)
                    np_ = random.randint(*C.PROPS_RANGE)

                    frame = FG.make(sys, nw)
                    if not FG.validate(frame): continue
                    model = make_model(frame, np_, deontic)

                    pool = get_pool(model.props, sys, tier, deontic)
                    if not pool: continue
                    item = random.choice(pool)
                    formula = item["f"]

                    ew = random.choice(frame.worlds)
                    try:
                        gt = formula.ev(model, ew)
                    except: continue

                    # Label balance
                    if gt and count_t >= half: continue
                    if not gt and count_f >= half: continue
                    if gt: count_t += 1
                    else: count_f += 1

                    # Build both tracks
                    formal_text = formal_desc(model, formula, ew, deontic)
                    nl_text = nl_desc(model, formula, ew, deontic)

                    base = dict(
                        system=sys.value, tier=tier, task_type="qa",
                        num_worlds=nw, worlds=frame.worlds,
                        relations=frame.rels, propositions=model.props,
                        valuation=model.val, formula_str=formula.s(deontic),
                        formula_symbolic=formula.s(deontic),
                        evaluation_world=ew, ground_truth=gt,
                        modal_depth=formula.depth(),
                        formula_complexity=formula.complexity(),
                        deontic=deontic, axiom_tag=item["tag"],
                        paradox_name=item["para"],
                        frame_density=frame.density(),
                    )

                    # Formal track problem
                    problems.append(BP(id=_mid(sys,tier,"qa"),
                                       presentation="formal",
                                       nl_description=formal_text, **base))
                    # NL track problem (same ground truth)
                    problems.append(BP(id=_mid(sys,tier,"qa"),
                                       presentation="nl",
                                       nl_description=nl_text, **base))

                print(f"  {sys.value} T{tier} {'D' if deontic else 'A'}: "
                      f"{count_t+count_f} problems (T:{count_t} F:{count_f})")

    print(f"\n✅ Generated {len(problems)} total problems "
          f"({len(problems)//2} × 2 tracks)")

    # Statistics
    df = pd.DataFrame([{"sys":p.system,"tier":p.tier,"pres":p.presentation,
                         "deontic":p.deontic,"gt":p.ground_truth,
                         "nw":p.num_worlds,"depth":p.modal_depth}
                        for p in problems])
    print(f"\nBy system×tier: {df.groupby(['sys','tier']).size().to_dict()}")
    print(f"Label balance: {df['gt'].value_counts(normalize=True).to_dict()}")
    print(f"Track split: {df['pres'].value_counts().to_dict()}")
    print(f"Worlds range: {df['nw'].min()}-{df['nw'].max()}, mean={df['nw'].mean():.1f}")

    return problems

# Module-level demo execution moved to if __name__ == '__main__' block at bottom



# ── Cell 8: Prompting Strategies (W2) ────────────────────────────────────────

STRATEGIES = {
    "zero_shot": "You are a logic expert. Answer exactly 'True' or 'False'.",
    "cot": (
        "You are a modal logic expert. Think step by step. Show reasoning, "
        "then answer exactly 'True' or 'False'."),
    "kripke_few_shot": (
        "You are a modal logic expert. Follow this procedure:\n"
        "STEP 1: List accessible worlds from evaluation world.\n"
        "STEP 2: For □P check ALL accessible. For ◇P check ANY.\n"
        "STEP 3: For nested operators, work inside-out.\n"
        "STEP 4: Build truth table if needed.\n"
        "STEP 5: Answer exactly 'True' or 'False'.\n\n"
        "Example:\nWorlds: w0,w1,w2. Rels: w0→w1, w0→w2.\n"
        "Vals: w0:p=T | w1:p=F | w2:p=T\n"
        "Q: At w0, □p? → w0 sees w1,w2. p@w1=F. So □p=False.\nAnswer: False"),
    "world_enum": (
        "You are a modal logic expert. ALWAYS:\n"
        "1. ENUMERATE all accessible worlds.\n"
        "2. TABULATE truth value of sub-formula in each.\n"
        "3. AGGREGATE: □=ALL true, ◇=ANY true.\n"
        "4. For nesting, recurse inside-out.\n"
        "Show your enumeration table. Answer exactly 'True' or 'False'."),
}

print(f"✅ {len(STRATEGIES)} prompting strategies loaded (W2)")



# ── Cell 9: Free Model Evaluator ─────────────────────────────────────────────

FREE_MODELS = {
    "gemini-2.0-flash": dict(prov="google", mid="gemini-2.0-flash",
        reason=False, key="GOOGLE_API_KEY", note="1500 req/day free"),
    "gemini-2.5-flash": dict(prov="google", mid="gemini-2.5-flash",
        reason=True, key="GOOGLE_API_KEY", note="500 req/day free"),
    "llama-70b-groq": dict(prov="groq", mid="llama-3.3-70b-versatile",
        reason=False, key="GROQ_API_KEY", note="1000 req/day free"),
    "qwq-32b-groq": dict(prov="groq", mid="qwen-qwq-32b",
        reason=True, key="GROQ_API_KEY", note="1000 req/day free"),
    "qwen3-235b-think-cerebras": dict(
    prov="cerebras", mid="qwen-3-235b-a22b-thinking-2507",
    reason=True, key="CEREBRAS_API_KEY",
    note="235B thinking variant, preview"
),
    "llama-70b-cerebras": dict(prov="cerebras", mid="llama-3.3-70b",
        reason=False, key="CEREBRAS_API_KEY", note="1M tok/day free"),
    "qwen3-235b-cerebras": dict(prov="cerebras", mid="qwen-3-235b",
        reason=True, key="CEREBRAS_API_KEY", note="frontier free"),
    "deepseek-r1-or": dict(prov="openrouter", mid="deepseek/deepseek-r1:free",
        reason=True, key="OPENROUTER_API_KEY", note="community free"),
}
# Patch the dictionary with the new Instruct Preview ID
FREE_MODELS["qwen3-235b-cerebras"] = dict(
    prov="cerebras",
    mid="qwen-3-235b-a22b-instruct-2507",
    reason=False,
    key="CEREBRAS_API_KEY",
    note="frontier free instruct"
)
print("✅ Qwen 3 235B model ID updated successfully.")
PROV_URL = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

def _detect_models():
    avail = {}
    for name, info in FREE_MODELS.items():
        if os.environ.get(info["key"], ""):
            avail[name] = info
    return avail


async def async_query_model(name, prompt, sys_prompt="", max_tok=1024, temp=0.0):
    """Asynchronously query a free model. Returns text or None."""
    info = FREE_MODELS.get(name)
    if not info: return None
    key = os.environ.get(info["key"], "")
    if not key: return None

    for attempt in range(C.MAX_RETRIES):
        try:
            if info["prov"] == "google":
                # --- NEW GOOGLE GENAI SDK ---
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=key)

                r = await client.aio.models.generate_content(
                    model=info["mid"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt if sys_prompt else None,
                        temperature=temp,
                        max_output_tokens=max_tok,
                    )
                )
                return r.text

            else:
                # --- OPENAI / GROQ / CEREBRAS / OPENROUTER ---
                from openai import AsyncOpenAI
                c = AsyncOpenAI(api_key=key, base_url=PROV_URL[info["prov"]])
                msgs = []
                if sys_prompt: msgs.append({"role":"system","content":sys_prompt})
                msgs.append({"role":"user","content":prompt})
                kw = dict(model=info["mid"], messages=msgs,
                          max_tokens=max_tok, temperature=temp)
                if info["prov"]=="openrouter":
                    kw["extra_headers"]={"HTTP-Referer":"https://modalbench.org",
                                         "X-Title":"ModalBench"}

                r = await c.chat.completions.create(**kw)
                return r.choices[0].message.content

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                import random
                w = (2 ** attempt) + random.uniform(0, 1)
                print(f"    ⏳ Rate limit hit for {name}. Backing off for {w:.2f}s...")
                await asyncio.sleep(w)
            else:
                w = C.RETRY_DELAY
                print(f"    ⚠️ {name} err: {e}. Retry {w}s")
                await asyncio.sleep(w)
    return None

def query_model(name, prompt, sys_prompt="", max_tok=1024, temp=0.0):
    """Query a free model. Returns text or None."""
    info = FREE_MODELS.get(name)
    if not info: return None
    key = os.environ.get(info["key"], "")
    if not key: return None

    for attempt in range(C.MAX_RETRIES):
        try:
            if info["prov"] == "google":
                import google.generativeai as genai
                genai.configure(api_key=key)
                m = genai.GenerativeModel(info["mid"],
                    system_instruction=sys_prompt or None)
                r = m.generate_content(prompt,
                    generation_config={"temperature":temp,"max_output_tokens":max_tok})
                return r.text
            else:
                from openai import OpenAI
                c = OpenAI(api_key=key, base_url=PROV_URL[info["prov"]])
                msgs = []
                if sys_prompt: msgs.append({"role":"system","content":sys_prompt})
                msgs.append({"role":"user","content":prompt})
                kw = dict(model=info["mid"], messages=msgs,
                          max_tokens=max_tok, temperature=temp)
                if info["prov"]=="openrouter":
                    kw["extra_headers"]={"HTTP-Referer":"https://modalbench.org",
                                         "X-Title":"ModalBench"}
                r = c.chat.completions.create(**kw)
                return r.choices[0].message.content
        except Exception as e:
            w = C.RETRY_DELAY * (attempt+1)
            print(f"    ⚠️ {name} err: {e}. Retry {w}s")
            time.sleep(w)
    return None

def parse_answer(resp):
    if resp is None: return None
    r = resp.lower().strip()
    for pat in [r"answer:\s*(true|false)", r"\*\*(true|false)\*\*",
                r"final answer:\s*(true|false)"]:
        m = re.search(pat, r)
        if m: return m.group(1)=="true"
    for line in reversed(r.split('\n')):
        l=line.strip().rstrip('.')
        if l in ('true',): return True
        if l in ('false',): return False
    tc=r.count('true'); fc=r.count('false')
    if tc>fc: return True
    if fc>tc: return False
    return None

avail_models = _detect_models()
print(f"✅ Free models available: {list(avail_models.keys()) or '(none — set API keys)'}")



# ── Cell 10: Google Drive Checkpoint ─────────────────────────────────────────

class GDrive:
    def __init__(self, base="/content/drive/MyDrive/ICLR2026/ModalBench"):
        self.base = base
        try:
            # from google.colab import drive  # Colab only
            drive.mount('/content/drive', force_remount=False)
        except:
            self.base = "modalbench_local_ckpt"
        for d in ["benchmarks","results","figures"]:
            os.makedirs(f"{self.base}/{d}", exist_ok=True)
        print(f"✅ GDrive ready: {self.base}")

    def save_bench(self, problems, tag="v2"):
        path = f"{self.base}/benchmarks/bench_{tag}.json"
        data = []
        for p in problems:
            d = asdict(p)
            d["relations"] = [list(r) for r in d.get("relations",[])]
            data.append(d)
        with open(path,'w') as f: json.dump(data,f)
        print(f"✅ Saved {len(data)} problems → {path}")

    def load_bench(self, tag="v2"):
        path = f"{self.base}/benchmarks/bench_{tag}.json"
        with open(path) as f: data=json.load(f)
        print(f"✅ Loaded {len(data)} problems ← {path}")
        return data

    def save_results(self, df, tag="run1"):
        path = f"{self.base}/results/res_{tag}.csv"
        df.to_csv(path, index=False)
        print(f"✅ Saved {len(df)} results → {path}")

    def load_results(self, tag="run1"):
        path = f"{self.base}/results/res_{tag}.csv"
        return pd.read_csv(path)

    def save_fig(self, fig, name):
        for ext in ['png','pdf']:
            fig.savefig(f"{self.base}/figures/{name}.{ext}",
                        dpi=300, bbox_inches='tight')
# 1. Mount Drive first
gd = GDrive()
drive_path = f"{gd.base}/partial_results.jsonl"


# ── Cell 11: Evaluation Orchestrator ─────────────────────────────────────────

def stratified_sample(problems, n=600):
    """Stratified sample preserving system×tier×track balance."""
    bk = defaultdict(list)
    for p in problems:
        bk[(p.system, p.tier, p.presentation)].append(p)
    per = max(1, n // len(bk))
    out = []
    for k, bucket in sorted(bk.items()):
        random.shuffle(bucket)
        out.extend(bucket[:per])
    random.shuffle(out)
    print(f"✅ Sampled {len(out)} from {len(problems)} ({len(bk)} strata, {per}/stratum)")
    return out

def run_evaluation(problems, models=None, strategies=None,
                   max_problems=None, delay=1.0):
    """Full evaluation: models × strategies × problems."""
    if models is None:
        models = list(avail_models.keys())[:4]
    if strategies is None:
        strategies = ["cot"]
    if max_problems:
        problems = stratified_sample(problems, max_problems)

    total = len(problems)*len(models)*len(strategies)
    print(f"\n📊 Evaluation: {len(problems)} × {len(models)} models × "
          f"{len(strategies)} strats = {total} calls")

    rows = []; done = 0
    for mn in models:
        for sn in strategies:
            sp = STRATEGIES[sn]
            ok = 0
            print(f"\n  🔄 {mn} / {sn}...")
            for i, p in enumerate(problems):
                resp = query_model(mn, p.nl_description, sys_prompt=sp)
                pred = parse_answer(resp)
                correct = pred == p.ground_truth if pred is not None else None
                if correct: ok += 1
                rows.append(dict(
                    problem_id=p.id, model=mn, prompt_strategy=sn,
                    presentation=p.presentation, system=p.system,
                    tier=p.tier, deontic=p.deontic,
                    modal_depth=p.modal_depth, num_worlds=p.num_worlds,
                    formula=p.formula_str, axiom_tag=p.axiom_tag,
                    paradox_name=p.paradox_name,
                    ground_truth=p.ground_truth, predicted=pred,
                    correct=correct,
                    response=(resp[:500] if resp else None),
                ))
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{total}")
                time.sleep(delay)
            print(f"    ✅ {mn}/{sn}: {ok}/{len(problems)} = "
                  f"{ok/max(len(problems),1):.1%}")

    df = pd.DataFrame(rows)
    print(f"\n✅ Done: {len(df)} results")
    return df



async def process_single_problem(p, mn, sn, sem, pbar, backup_file):
    """Worker function for a single problem evaluation."""
    async with sem:
        sp = STRATEGIES[sn]
        resp = await async_query_model(mn, p.nl_description, sys_prompt=sp)
        pred = parse_answer(resp)
        correct = pred == p.ground_truth if pred is not None else None

        result = dict(
            problem_id=p.id, model=mn, prompt_strategy=sn,
            presentation=p.presentation, system=p.system,
            tier=p.tier, deontic=p.deontic,
            modal_depth=p.modal_depth, num_worlds=p.num_worlds,
            formula=p.formula_str, axiom_tag=p.axiom_tag,
            paradox_name=p.paradox_name,
            ground_truth=p.ground_truth, predicted=pred,
            correct=correct,
            response=(resp[:500] if resp else None),
        )

        # Incremental save to prevent data loss
        if backup_file:
            with open(backup_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

        pbar.update(1)
        return result




# --- 3. Main Async Orchestrator (Updated for precise resume) ---
async def async_run_evaluation(problems, models=None, strategies=None,
                               max_problems=None, concurrency=15,
                               backup_file=f"{C.OUT}/partial_results.jsonl"):
    """Concurrent evaluation with precise task filtering."""
    if models is None:
        models = list(avail_models.keys())[:4]
    if strategies is None:
        strategies = ["cot"]
    if max_problems:
        problems = stratified_sample(problems, max_problems)

    completed_tasks = set()
    existing_results = []

    # 1. Load saved results
    if backup_file and os.path.exists(backup_file):
        with open(backup_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        existing_results.append(res)
                        task_fingerprint = (res["problem_id"], res["model"], res["prompt_strategy"])
                        completed_tasks.add(task_fingerprint)
                    except json.JSONDecodeError:
                        continue

        if completed_tasks:
            print(f"🔄 RESUMING: Found {len(completed_tasks)} total tasks in backup.")

    total_target = len(problems) * len(models) * len(strategies)

    # 2. Precisely calculate ONLY the missing tasks for our CURRENT lineup
    missing_tasks = []
    for mn in models:
        for sn in strategies:
            for p in problems:
                if (p.id, mn, sn) not in completed_tasks:
                    missing_tasks.append((p, mn, sn))

    tasks_to_run = len(missing_tasks)

    print(f"\n📊 Async Eval: {total_target} target tasks. {tasks_to_run} actually remaining to run.")
    print(f"⚡ Concurrency limit: {concurrency} simultaneous requests")

    # Filter out any old "ghost" data from models we removed
    filtered_existing = [r for r in existing_results if r["model"] in models and r["prompt_strategy"] in strategies]

    if tasks_to_run == 0:
        print("✅ All target tasks already completed! Returning cached results.")
        return pd.DataFrame(filtered_existing)

    # 3. Execute only the precisely missing tasks
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    with tqdm(total=tasks_to_run, desc="Evaluating API Calls") as pbar:
        for p, mn, sn in missing_tasks:
            task = asyncio.create_task(
                process_single_problem(p, mn, sn, sem, pbar, backup_file)
            )
            tasks.append(task)

        new_results = await asyncio.gather(*tasks)

    all_results = filtered_existing + list(new_results)
    df = pd.DataFrame(all_results)

    print(f"\n✅ Done: {len(df)} total polished results collected")
    return df




# ── Cell 12: Synthetic Results (for paper dev without API keys) ──────────────

def synthetic_results(problems):
    """Generate realistic synthetic results for paper figures."""
    MODEL_P = {
        "gemini-2.0-flash":    dict(b=0.65, td=0.13, kb=0.07, r=False),
        "gemini-2.5-flash":    dict(b=0.72, td=0.11, kb=0.06, r=True),
        "llama-70b-groq":      dict(b=0.60, td=0.14, kb=0.06, r=False),
        "qwq-32b-groq":        dict(b=0.74, td=0.10, kb=0.05, r=True),
        "llama-70b-cerebras":  dict(b=0.61, td=0.14, kb=0.06, r=False),
        "qwen3-235b-cerebras": dict(b=0.76, td=0.09, kb=0.06, r=True),
    }
    SYS_MOD = {"K":0,"T":-0.02,"S4":-0.05,"S5":-0.08,"D":-0.04}
    STRATS = ["zero_shot","cot","kripke_few_shot","world_enum"]
    STRAT_BOOST = {"zero_shot":0, "cot":0.04, "kripke_few_shot":0.07, "world_enum":0.09}

    rows = []
    for mn, mp in MODEL_P.items():
        for sn in STRATS:
            for p in problems:
                pr = max(0.15, min(0.95,
                    mp["b"]
                    - mp["td"]*(p.tier-1)
                    + SYS_MOD.get(p.system,0)
                    + STRAT_BOOST[sn]
                    - p.modal_depth*0.03
                    - max(0,(p.num_worlds-3)*0.02)
                    - (0.03 if p.deontic else 0)
                    - (0.05 if p.presentation=="nl" else 0)
                    + np.random.normal(0, 0.05)
                ))
                correct = random.random() < pr
                pred = p.ground_truth if correct else (not p.ground_truth)
                rows.append(dict(
                    problem_id=p.id, model=mn, prompt_strategy=sn,
                    presentation=p.presentation, system=p.system,
                    tier=p.tier, deontic=p.deontic,
                    modal_depth=p.modal_depth, num_worlds=p.num_worlds,
                    formula=p.formula_str, axiom_tag=p.axiom_tag,
                    paradox_name=p.paradox_name,
                    ground_truth=p.ground_truth, predicted=pred,
                    correct=correct, response=None,
                ))
    df = pd.DataFrame(rows)
    print(f"✅ Synthetic results: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['prompt_strategy'].nunique()} strategies")
    return df

# results_df = synthetic_results(problems)  # moved to __main__



# ── Cell 13: Diagnostic Analysis Suite (W4) ──────────────────────────────────

class Diag:
    """Full diagnostic analysis suite addressing W4."""

    def __init__(self, df, out=f"{C.OUT}/{C.FIG}"):
        # Robust dtype coercion: real CSVs load 'correct' as object (string)
        # which crashes seaborn heatmaps. Convert to float (NaN-safe).
        df = df.copy()
        if df['correct'].dtype == object:
            df['correct'] = df['correct'].map(
                {'True': True, 'False': False, True: True, False: False}
            )
        df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
        if 'predicted' in df.columns and df['predicted'].dtype == object:
            df['predicted'] = df['predicted'].map(
                {'True': True, 'False': False, True: True, False: False}
            )
        self.df = df; self.out = out
        os.makedirs(out, exist_ok=True)
        plt.rcParams.update({'font.size':11,'figure.dpi':150,
                             'savefig.dpi':300,'savefig.bbox':'tight'})
        sns.set_theme(style="whitegrid")

    def _save(self, fig, name):
        for ext in ['png','pdf']:
            fig.savefig(f"{self.out}/{name}.{ext}")
        plt.close(fig)
        print(f"  ✅ {name}")

    # --- Tables ---

    def table_main(self):
        """Table 1: Overall accuracy by model × system."""
        d=self.df[self.df.prompt_strategy=="cot"]
        t=d.groupby(["model","system"])["correct"].mean().unstack()
        t["Overall"]=d.groupby("model")["correct"].mean()
        print("\n📊 TABLE 1: Main Results (CoT strategy)")
        print(t.round(3).to_string())
        t.round(3).to_csv(f"{self.out}/table1_main.csv")
        return t

    def table_strategies(self):
        """Table 2: Strategy comparison (W2)."""
        t=self.df.groupby(["model","prompt_strategy"])["correct"].mean().unstack()
        print("\n📊 TABLE 2: Prompting Strategy Comparison (W2)")
        print(t.round(3).to_string())
        t.round(3).to_csv(f"{self.out}/table2_strategies.csv")
        return t

    def table_formal_vs_nl(self):
        """Table 3: Formal vs NL gap (W1)."""
        t=self.df.groupby(["model","presentation"])["correct"].mean().unstack()
        if "formal" in t.columns and "nl" in t.columns:
            t["gap"]=t["formal"]-t["nl"]
        print("\n📊 TABLE 3: Formal vs NL Gap (W1)")
        print(t.round(3).to_string())
        t.round(3).to_csv(f"{self.out}/table3_formal_nl.csv")
        return t

    def table_axiom(self):
        """Table 4: Per-axiom accuracy (W4 headline)."""
        d=self.df[self.df.axiom_tag.notna() & (self.df.prompt_strategy=="cot")]
        if d.empty: print("⚠️ No axiom data"); return
        t=d.groupby(["model","axiom_tag"])["correct"].mean().unstack()
        print("\n📊 TABLE 4: Per-Axiom Accuracy (W4)")
        print(t.round(3).to_string())
        t.round(3).to_csv(f"{self.out}/table4_axiom.csv")
        return t

    def table_paradox(self):
        """Table 5: Deontic paradox results (W3)."""
        d=self.df[self.df.paradox_name.notna() & (self.df.prompt_strategy=="cot")]
        if d.empty: print("⚠️ No paradox data"); return
        t=d.groupby(["model","paradox_name"])["correct"].mean().unstack()
        print("\n📊 TABLE 5: Deontic Paradox Accuracy (W3)")
        print(t.round(3).to_string())
        t.round(3).to_csv(f"{self.out}/table5_paradox.csv")
        return t

    # --- Figures ---

    def fig1_heatmap(self):
        """Fig1: Accuracy heatmap model×system×tier."""
        d=self.df[(self.df.prompt_strategy=="cot")&(self.df.presentation=="formal")]
        models=sorted(d.model.unique())
        n=len(models); nc=min(4,n); nr=math.ceil(n/nc)
        fig,axes=plt.subplots(nr,nc,figsize=(5*nc,4*nr))
        axes=np.array(axes).flatten()
        for i,mn in enumerate(models):
            md=d[d.model==mn]
            pv=md.groupby(["system","tier"])["correct"].mean().unstack()
            sns.heatmap(pv,annot=True,fmt='.2f',cmap='RdYlGn',
                        vmin=0.3,vmax=0.9,ax=axes[i],cbar_kws={'shrink':0.8})
            axes[i].set_title(mn,fontsize=10,fontweight='bold')
        for j in range(n,len(axes)): axes[j].set_visible(False)
        fig.suptitle("Accuracy heatmap (system × tier)",fontweight='bold',y=1.02)
        fig.tight_layout()
        self._save(fig,"fig1_heatmap")

    def fig2_strategy_comparison(self):
        """Fig2: Strategy comparison bar chart (W2)."""
        fig,ax=plt.subplots(figsize=(14,6))
        t=self.df.groupby(["model","prompt_strategy"])["correct"].mean().unstack()
        t.plot(kind='bar',ax=ax,width=0.8,edgecolor='black',linewidth=0.5)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0.2,0.95)
        ax.set_title("Prompting strategy comparison (W2)",fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
        ax.legend(title="Strategy"); ax.axhline(0.5,color='red',ls='--',alpha=0.3)
        fig.tight_layout()
        self._save(fig,"fig2_strategies")

    def fig3_formal_vs_nl(self):
        """Fig3: Formal vs NL track comparison (W1)."""
        d=self.df[self.df.prompt_strategy=="cot"]
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
        # Panel a: by model
        models=sorted(d.model.unique()); x=np.arange(len(models)); w=0.35
        fa=[d[(d.model==m)&(d.presentation=="formal")]["correct"].mean() for m in models]
        na=[d[(d.model==m)&(d.presentation=="nl")]["correct"].mean() for m in models]
        ax1.bar(x-w/2,fa,w,label='Formal',color='#3498db',edgecolor='black')
        ax1.bar(x+w/2,na,w,label='Natural Language',color='#e74c3c',edgecolor='black')
        for i in range(len(models)):
            gap=(fa[i]-na[i])*100
            ax1.annotate(f'{gap:+.1f}pp',xy=(x[i],max(fa[i],na[i])+0.01),
                         ha='center',fontsize=8,color='purple')
        ax1.set_xticks(x); ax1.set_xticklabels(models,rotation=45,ha='right')
        ax1.set_ylabel("Accuracy"); ax1.set_title("(a) By model"); ax1.legend()
        # Panel b: by system
        systems=["K","T","S4","S5","D"]; x2=np.arange(len(systems))
        fa2=[d[(d.system==s)&(d.presentation=="formal")]["correct"].mean() for s in systems]
        na2=[d[(d.system==s)&(d.presentation=="nl")]["correct"].mean() for s in systems]
        ax2.bar(x2-w/2,fa2,w,label='Formal',color='#3498db',edgecolor='black')
        ax2.bar(x2+w/2,na2,w,label='NL',color='#e74c3c',edgecolor='black')
        ax2.set_xticks(x2); ax2.set_xticklabels(systems)
        ax2.set_ylabel("Accuracy"); ax2.set_title("(b) By system"); ax2.legend()
        fig.suptitle("Formal vs natural language track (W1)",fontweight='bold')
        fig.tight_layout()
        self._save(fig,"fig3_formal_vs_nl")

    def fig4_axiom_accuracy(self):
        """Fig4: Per-axiom accuracy heatmap (W4 headline)."""
        d=self.df[self.df.axiom_tag.notna()&(self.df.prompt_strategy=="cot")&
                  (self.df.presentation=="formal")]
        if d.empty: return
        fig,ax=plt.subplots(figsize=(14,6))
        pv=d.groupby(["model","axiom_tag"])["correct"].mean().unstack()
        sns.heatmap(pv,annot=True,fmt='.2f',cmap='RdYlGn',vmin=0.2,vmax=0.9,ax=ax)
        ax.set_title("Per-axiom accuracy (W4)",fontweight='bold')
        fig.tight_layout()
        self._save(fig,"fig4_axiom")

    def fig5_s5_bias(self):
        """Fig5: Implicit S5 bias test (W4)."""
        d=self.df[(self.df.axiom_tag=="5")&(self.df.prompt_strategy=="cot")&
                  (self.df.presentation=="formal")]
        if d.empty: return
        fig,ax=plt.subplots(figsize=(12,6))
        # Compute how often models say True for axiom 5 by system
        data=[]
        for mn in sorted(d.model.unique()):
            for sys in ["K","T","S4","S5"]:
                sub=d[(d.model==mn)&(d.system==sys)]
                if sub.empty: continue
                tr=(sub.predicted==True).mean() if sub.predicted.notna().any() else 0
                gt_tr=sub.ground_truth.mean()
                data.append(dict(model=mn,system=sys,pred_true=tr,gt_true=gt_tr,
                                 bias=tr-gt_tr))
        if not data: return
        bdf=pd.DataFrame(data)
        pv=bdf.pivot(index="model",columns="system",values="bias").fillna(0)
        sns.heatmap(pv,annot=True,fmt='+.2f',cmap='RdBu_r',center=0,ax=ax)
        ax.set_title("Implicit S5 bias: axiom-5 over-application (W4)",fontweight='bold')
        fig.tight_layout()
        self._save(fig,"fig5_s5_bias")

    def fig6_worlds_scaling(self):
        """Fig6: Accuracy vs number of worlds."""
        d=self.df[(self.df.prompt_strategy=="cot")&(self.df.presentation=="formal")]
        fig,ax=plt.subplots(figsize=(10,6))
        for mn in sorted(d.model.unique()):
            md=d[d.model==mn]
            bw=md.groupby("num_worlds")["correct"].mean()
            ax.plot(bw.index,bw.values,'o-',label=mn,lw=2,ms=6)
        ax.set_xlabel("Number of worlds"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs frame size",fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05,1),loc='upper left')
        ax.axhline(0.5,color='red',ls='--',alpha=0.3)
        fig.tight_layout()
        self._save(fig,"fig6_worlds")

    def fig7_depth_degradation(self):
        """Fig7: Accuracy vs modal nesting depth."""
        d=self.df[(self.df.prompt_strategy=="cot")&(self.df.presentation=="formal")]
        fig,ax=plt.subplots(figsize=(10,6))
        for mn in sorted(d.model.unique()):
            md=d[d.model==mn]
            bd=md.groupby("modal_depth")["correct"].mean()
            ax.plot(bd.index,bd.values,'o-',label=mn,lw=2,ms=6)
        ax.set_xlabel("Modal nesting depth"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs nesting depth",fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05,1),loc='upper left')
        fig.tight_layout()
        self._save(fig,"fig7_depth")

    def fig8_box_vs_diamond(self):
        """Fig8: □ vs ◇ operator accuracy."""
        d=self.df[(self.df.prompt_strategy=="cot")&(self.df.presentation=="formal")]
        def classify(fs):
            if not isinstance(fs,str): return "other"
            has_box = "□" in fs or "OB(" in fs
            has_dia = "◇" in fs or "PE(" in fs
            if has_box and not has_dia: return "□ only"
            if has_dia and not has_box: return "◇ only"
            if has_box and has_dia: return "mixed"
            return "other"
        d=d.copy(); d["op_class"]=d["formula"].apply(classify)
        fig,ax=plt.subplots(figsize=(12,6))
        pv=d.groupby(["model","op_class"])["correct"].mean().unstack()
        pv.plot(kind='bar',ax=ax,width=0.8,edgecolor='black')
        ax.set_ylabel("Accuracy"); ax.set_title("□ vs ◇ accuracy (W4)")
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
        fig.tight_layout()
        self._save(fig,"fig8_box_diamond")

    def fig9_tier_by_model(self):
        """Fig9: Tier difficulty comparison."""
        d=self.df[(self.df.prompt_strategy=="cot")&(self.df.presentation=="formal")]
        fig,ax=plt.subplots(figsize=(14,6))
        pv=d.groupby(["model","tier"])["correct"].mean().unstack()
        pv.columns=[f"Tier {c}" for c in pv.columns]
        pv.plot(kind='bar',ax=ax,width=0.8,edgecolor='black')
        ax.set_ylabel("Accuracy"); ax.set_ylim(0.2,0.95)
        ax.set_title("Accuracy by difficulty tier",fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
        ax.axhline(0.5,color='red',ls='--',alpha=0.3)
        fig.tight_layout()
        self._save(fig,"fig9_tiers")

    def heuristic_check(self):
        """Heuristic resistance validation."""
        d=self.df
        h1 = (d.ground_truth==True).mean()
        box_mask = d.formula.apply(lambda x: "□" in str(x) or "OB(" in str(x))
        h2 = (d.loc[box_mask,"ground_truth"]==True).mean() if box_mask.any() else 0
        print(f"\n📊 HEURISTIC RESISTANCE:")
        print(f"  H1 (always True baseline): {h1:.1%} "
              f"{'✅ balanced' if 0.4<h1<0.6 else '⚠️'}")
        print(f"  H2 (□→True heuristic):     {h2:.1%} "
              f"{'✅ resistant' if 0.4<h2<0.6 else '⚠️'}")

    def run_all(self):
        """Generate everything."""
        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYSIS SUITE (W4)")
        print("="*60)
        self.table_main()
        self.table_strategies()
        self.table_formal_vs_nl()
        self.table_axiom()
        self.table_paradox()
        self.heuristic_check()
        print("\n📈 Generating figures...")
        self.fig1_heatmap()
        self.fig2_strategy_comparison()
        self.fig3_formal_vs_nl()
        self.fig4_axiom_accuracy()
        self.fig5_s5_bias()
        self.fig6_worlds_scaling()
        self.fig7_depth_degradation()
        self.fig8_box_vs_diamond()
        self.fig9_tier_by_model()
        print(f"\n✅ All outputs in {self.out}/")

# Run analysis (moved to __main__ block at bottom of file)
# diag = Diag(results_df)
# diag.run_all()



# ── Cell 14: Statistical Significance ────────────────────────────────────────

def stat_tests(df):
    """Run statistical tests on results DataFrame.

    Robust to parse failures (NaN in `correct`). Uses Welch's t-test for
    formal vs NL (unpaired, different sample sizes), one-way ANOVA for
    strategy and tier effects, and reports Cohen's d effect sizes.
    """
    import numpy as np
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)

    # Ensure numeric correct column
    df = df.copy()
    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')

    # 1. Formal vs NL — Welch t-test on all data (not just CoT)
    print("\n--- Formal vs NL (Welch t-test, all strategies) ---")
    for mn in sorted(df.model.unique()):
        fa = df[(df.model==mn)&(df.presentation=="formal")]["correct"].dropna()
        na = df[(df.model==mn)&(df.presentation=="nl")]["correct"].dropna()
        if len(fa)<10 or len(na)<10:
            print(f"  {mn}: insufficient data")
            continue
        t,p = stats.ttest_ind(fa.values, na.values, equal_var=False)
        pooled_sd = np.sqrt((fa.var()+na.var())/2)
        d = (fa.mean()-na.mean()) / pooled_sd if pooled_sd > 0 else 0
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {mn}: formal={fa.mean():.3f} nl={na.mean():.3f} "
              f"Δ={fa.mean()-na.mean():+.3f} t={t:.2f} p={p:.2e} {sig} d={d:+.2f}")

    # 2. Strategy ANOVA (all presentations combined)
    print("\n--- Strategy effect (one-way ANOVA) ---")
    for mn in sorted(df.model.unique()):
        md = df[df.model==mn]
        groups = [md[md.prompt_strategy==s]["correct"].dropna().values
                  for s in sorted(df.prompt_strategy.unique())]
        groups = [g for g in groups if len(g)>10]
        if len(groups)<2:
            print(f"  {mn}: insufficient data")
            continue
        f_stat,p = stats.f_oneway(*groups)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {mn}: F={f_stat:.2f} p={p:.2e} {sig}")

    # 3. Tier difficulty (CoT only, standardized comparison)
    print("\n--- Tier difficulty (one-way ANOVA, CoT) ---")
    cot = df[df.prompt_strategy=="cot"]
    for mn in sorted(cot.model.unique()):
        md = cot[cot.model==mn]
        groups = [md[md.tier==t]["correct"].dropna().values for t in [1,2,3]]
        if not all(len(g)>10 for g in groups):
            continue
        f_stat,p = stats.f_oneway(*groups)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {mn}: T1={groups[0].mean():.3f} T2={groups[1].mean():.3f} "
              f"T3={groups[2].mean():.3f} F={f_stat:.2f} p={p:.2e} {sig}")

    # 4. System difficulty (CoT only)
    print("\n--- System difficulty (one-way ANOVA, CoT) ---")
    for mn in sorted(cot.model.unique()):
        md = cot[cot.model==mn]
        groups = [md[md.system==s]["correct"].dropna().values
                  for s in ["K","T","S4","S5","D"]]
        groups = [g for g in groups if len(g)>10]
        if len(groups)<2:
            continue
        f_stat,p = stats.f_oneway(*groups)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {mn}: F={f_stat:.2f} p={p:.2e} {sig}")

# stat_tests(results_df)  # moved to __main__



# ── Cell 15: Abstract Statistics ─────────────────────────────────────────────

def abstract_stats(df, problems):
    print("\n" + "="*60)
    print("KEY STATISTICS FOR PAPER ABSTRACT")
    print("="*60)

    cot = df[df.prompt_strategy=="cot"]
    n_prob = len(set(p.id for p in problems))

    print(f"\n📊 Benchmark: {n_prob} unique problems × 2 tracks = {len(problems)} total")
    print(f"   5 modal systems (K,T,S4,S5,D), 3 tiers, 4 strategies")
    print(f"   Models: {df.model.nunique()}")

    best = cot.groupby("model")["correct"].mean()
    print(f"\n📊 Best model: {best.idxmax()} ({best.max():.1%})")
    print(f"   Worst: {best.idxmin()} ({best.min():.1%})")

    t1 = cot[cot.tier==1]["correct"].mean()
    t3 = cot[cot.tier==3]["correct"].mean()
    print(f"\n📊 Tier 1→3 drop: {t1:.1%}→{t3:.1%} ({(t1-t3)*100:.1f}pp)")

    fa = cot[cot.presentation=="formal"]["correct"].mean()
    na = cot[cot.presentation=="nl"]["correct"].mean()
    print(f"📊 Formal vs NL gap: {fa:.1%} vs {na:.1%} ({(fa-na)*100:.1f}pp)")

    ws = df.groupby("prompt_strategy")["correct"].mean()
    best_s = ws.idxmax(); worst_s = ws.idxmin()
    print(f"📊 Best strategy: {best_s} ({ws[best_s]:.1%}) "
          f"vs worst: {worst_s} ({ws[worst_s]:.1%})")

# abstract_stats(results_df, problems)  # moved to __main__

# ── Cell 16: Instructions for Real Evaluation ────────────────────────────────

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║                  REAL EVALUATION GUIDE                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. SET API KEYS (free!):                                    ║
║     In Colab → 🔑 icon → Add secrets:                       ║
║       GOOGLE_API_KEY    → aistudio.google.com/apikey         ║
║       GROQ_API_KEY      → console.groq.com                  ║
║       CEREBRAS_API_KEY  → cloud.cerebras.ai                 ║
║       OPENROUTER_API_KEY → openrouter.ai/keys               ║
║                                                              ║
║  2. LOAD KEYS:                                               ║
║     # from google.colab import userdata  # Colab only                        ║
║     import os                                                ║
║     os.environ["GOOGLE_API_KEY"]=userdata.get("GOOGLE_...")  ║
║     os.environ["GROQ_API_KEY"]=userdata.get("GROQ_...")     ║
║                                                              ║
║  3. RE-DETECT MODELS:                                        ║
║     avail_models = _detect_models()                          ║
║                                                              ║
║  4. RUN:                                                     ║
║     results = run_evaluation(                                ║
║       problems,                                              ║
║       models=["gemini-2.0-flash","llama-70b-groq",          ║
║               "qwq-32b-groq","qwen3-235b-cerebras"],        ║
║       strategies=["zero_shot","cot","world_enum"],           ║
║       max_problems=400,                                      ║
║     )                                                        ║
║                                                              ║
║  5. ANALYZE:                                                 ║
║     Diag(results).run_all()                                  ║
║                                                              ║
║  6. SAVE TO GOOGLE DRIVE:                                    ║
║     gd = GDrive()                                            ║
║     gd.save_bench(problems, "final")                         ║
║     gd.save_results(results, "final")                        ║
║                                                              ║
║  7. RESUME AFTER DISCONNECT:                                 ║
║     gd = GDrive()                                            ║
║     old = gd.load_results("final")                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════════
# Demo execution: run `python modalbench_complete.py` to test the pipeline
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(INSTRUCTIONS)
    print("\n🚀 Running demo pipeline (synthetic results, no API calls)...")

    # 1. Generate the benchmark
    problems = generate_benchmark()

    # 2. Use synthetic results for demo (or load real ones from CSV)
    results_df = synthetic_results(problems)

    # 3. Run diagnostics
    Diag(results_df).run_all()
    stat_tests(results_df)
    abstract_stats(results_df, problems)

    print("\n🎉 MODALBENCH DEMO COMPLETE!")
    print(f"   Outputs: {C.OUT}/")
    print(f"   Figures: {C.OUT}/{C.FIG}/")
    print("\n💡 To run real evaluation, set API keys and call run_evaluation_async()")

