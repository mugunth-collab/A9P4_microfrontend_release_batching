#!/usr/bin/env python3
"""A9P4 - Sine Cosine Algorithm (SCA) for module release-batching in a micro-frontend
wallet / energy-trading mobile app.  Runs everything: preprocessing, incident-risk model,
SCA, baselines (monolithic / per-module / greedy batching, plus random search), ablation,
sensitivity, statistics, tables and figures.  Usage:  python3 run_all.py  (from any cwd)."""
import os, json, time, shutil, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'A9P4_wallet_microfrontend_delivery.csv')
import sys; sys.path.insert(0, HERE)
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 100})

# ------------------------------------------------------------------ cost model constants
COST = dict(omega=0.05,   # eng-h per change-hour of delivery delay
            F0=2.0,       # fixed cost per release (pipeline run, OTA / store submission), eng-h
            F1=0.4,       # coordination cost per pair of owning teams in a release, eng-h
            F2=1.0,       # regression-test cost per module in the release's test scope, eng-h
            CI=20.0,      # fixed cost of a failed release (hotfix release, triage), eng-h
            cR=3.0,       # re-verification / re-delivery cost per change rolled back with a failed batch, eng-h
            cX=2.0,       # cost per uncontracted cross-module touch shipped out of sync, eng-h
            nu=2.0)       # weight of a shared-dependency bump relative to one cross-module touch
TAU_MIN, TAU_MAX = 4.0, 336.0          # release cadence bounds (h)
GRID = np.geomspace(TAU_MIN, TAU_MAX, 24)
POP, ITERS, A_SCA, RELOC_EVERY = 30, 200, 2.0, 10
SEEDS = [11, 23, 37, 41, 59]
OPT_W, TEST_W = 8 * 168.0, 4 * 168.0    # optimisation window 8 weeks, test window 4 weeks

# ------------------------------------------------------------------ data + preprocessing
df = pd.read_csv(DATA, parse_dates=['merged_ts']).sort_values('merged_ts').reset_index(drop=True)
OUTCOMES = ['lead_time_hours', 'post_release_incident', 'crash_free_sessions_delta_pct']  # never features
prep = {}
for c in ['test_coverage_pct', 'build_time_min', 'code_review_hours']:
    prep[f'missing_{c}'] = int(df[c].isna().sum())
    df[c] = df[c].fillna(df.groupby('architecture')[c].transform('median'))
for c in ['loc_changed', 'files_changed', 'build_time_min', 'code_review_hours', 'bundle_size_delta_mb']:
    lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
    prep[f'clipped_{c}'] = int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c] = df[c].clip(lo, hi)
df['t_h'] = (df.merged_ts - df.merged_ts.min()).dt.total_seconds() / 3600.0
MODULES = sorted(df.feature_module.unique()); M = len(MODULES); K = M
TEAMS = sorted(df.owning_team.unique())
df['mod'] = df.feature_module.map({m: i for i, m in enumerate(MODULES)})
df['team'] = df.owning_team.map({m: i for i, m in enumerate(TEAMS)})
TEAM_OF_MOD = df.groupby('mod').team.agg(lambda x: x.mode()[0]).sort_index().values; NT = len(TEAMS)
TEAM_1H = np.eye(NT, dtype=np.float32)[TEAM_OF_MOD]
df['mfe'] = (df.architecture == 'micro_frontend').astype(int)
df['mm'] = (df.architecture == 'modular_monolith').astype(int)
df['eta'] = df.architecture.map({'monolith': 1.0, 'modular_monolith': 0.5, 'micro_frontend': 0.0})
df['xc'] = (df.cross_module_imports_touched + COST['nu'] * df.shared_dependency_bumped) * (1 - df.has_interface_contract_tests)

FEATS_NUM = ['loc_changed', 'files_changed', 'cross_module_imports_touched', 'teams_involved',
             'test_coverage_pct', 'build_time_min', 'bundle_size_delta_mb', 'code_review_hours']
FEATS_BIN = ['shared_dependency_bumped', 'has_interface_contract_tests', 'security_critical_module',
             'smart_contract_abi_change', 'feature_flagged', 'mfe', 'mm']
def featurise(d):
    X = d[FEATS_NUM].copy()
    for c in ['loc_changed', 'files_changed', 'build_time_min', 'code_review_hours']: X[c] = np.log1p(X[c])
    X = pd.concat([X, d[FEATS_BIN].astype(float), pd.get_dummies(d['mod'], prefix='m').reindex(
        columns=[f'm_{i}' for i in range(M)], fill_value=0).astype(float)], axis=1)
    return X.values

# dataset summary by architecture (Table I) + Mann-Whitney autonomy tests
rows = []
for a in ['monolith', 'modular_monolith', 'micro_frontend']:
    s = df[df.architecture == a]
    rows.append(dict(architecture=a, changes=len(s), median_lead_h=s.lead_time_hours.median(),
                     incident_pct=100 * s.post_release_incident.mean(), mean_teams=s.teams_involved.mean(),
                     mean_xmod=s.cross_module_imports_touched.mean(), contract_pct=100 * s.has_interface_contract_tests.mean(),
                     median_build_min=s.build_time_min.median()))
tab1 = pd.DataFrame(rows); tab1.to_csv(os.path.join(RES, 'table1_dataset_by_architecture.csv'), index=False)
mw = {}
for c in ['teams_involved', 'cross_module_imports_touched', 'lead_time_hours']:
    u = stats.mannwhitneyu(df[df.architecture == 'micro_frontend'][c], df[df.architecture == 'monolith'][c])
    mw[c] = dict(U=float(u.statistic), p=float(u.pvalue))

# ------------------------------------------------------------------ rolling folds
tmax = df.t_h.max()
starts = np.linspace(26 * 168, tmax - TEST_W, 5)       # test-window starts
FOLDS = []
for f, ts in enumerate(starts):
    os_ = ts - OPT_W
    hist = df[df.t_h < os_]                              # risk model only sees history before opt window
    sc = StandardScaler().fit(featurise(hist))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(featurise(hist)), hist.post_release_incident)
    def win(a, b):
        w = df[(df.t_h >= a) & (df.t_h < b)].copy()
        w['p'] = np.clip(clf.predict_proba(sc.transform(featurise(w)))[:, 1], 1e-5, 0.999)
        w['tt'] = w.t_h - a + 1e-6
        return w
    wo, wt = win(os_, ts), win(ts, ts + TEST_W)
    FOLDS.append(dict(fold=f + 1, opt=wo, test=wt, mfe_share=float(wt.mfe.mean()),
                      auc=float(roc_auc_score(wt.post_release_incident, wt.p)),
                      ap=float(average_precision_score(wt.post_release_incident, wt.p)),
                      start=str(df.merged_ts.min() + pd.Timedelta(hours=float(ts)))[:10]))

def pack(w):
    return dict(t=w.tt.values, mod=w['mod'].values, lp=np.log1p(-w.p.values), team=w.team.values,
                xc=w.xc.values.astype(float), e05=(w.eta.values == 0.5).astype(float), e1=(w.eta.values == 1.0).astype(float), n=len(w), inc=w.post_release_incident.values)

# ------------------------------------------------------------------ cost model J (Eqs. 1-7)
JMAX = int(OPT_W / TAU_MIN) + 2
EYE_K = np.eye(K, dtype=np.float32)
def evaluate(a, tau, W, C=COST, detail=False):
    k = a[W['mod']]; tk = tau[k]
    j = np.ceil(W['t'] / tk); s = j * tk
    D = float(np.sum(s - W['t']))
    key = k * JMAX + j.astype(np.int64)
    cnt = np.bincount(key, minlength=K * JMAX); occ = cnt > 0
    inv = (np.cumsum(occ) - 1)[key]; nB = cnt[occ]; nb = len(nB)
    PB = 1 - np.exp(np.bincount(inv, weights=W['lp'], minlength=nb))
    msize = np.bincount(a, minlength=K)                                # modules per train |S_k|
    tteam = ((EYE_K[a].T @ TEAM_1H) > 0).sum(1)   # owning teams per train T_k
    kb = np.zeros(nb, np.int64); kb[inv] = k                           # train of each batch
    eta = np.maximum(0.5 * (np.bincount(inv, weights=W['e05'], minlength=nb) > 0),
                     1.0 * (np.bincount(inv, weights=W['e1'], minlength=nb) > 0))
    scope = msize[kb] + (M - msize[kb]) * eta                          # regression scope (modules), Eq. (4)
    nteam = tteam[kb]
    frac_out = 1 - (msize[k] - 1) / (M - 1)
    O = nb * C['F0'] + C['F1'] * float(np.sum(nteam * (nteam - 1) / 2)) + C['F2'] * float(scope.sum())
    R = float(np.sum(PB * (C['CI'] + C['cR'] * nB)))
    X = C['cX'] * float(np.sum(W['xc'] * frac_out))
    J = C['omega'] * D + O + R + X
    if not detail: return J
    inc_b = np.bincount(inv, weights=W['inc'], minlength=nb) > 0          # realised failed batches
    return dict(J=J, J_per_change=J / W['n'], delay_h=D / W['n'], releases=int(nb), delay_cost=C['omega'] * D,
                overhead=O, risk=R, coupling=X, exp_failed_rel=float(PB.sum()), trains=int((msize > 0).sum()),
                multi_team_rel_pct=100 * float((nteam > 1).mean()), mean_teams=float(nteam.mean()), mean_batch=float(nB.mean()),
                mean_scope=float(scope.mean()), realised_failed_rel=int(inc_b.sum()), realised_rolled_back=int(nB[inc_b].sum()))

def decode(x):
    """x[:M] -> train label of each module; x[M:] -> each module's preferred log-cadence.
    A train's cadence is the geometric mean of its members' preferred cadences."""
    lab = np.minimum((x[:M] * K).astype(int), K - 1)
    _, a = np.unique(lab, return_inverse=True)
    logt = np.log(TAU_MIN) + x[M:] * np.log(TAU_MAX / TAU_MIN)
    n = np.bincount(a, minlength=K)
    tau = np.full(K, TAU_MAX); used = n > 0
    tau[used] = np.exp(np.bincount(a, weights=logt, minlength=K)[used] / n[used])
    return a, tau

# ------------------------------------------------------------------ Sine Cosine Algorithm (Eqs. 8-10)
def relocate(x, f, W, C):
    """Elite refinement: move each module of the destination P to every other existing
    train or to a new train and keep the best improving move; then scale each train's cadence
    by 0.6-1.6 and keep improvements. Returns (x, f, evals)."""
    a, _ = decode(x); labs = np.minimum((x[:M] * K).astype(int), K - 1); used = np.unique(labs)
    free = [l for l in range(K) if l not in used]
    best = (f, None); ev = 0
    for m in range(M):
        for l in list(used) + free[:1]:
            if l == labs[m]: continue
            y = x.copy(); y[m] = (l + 0.5) / K; fy = evaluate(*decode(y), W, C); ev += 1
            if fy < best[0]: best = (fy, y)
    if best[1] is not None: x, f = best[1], best[0]
    a, _ = decode(x)                                    # cadence refinement of each train of P
    for g in range(a.max() + 1):
        for fac in (0.6, 0.8, 1.25, 1.6):
            y = x.copy(); y[M:][a == g] = np.clip(y[M:][a == g] + np.log(fac) / np.log(TAU_MAX / TAU_MIN), 0, 1)
            fy = evaluate(*decode(y), W, C); ev += 1
            if fy < f: x, f = y, fy
    return x, f, ev

def sca(W, seed, C=COST, iters=ITERS, pop=POP, variant='full'):
    rng = np.random.default_rng(seed); dim = 2 * M
    X = rng.random((pop, dim))
    if variant == 'cadence_only': X[:, :M] = (np.arange(M) + 0.5) / K   # per-module trains, only tau searched
    fit = np.array([evaluate(*decode(x), W, C) for x in X])
    b = fit.argmin(); P, Pf = X[b].copy(), fit[b]; curve = []; nev = pop
    for t in range(iters):
        r1 = A_SCA if variant == 'const_r1' else A_SCA - t * A_SCA / iters        # Eq. (10)
        r2 = rng.uniform(0, 2 * np.pi, (pop, dim)); r3 = rng.uniform(0, 2, (pop, dim)); r4 = rng.random((pop, dim))
        if variant == 'sine_only': r4[:] = 0.0
        if variant == 'cosine_only': r4[:] = 1.0
        step = np.abs(r3 * P - X)
        X = np.where(r4 < 0.5, X + r1 * np.sin(r2) * step, X + r1 * np.cos(r2) * step)  # Eq. (9)
        X = np.clip(X, 0, 1 - 1e-9)
        if variant == 'cadence_only': X[:, :M] = (np.arange(M) + 0.5) / K
        fit = np.array([evaluate(*decode(x), W, C) for x in X]); nev += pop
        b = fit.argmin()
        if fit[b] < Pf: P, Pf = X[b].copy(), fit[b]
        if variant not in ('no_relocation', 'cadence_only') and (t + 1) % RELOC_EVERY == 0:
            P, Pf, e = relocate(P, Pf, W, C); X[fit.argmax()] = P; nev += e   # re-inject refined elite
        curve.append(Pf)
    return decode(P), np.array(curve), nev

def random_search(W, seed, n, C=COST):
    rng = np.random.default_rng(seed); best, bx, curve = np.inf, None, []
    step = n / ITERS                         # report the curve on the SCA iteration axis
    for i in range(n):
        x = rng.random(2 * M); f = evaluate(*decode(x), W, C)
        if f < best: best, bx = f, x
        if len(curve) < ITERS and i + 1 >= POP + (len(curve) + 1) * step - POP: curve.append(best)
    while len(curve) < ITERS: curve.append(best)
    return decode(bx), np.array(curve)

# ------------------------------------------------------------------ baselines
def best_tau_for(a, W, members, tau, C):
    bt, bf = None, np.inf
    for g in GRID:
        tt = tau.copy(); tt[members] = g; f = evaluate(a, tt, W, C)
        if f < bf: bf, bt = f, g
    return bt

def monolithic(W, C=COST):
    a = np.zeros(M, int); tau = np.full(K, TAU_MAX)
    tau[0] = best_tau_for(a, W, [0], tau, C); return a, tau

def monolithic_weekly(W, C=COST):
    a = np.zeros(M, int); tau = np.full(K, TAU_MAX); tau[0] = 168.0; return a, tau

def per_module(W, C=COST):
    a = np.arange(M); tau = np.full(K, 24.0)
    for _ in range(2):                       # coordinate descent on per-module cadences
        for m in range(M): tau[m] = best_tau_for(a, W, [m], tau, C)
    return a, tau

def greedy(W, C=COST):
    a, tau = per_module(W, C); cur = evaluate(a, tau, W, C)
    while True:
        labs = np.unique(a); best = (cur, None)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a2 = np.where(a == labs[j], labs[i], a)
                for g in GRID:
                    t2 = tau.copy(); t2[labs[i]] = g; f = evaluate(a2, t2, W, C)
                    if f < best[0]: best = (f, (a2, t2))
        if best[1] is None: break
        cur, (a, tau) = best
    _, a2 = np.unique(a, return_inverse=True); t2 = np.full(K, TAU_MAX)
    for new, old in enumerate(np.unique(a)): t2[new] = tau[old]
    return a2, t2

# ------------------------------------------------------------------ experiment jobs (2 worker processes)
from multiprocessing import Pool
BASES = [('Monolithic (weekly)', monolithic_weekly), ('Monolithic (tuned)', monolithic),
         ('Per-module', per_module), ('Greedy batching', greedy)]
def job(spec):
    kind, f, seed, extra = spec
    F = FOLDS[f - 1]; Wo, Wt = pack(F['opt']), pack(F['test']); C = extra.get('C', COST)
    t1 = time.time()
    if kind == 'base':
        out = []
        for name, fn in BASES:
            t1 = time.time(); a, tau = fn(Wo, C); el = time.time() - t1
            out.append(dict(fold=f, method=name, opt_J_per_change=evaluate(a, tau, Wo, C) / Wo['n'], runtime_s=el,
                            plan=(a.tolist(), tau[:a.max() + 1].tolist()), **evaluate(a, tau, Wt, C, detail=True)))
        return out
    if kind == 'sca':
        (a, tau), cv, nev = sca(Wo, seed + f, C=C, iters=extra.get('iters', ITERS), variant=extra.get('variant', 'full'))
    else:
        (a, tau), cv = random_search(Wo, seed + f, extra['n'], C=C); nev = extra['n']
    return dict(fold=f, seed=seed, kind=kind, variant=extra.get('variant', 'full'), tag=extra.get('tag'),
                opt_J_per_change=evaluate(a, tau, Wo, C) / Wo['n'], runtime_s=time.time() - t1, evals=nev,
                curve=(cv / Wo['n']).tolist(), plan=(a.tolist(), tau[:a.max() + 1].tolist()),
                **evaluate(a, tau, Wt, C, detail=True))

methods = ['Monolithic (weekly)', 'Monolithic (tuned)', 'Per-module', 'Greedy batching', 'Random search', 'SCA (proposed)']
ABL = [('full', 'Full SCA'), ('no_relocation', 'No elite refinement'), ('const_r1', 'Constant r1 = 2'),
       ('sine_only', 'Sine branch only'), ('cadence_only', 'Cadence only (no grouping)')]
F0s, CIs = [0.5, 2.0, 8.0], [5.0, 20.0, 80.0]
with Pool(2) as pool:
    folds = [F['fold'] for F in FOLDS]
    base_out = pool.map(job, [('base', f, 0, {}) for f in folds])
    sca_out = pool.map(job, [('sca', f, s, {}) for f in folds for s in SEEDS])
    print('SCA done', round(time.time() - T0), 's', flush=True)
    n_rs = int(np.mean([r['evals'] for r in sca_out]))
    rs_out = pool.map(job, [('rs', f, s, {'n': n_rs}) for f in folds for s in SEEDS])
    print('random search done', round(time.time() - T0), 's', flush=True)
    abl_out = pool.map(job, [('sca', f, s, {'variant': v}) for v, _ in ABL[1:] for f in folds for s in SEEDS[:2]])
    print('ablation done', round(time.time() - T0), 's', flush=True)
    sens_specs = [(f0, ci, f) for f0 in F0s for ci in CIs for f in (1, 3, 5)]
    sens_sca = pool.map(job, [('sca', f, 7, {'C': dict(COST, F0=f0, CI=ci), 'iters': 100}) for f0, ci, f in sens_specs])
    sens_base = pool.map(job, [('base', f, 0, {'C': dict(COST, F0=f0, CI=ci)}) for f0, ci, f in sens_specs])
    print('sensitivity done', round(time.time() - T0), 's', flush=True)

# ------------------------------------------------------------------ main results
rec = []
for out in base_out:
    for r in out:
        for s in SEEDS: rec.append(dict(r, seed=s))   # deterministic baselines repeated per seed for paired tests
for r in sca_out: rec.append(dict(r, method='SCA (proposed)'))
for r in rs_out: rec.append(dict(r, method='Random search'))
plans = {r['fold']: r['plan'] for r in sca_out if r['seed'] == SEEDS[0]}
greedy_plans = {r['fold']: r['plan'] for out in base_out for r in out if r['method'] == 'Greedy batching'}
curves = {'SCA (proposed)': [np.array(r['curve']) for r in sca_out], 'Random search': [np.array(r['curve']) for r in rs_out]}
R = pd.DataFrame(rec).drop(columns=['curve', 'plan', 'kind', 'variant', 'tag'], errors='ignore')
R.to_csv(os.path.join(RES, 'all_runs_test.csv'), index=False)
mono = R[R.method == 'Monolithic (tuned)'].set_index(['fold', 'seed']).J
R['saving_vs_mono_pct'] = [100 * (1 - j / mono.loc[(f, s)]) for j, f, s in zip(R.J, R.fold, R.seed)]
cols = ['J_per_change', 'opt_J_per_change', 'saving_vs_mono_pct', 'delay_h', 'releases', 'trains', 'mean_teams', 'mean_scope',
        'exp_failed_rel', 'realised_failed_rel', 'realised_rolled_back', 'mean_batch', 'runtime_s']
fm = R.groupby(['method', 'fold'])[cols].mean().reset_index()      # seed-mean per fold
main = fm.groupby('method')[cols].agg(['mean', 'std']).reindex(methods)
main.columns = [f'{a}_{b}' for a, b in main.columns]
main.to_csv(os.path.join(RES, 'table3_main_comparison.csv'))
fm.to_csv(os.path.join(RES, 'per_fold_results.csv'), index=False)
seed_std = R[R.method == 'SCA (proposed)'].groupby('fold').J_per_change.std().mean()

tests = {}
sc_ = R[R.method == 'SCA (proposed)'].set_index(['fold', 'seed']).J_per_change
for m in methods[:-1]:
    b = R[R.method == m].set_index(['fold', 'seed']).J_per_change.loc[sc_.index]
    t = stats.ttest_rel(sc_, b); w = stats.wilcoxon(sc_, b)
    fsca = fm[fm.method == 'SCA (proposed)'].J_per_change.values; fb = fm[fm.method == m].J_per_change.values
    tests[m] = dict(mean_diff=float((sc_ - b).mean()), t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue),
                    sca_better_pairs=int((sc_ < b).sum()), n_pairs=int(len(sc_)),
                    sca_better_folds=int((fsca < fb).sum()), fold_level_t_p=float(stats.ttest_rel(fsca, fb).pvalue))
pd.DataFrame(tests).T.to_csv(os.path.join(RES, 'stat_tests.csv'))

# ------------------------------------------------------------------ ablation table
arec = [dict(variant='Full SCA', **{k: r[k] for k in ('fold', 'seed', 'opt_J_per_change', 'J_per_change', 'trains', 'evals')})
        for r in sca_out if r['seed'] in SEEDS[:2]]
lab = dict(ABL)
arec += [dict(variant=lab[r['variant']], **{k: r[k] for k in ('fold', 'seed', 'opt_J_per_change', 'J_per_change', 'trains', 'evals')})
         for r in abl_out]
A = pd.DataFrame(arec); A.to_csv(os.path.join(RES, 'ablation_runs.csv'), index=False)
abl = A.groupby(['variant', 'fold']).mean(numeric_only=True).groupby('variant')[
    ['opt_J_per_change', 'J_per_change', 'trains', 'evals']].agg(['mean', 'std']).reindex([l for _, l in ABL])
abl.columns = [f'{a}_{b}' for a, b in abl.columns]; abl.to_csv(os.path.join(RES, 'table4_ablation.csv'))
acurve = {lab[v]: [np.array(r['curve']) for r in abl_out if r['variant'] == v] for v, _ in ABL[1:]}
acurve_ref = {lab[v]: [next(x for x in sca_out if x['fold'] == r['fold'] and x['seed'] == r['seed'])['curve'][-1]
                       for r in abl_out if r['variant'] == v] for v, _ in ABL[1:]}

# ------------------------------------------------------------------ sensitivity table
srec = []
for (f0, ci, f), rs, rb in zip(sens_specs, sens_sca, sens_base):
    g = next(x for x in rb if x['method'] == 'Greedy batching'); mo = next(x for x in rb if x['method'] == 'Monolithic (tuned)')
    srec.append(dict(F0=f0, CI=ci, fold=f, sca_J=rs['J'], greedy_J=g['J'], mono_J=mo['J'], sca_trains=rs['trains'],
                     greedy_trains=g['trains'], sca_release_per_week=rs['releases'] / 4,
                     sca_opt=rs['opt_J_per_change'], greedy_opt=g['opt_J_per_change']))
S = pd.DataFrame(srec)
S['sca_vs_greedy_pct'] = 100 * (1 - S.sca_J / S.greedy_J); S['sca_vs_mono_pct'] = 100 * (1 - S.sca_J / S.mono_J)
S.to_csv(os.path.join(RES, 'sensitivity_runs.csv'), index=False)
sens = S.groupby(['F0', 'CI'])[['sca_vs_greedy_pct', 'sca_vs_mono_pct', 'sca_trains', 'greedy_trains', 'sca_release_per_week']].mean().reset_index()
sens.to_csv(os.path.join(RES, 'sensitivity_F0_CI.csv'), index=False)

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})
    plt.close(fig)

# Fig 2 convergence: gap of the best-so-far cost to the final full-SCA cost of the same fold and seed
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ref = np.array([r['curve'][-1] for r in sca_out])[:, None]
for lab, cv, c in [('SCA (proposed)', curves['SCA (proposed)'], 'C3'), ('Random search', curves['Random search'], '0.4')]:
    arr = 100 * (np.array(cv) / ref - 1); mu, sd = arr.mean(0), arr.std(0); it = np.arange(1, arr.shape[1] + 1)
    ax.plot(it, mu, color=c, label=lab, lw=1.2); ax.fill_between(it, np.maximum(mu - sd, 0), mu + sd, color=c, alpha=0.2, lw=0)
for lab, ls in [('No elite refinement', '--'), ('Constant r1 = 2', ':')]:
    arr = 100 * (np.array(acurve[lab]) / np.array(acurve_ref[lab])[:, None] - 1)
    ax.plot(np.arange(1, arr.shape[1] + 1), arr.mean(0), color='C0', ls=ls, lw=1, label=lab)
ax.set_xlabel(f'Iteration ({POP} plan evaluations each)'); ax.set_ylabel('Gap to final SCA cost (%)')
ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, None); ax.grid(alpha=0.3); ax.legend(frameon=False)
save(fig, 'fig2_convergence')

pd.DataFrame([dict(parameter=k, value=v) for k, v in list(COST.items()) + [('tau_min_h', TAU_MIN), ('tau_max_h', TAU_MAX),
    ('agents', POP), ('iterations', ITERS), ('a', A_SCA), ('max_trains_K', K), ('dimension', 2 * M), ('refine_every', RELOC_EVERY),
    ('seeds', len(SEEDS)), ('folds', len(FOLDS))]]).to_csv(os.path.join(RES, 'table2_settings.csv'), index=False)

# cost components per change (Fig. 3)
R['n_test'] = [len(FOLDS[f - 1]['test']) for f in R.fold]
comp = R.groupby('method')[['delay_cost', 'overhead', 'risk', 'coupling']].sum().div(R.groupby('method').n_test.sum(), axis=0).reindex(methods)
comp.to_csv(os.path.join(RES, 'cost_components_per_change.csv'))

# ------------------------------------------------------------------ metrics.json
metrics = dict(preprocessing=prep, n_changes=len(df), modules=MODULES, cost_constants=COST,
               sca=dict(pop=POP, iters=ITERS, a=A_SCA, tau_bounds=[TAU_MIN, TAU_MAX], dims=2 * M, seeds=SEEDS, reloc_every=RELOC_EVERY),
               folds=[dict(fold=F['fold'], test_start=F['start'], n_opt=len(F['opt']), n_test=len(F['test']),
                           mfe_share=F['mfe_share'], risk_auc=F['auc'], risk_ap=F['ap'],
                           test_incident_rate=float(F['test'].post_release_incident.mean())) for F in FOLDS],
               main=main.round(4).to_dict(orient='index'), stat_tests=tests, sca_mean_seed_std_J=float(seed_std),
               ablation=abl.round(4).to_dict(orient='index'), mannwhitney_mfe_vs_monolith=mw,
               sca_plans_seed11=plans, greedy_plans=greedy_plans, n_eval_random_search=n_rs,
               sensitivity=sens.round(3).to_dict(orient='records'), runtime_s=round(time.time() - T0, 1))
json.dump(metrics, open(os.path.join(RES, 'metrics.json'), 'w'), indent=1, default=float)
import figures; figures.make(RES, FIG)            # Figs. 3-6 from the saved results
import subprocess                                  # Fig. 1 (Graphviz)
subprocess.run(['dot', '-Tpng', '-Gdpi=300', os.path.join(HERE, 'fig1_flowchart.dot'), '-o', os.path.join(FIG, 'fig1_flowchart.png')], check=False)
subprocess.run(['dot', '-Tjpg', '-Gdpi=300', os.path.join(HERE, 'fig1_flowchart.dot'), '-o', os.path.join(RES, 'fig1_flowchart.jpg')], check=False)
print(main[['J_per_change_mean', 'J_per_change_std', 'saving_vs_mono_pct_mean', 'delay_h_mean', 'releases_mean', 'trains_mean']].round(3))
print(abl.round(3)); print(pd.DataFrame(tests).T.round(4)); print(sens.round(2))
print('total runtime', round(time.time() - T0), 's')
