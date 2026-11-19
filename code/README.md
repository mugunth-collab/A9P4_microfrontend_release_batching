# A9P4 code – SCA release batching for a modular wallet / energy-trading app

Run everything (about 5 min on 2 free CPU cores; it uses 2 worker processes):

    cd code && python3 run_all.py
    dot -Tpng -Gdpi=300 fig1_flowchart.dot -o ../figures/fig1_flowchart.png   # Fig. 1

Input: `../data/A9P4_wallet_microfrontend_delivery.csv`. Post-release columns (lead time, incident flag,
crash-free delta) are never used as optimisation inputs. The incident flag only trains the risk model on
history before each optimisation window and counts realised failed releases.

Pipeline: median imputation per architecture, 0.5/99.5 % clipping, 5 rolling folds (8-week optimisation
window + 4-week test window), logistic incident-risk model, cost model J (delay, release overhead, batch
risk, coupling), SCA with 5 seeds, baselines (monolithic weekly/tuned, per-module, greedy batching,
random search with the same evaluation budget), ablation (2 seeds), F0 x CI sensitivity (folds 1, 3, 5).

Outputs (`../results/`, figures also as 300-dpi PNG in `../figures/`):
- table1_dataset_by_architecture.csv – Table I
- table3_main_comparison.csv – Table III (mean/std over 5 folds of seed-averaged test metrics)
- per_fold_results.csv, all_runs_test.csv – per fold / per run test metrics
- stat_tests.csv – paired t-test and Wilcoxon, SCA vs each baseline (25 fold-seed pairs)
- table4_ablation.csv, ablation_runs.csv – Table IV
- sensitivity_F0_CI.csv, sensitivity_runs.csv – Fig. 5
- cost_components_per_change.csv – Fig. 3
- metrics.json – all settings, fold info, risk-model AUC/AP, plans, statistics
- fig2_convergence, fig3_cost_components, fig4_plans, fig5_sensitivity, fig6_tradeoff (.jpg)
