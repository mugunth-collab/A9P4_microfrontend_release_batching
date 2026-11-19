# A9P4 – A Scalable Micro-Frontend Architecture for Modular Native Feature Delivery in Large-Scale Mobile Engineering Environments

Author: Mug (ICE2CPT 2026, Track 6). Application: modular blockchain wallet and P2P energy-trading mobile app.
Method: Sine Cosine Algorithm (SCA) for module release-batching, compared with monolithic, per-module, greedy batching and random-search plans.

Contents
- paper/   IEEE paper (.docx and .pdf, 6 pages)
- data/    dataset CSV (8,000 changes) and its README
- code/    run_all.py (all experiments), figures.py (Figs. 3-6), fig1_flowchart.dot (Fig. 1), requirements.txt, README.md
- results/ all figures (JPG), all tables (CSV), metrics.json

Reproduce:
    pip install -r code/requirements.txt   # plus Graphviz `dot`
    cd code && python3 run_all.py          # about 5 min on 2 free cores
