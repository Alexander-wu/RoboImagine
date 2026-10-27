# Tables

Tables1–2 include RoboImagine rows only (no external-baseline ranks). `data/our_model_main_tables.csv` contains their reported values. Tables3–5 retain the within-model feedback/search/memory experiments and required SFT controls.

Run `python build_tables.py` for five standalone preview pages (requiresTeX/latexmk). Existing quantitative summaries are included in `data/`. Main-table1 is author-provided aggregate input, not a new evaluator run. Bridge main-table2 is a32-trajectory validation result. Table5 uses30 matched long-eligible test trajectories; it differs from Figure13's historical11.
