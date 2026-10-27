# Likelihood-guided MCTS

[简体中文](README.zh-CN.md) | [Source index](../README.md)

Paper: Table 4; Figures 8/11. 4 GPUs, accumulation 2, 16 simulations, 500 updates.

- [Original execution](source_v1/train_mcts.py)
- [Implementation / scoring](source_v1/mcts_search.py)
- [Original protocol or cohort](config_v1/protocol.json)

Read from the already executed server directory; no retraining, inference or scoring was performed for this export. Computation is unchanged. Machine paths and GPU UUIDs are replaced with placeholders; original and published SHA256 values are in the [source manifest](../SOURCE_MANIFEST.json).

See the [input and execution guide](../README.md#inputs-and-execution-order). Full prepared data, some ablation weights and historical receipts are not bundled. This is not a standalone all-experiment launcher. Historical hash gates remain; record relocated provenance in a new run directory instead of disabling checks or overwriting archived records.
