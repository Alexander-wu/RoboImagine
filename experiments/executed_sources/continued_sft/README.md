# Continued SFT

[简体中文](README.zh-CN.md) | [Source index](../README.md)

Paper: Figures 8/9. Final v4: 4 GPUs, accumulation 2; v3 retained only for cache provenance.

- [Original execution](source_sft_v4/sft/train.py)
- [Implementation / scoring](source_sft_v4/sft/evaluate.py)
- [Original protocol or cohort](sft/rt1_sft500_v1/four_gpu_preparation_admission.json)

Read from the already executed server directory; no retraining, inference or scoring was performed for this export. Computation is unchanged. Machine paths and GPU UUIDs are replaced with placeholders; original and published SHA256 values are in the [source manifest](../SOURCE_MANIFEST.json).

See the [input and execution guide](../README.md#inputs-and-execution-order). Full prepared data, some ablation weights and historical receipts are not bundled. This is not a standalone all-experiment launcher. Historical hash gates remain; record relocated provenance in a new run directory instead of disabling checks or overwriting archived records.
