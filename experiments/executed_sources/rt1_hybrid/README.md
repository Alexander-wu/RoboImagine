# RT-1 Hybrid / GDBS

[简体中文](README.zh-CN.md) | [Source index](../README.md)

Paper: Table 2/3/4; Figures 8/9/10. 8 GPUs; 500 updates; global batch 8.

- [Original execution](train.py)
- [Implementation / scoring](common.py)
- [Original protocol or cohort](protocol.json)

Read from the already executed server directory; no retraining, inference or scoring was performed for this export. Computation is unchanged. Machine paths and GPU UUIDs are replaced with placeholders; original and published SHA256 values are in the [source manifest](../SOURCE_MANIFEST.json).

See the [input and execution guide](../README.md#inputs-and-execution-order). Full prepared data, some ablation weights and historical receipts are not bundled. This is not a standalone all-experiment launcher. Historical hash gates remain; record relocated provenance in a new run directory instead of disabling checks or overwriting archived records.
