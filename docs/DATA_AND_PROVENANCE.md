# Data and asset provenance

`release_manifest.json` on the model hub records the SHA-256, byte count, checkpoint identity and original trajectory IDs for every published asset. Checkpoint tensors and codec weights are copied byte-for-byte from the experiments. Optimizer states, private paths, credentials and cluster control scripts are excluded.

The six input examples are original evaluation indices 0 and 1 from each dataset's reporting/evaluation list, not selected for a favorable score. Each contains two initial observations, the next 33 action entries, instruction text and 32 GT frames. Saved prediction references are separate files. Demonstration clips are converted/cropped to the model's existing 256×320 preprocessing and preserve the archived action layout. Source-clip hashes and logical/source split labels are recorded in the manifest.

## Dataset attribution

- **RT-1 / Fractal20220817:** Robotics at Google, Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*. [Official project](https://robotics-transformer1.github.io/) and [TensorFlow Datasets catalogue](https://www.tensorflow.org/datasets/catalog/fractal20220817_data). The source's rights and terms remain applicable; no new blanket dataset license is asserted here.
- **BridgeData V2:** Walke et al., *BridgeData V2: A Dataset for Robot Learning at Scale*, CoRL 2023. [Official dataset page](https://rail-berkeley.github.io/bridgedata/) provides the data under CC BY 4.0. The small packaged clips are preprocessed derivatives; credit remains with the dataset authors.
- **RoboCasa / RoboCasa365:** Nasiriany et al., *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots* (RSS 2024) and *RoboCasa365* (ICLR 2026). [Official repository](https://github.com/robocasa/robocasa) specifies CC BY 4.0 for assets/datasets and MIT for code. Packaged clips preserve the experiment's temporal/action preprocessing.

## Implementation attribution

The AR backbone uses Transformers' LLaMA implementation; text features use the frozen, pinned `google-t5/t5-base` encoder. The visual codec derives from iVideoGPT/diffusers and finite scalar quantization. Existing file-level notices and full upstream license texts are retained in `licenses/`. Policy-objective and reward primitives retain their original ByteDance/Hugging Face notices. The release does not claim that these standard components originate in this work.

`provenance/model_source_sha256.json` records the unmodified archived modules used to assemble this inference package. Public download, CLI and notebook wrappers are new. The internal `futureworlds_config.json` filename is retained for compatibility with the original checkpoint loader; it is not a claim that the old FutureWorlds checkpoints are the released RoboImagine post-trained models. Model selection is explicit in the hub manifest.
