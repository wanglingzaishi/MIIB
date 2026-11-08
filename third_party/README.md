# Third-party source and licensing inventory

Checked against the upstream repositories on 2026-11-06. License texts are retained in `licenses/` for the components identified below. This inventory does not assign a new license to third-party work.

| Upstream | Identified files/components | Observed license |
| --- | --- | --- |
| [CoMM](https://github.com/Duplums/CoMM) | Source infrastructure extracted through the authors' `comm2` research workspace | Repository metadata reports no license; no root LICENSE was present when checked. Permission/license confirmation is pending. |
| [MultiBench](https://github.com/pliang279/MultiBench) | Affect/MIMIC loader ancestry and benchmark infrastructure | [MIT](licenses/MULTIBENCH-MIT.txt) |
| [Stanford multimodal representation](https://github.com/stanford-iprl-lab/multimodal_representation) | `models/robotics.py`, `dataset/robotics/multimodal_manipulation.py` | [MIT](licenses/ROBOTICS-MIT.txt) |
| [SLIP](https://github.com/facebookresearch/SLIP) | `dataset/img_caption.py`; probing infrastructure ancestry | [MIT](licenses/SLIP-MIT.txt) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | `dataset/tokenizer.py`, bundled BPE vocabulary | [MIT](licenses/CLIP-MIT.txt) |
| [MoCo v3](https://github.com/facebookresearch/moco-v3) | `build_2d_sincos_posemb` in `models/input_adapters.py`, explicitly attributed in source | [CC BY-NC 4.0](licenses/MOCO-V3-CC-BY-NC-4.0.txt) |
| [SupContrast](https://github.com/HobbitLong/SupContrast) | Supervised contrastive loss ancestry | [BSD 2-Clause](licenses/SUPCONTRAST-BSD-2-Clause.txt) |

[HRIL](https://github.com/brightest66/HRIL) supplied the requested repository and README organization; no HRIL training implementation was imported for this extraction.

The current tree must not be described as uniformly MIT/Apache licensed: the CoMM permission/license and the explicitly attributed noncommercial component still require the authors' resolution before an open-source license is applied to the full distribution. Selecting a license for original MIIB contributions does not resolve those upstream terms. No dataset or pretrained weight redistribution rights are asserted here.
