# MIIB: Multimodal Information Interactions Bottleneck

Liangjian Wen, Jiangtao Zheng, Jiang Duan, Qun Dai, Yong Dai, Jianzhuang Liu, Jun Wang, Shu-Dong Huang, Xinwang Liu, Zenglin Xu, Zhao Kang

Official PyTorch implementation of **Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning**, accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026**.

[[Paper / DOI](https://doi.org/10.1109/TPAMI.2026.3751756)] · [Quick Start](#-quick-start) · [Reproduction](#-reproduction-check) · [Citation](#-citation)

---

## 🎯 TL;DR

MIIB learns three complementary kinds of multimodal representation:

- modality-specific information from every input modality;
- modality-fused information shared by the joint observation;
- synergistic information that emerges only through cross-modal interaction.

Two Transformer streams separately model fusion and synergy. The complete objective combines modality-specific, modality-fusion and synergy terms while retaining the historical linear-probing protocol used by the paper.

## 📊 Key Results

The following values are reported in the accepted paper. They are targets for reproduction, not results newly generated during this repository cleanup.

### Controlled information · Trifeature

| Method | Redundancy ↑ | Uniqueness ↑ | Synergy ↑ |
| --- | ---: | ---: | ---: |
| CoMM | 99.9 ± 0.06 | 86.8 ± 2.99 | 71.4 ± 3.47 |
| **MIIB** | **99.9 ± 0.12** | **89.8 ± 3.23** | **79.1 ± 2.97** |

### Classification · V&T and UR-FUNNY

| Modalities | V&T contact ↑ | UR-FUNNY ↑ |
| ---: | ---: | ---: |
| 2 | **88.4 ± 0.32** | **65.4 ± 0.90** |
| 3 | **94.2 ± 0.11** | **66.2 ± 0.85** |

### MultiBench and MM-IMDb

| Metric | MIIB |
| --- | ---: |
| V&T EE MSE × 10⁻⁴ ↓ | **5.74 ± 0.69** |
| MIMIC accuracy ↑ | **68.1 ± 0.3** |
| MOSI accuracy ↑ | **67.4 ± 1.8** |
| MUSTARD accuracy ↑ | **66.0 ± 2.0** |
| MM-IMDb weighted F1 ↑ | **62.80 ± 0.24** |
| MM-IMDb macro F1 ↑ | **56.20 ± 0.16** |

## 🧩 Approach

1. Create two augmented views of every modality and encode them with modality-specific backbones and latent converters.
2. Use the Fusion Transformer to obtain individual-modality and fused representations.
3. Use the independent Synergy Transformer to model cross-modal interactions.
4. Optimize modality-specific, modality-fusion and synergy objectives through a shared projection head.

The architecture figure is available in [`assets/Model_Framework.pdf`](assets/Model_Framework.pdf).

## 🗂 Repository Structure

The layout follows the concise organization of the [HRIL repository](https://github.com/brightest66/HRIL):

```text
MIIB/
├── assets/          # architecture figure
├── configs/         # one complete YAML profile per paper experiment
├── dataset/         # benchmark data modules and portable catalog
├── evaluation/      # linear probes and metrics
├── losses/          # MIIB objective
├── models/          # MIIB encoder and modality backbones
├── pl_modules/      # Lightning training modules
├── run_scripts/     # training, testing and aggregation commands
├── third_party/     # upstream attribution and license texts
├── run.py           # unified SSL training/test entry point
├── evaluate.py      # evaluate a saved SSL checkpoint
├── finetune.py      # downstream fine-tuning helper
├── environment.yml
└── README.md
```

Datasets, pretrained backbones, checkpoints and experiment outputs are intentionally excluded.

## 🚀 Quick Start

### 1. Installation

```bash
git clone <MIIB-repository-url>
cd MIIB
conda env create -f environment.yml
conda activate miib
```

The recorded stack is Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, CUDA 11.8 and PyTorch Lightning 2.1.1.

### 2. Data and pretrained backbones

MIIB expects the preprocessed files used by the original experiments:

```text
<data-root>/
├── mosi/mosi_data.pkl
├── sarcasm/sarcasm.pkl
├── humor/humor.pkl
├── mimic/im.pk
├── visionandtouch/triangle_real_data/*.h5
├── trifeatures_3combi/{train,test}/*.png
└── mmimdb/{split.json,dataset/}
```

Set the external data root before training:

```bash
export MIIB_DATA_ROOT=/absolute/path/to/preprocessed-data
export TOKENIZERS_PARALLELISM=false
```

MM-IMDb additionally uses CLIP ViT-B/32 and `clip-ViT-B-32-multilingual-v1`. For offline execution:

```bash
export MIIB_VISION_CHECKPOINT=/absolute/path/to/vit_base_patch32_clip_224.openai/pytorch_model.bin
export MIIB_TEXT_MODEL=/absolute/path/to/clip-ViT-B-32-multilingual-v1
```

Relative entries in [`dataset/catalog.json`](dataset/catalog.json) resolve below `MIIB_DATA_ROOT`. Set `MIIB_CATALOG` to use another catalog. Data and model weights are not redistributed by this repository.

### 3. Training

Available profiles are:

| Profile | Experiment |
| --- | --- |
| `trifeatures_ru` | redundancy and uniqueness |
| `trifeatures_synergy` | controlled synergy |
| `mosi`, `mustard`, `mimic` | two-modality MultiBench tasks |
| `urfunny2`, `urfunny3` | two-/three-modality UR-FUNNY |
| `vt_contact2`, `vt_contact3` | two-/three-modality V&T contact |
| `vt_ee` | V&T end-effector regression |
| `mmimdb` | poster/plot multilabel classification |

Run one seed or the five paper seeds (42–46):

```bash
CUDA_VISIBLE_DEVICES=0 bash run_scripts/run.sh mosi
CUDA_VISIBLE_DEVICES=0 bash run_scripts/run_main.sh mosi
```

One training process uses one GPU. With four GPUs, launch independent profiles or seeds on different devices:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_scripts/run_main.sh mosi
CUDA_VISIBLE_DEVICES=1 bash run_scripts/run_main.sh mustard
CUDA_VISIBLE_DEVICES=2 bash run_scripts/run_main.sh urfunny2
CUDA_VISIBLE_DEVICES=3 bash run_scripts/run_main.sh mimic
```

Hydra overrides can follow the profile name. For a short run that does not save checkpoints:

```bash
bash run_scripts/run.sh mosi trainer.max_epochs=3 save_checkpoints=false
```

Outputs are written to `outputs/MIIB/<profile>/seed_<seed>/` and are ignored by Git.

### 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash run_scripts/test.sh mosi /absolute/path/to/model.ckpt
```

Or evaluate directly from the architecture and probe settings stored in a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint /absolute/path/to/model.ckpt --device gpu
```

Aggregate one final `result.json` per seed with an explicit standard-deviation convention:

```bash
python run_scripts/summarize_results.py \
  /path/to/seed42/result.json /path/to/seed43/result.json \
  /path/to/seed44/result.json /path/to/seed45/result.json \
  /path/to/seed46/result.json --metric acc1 --scale 100 --ddof 0
```

Use `f1_weighted` / `f1_mean` for MM-IMDb and `mse --scale 10000` for V&T EE. Use `--ddof 1` for sample standard deviation.

### 5. Fine-tuning

```bash
python finetune.py --experiment mosi --checkpoint /path/to/ssl.ckpt \
  --lr 5e-5 --head-lr 1e-3 --dropout 0.1 --epochs 100
```

## 🔎 Original-Code Mapping

The research workspace retained the historical name `CoMM`, but the mapped files contain the MIIB dual-Transformer architecture and five-term objective. The public implementation uses MIIB names throughout the core:

| Function | Historical path in `comm2` | Public MIIB path |
| --- | --- | --- |
| Lightning model | `pl_modules/comm.py::CoMM` | `pl_modules/miib.py::MIIB` |
| fusion/synergy encoder | `models/mmfusion.py::MMFusion` | `models/miib.py::MIIBEncoder` |
| objective | `losses/comm_loss.py::CoMMLoss` | `losses/miib_loss.py::MIIBLoss` |
| two-modality entry | `main_multibench.py` | `run.py` + experiment YAML |
| three-modality entry | `main_multibench_all-mod.py` | `run.py` + experiment YAML |
| V&T regression entry | `main_multibench_reg.py` | `run.py` + `vt_ee.yaml` |
| Trifeature / MM-IMDb | `main_trifeatures.py` / `main_mmimdb.py` | `run.py` + corresponding YAML |

The extraction preserves model construction order, shared projection head, optimizer grouping, loss order and historical probe behavior. Portable device/path handling replaces hard-coded CUDA and local absolute paths. `compat.py` only translates metadata in older checkpoints.

## ✅ Reproduction Check

On 6 November 2026, the extracted MIIB code was compared with the corresponding `comm2` implementation on an RTX 4090 using the archived four-worker runtime:

| Dataset | Scope | Maximum metric difference | Final parameter state |
| --- | --- | ---: | --- |
| MUSTARD | 3 complete epochs | **0** | identical |
| MOSI | 3 complete epochs | **0** | identical |

Every recorded train/validation/probe metric was compared, and the final state dictionaries had identical SHA-256 hashes. The temporary runs and comparison report were deleted after verification as requested; no checkpoint was retained.

This verifies the extraction and early training trajectory. It is not a new five-seed, 100-epoch rerun of every paper table. In particular, the final V&T EE run record was not found in the inspected `comm2` paths, so `vt_ee.yaml` follows `comm2/run_scripts/reg_train.sh`; the old test-script path label disagrees on one penalty value and should not be treated as authoritative saved hyperparameters.

The historical protocol is retained for numerical comparability: classification `acc1` is macro per-class accuracy; V&T and Trifeature reuse their held-out split for validation/test; MIMIC normalization precedes its fixed split; MM-IMDb plot selection depends on the random worker sequence.

## 📝 Citation

```bibtex
@article{wen2026miib,
  title={Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning},
  author={Wen, Liangjian and Zheng, Jiangtao and Duan, Jiang and Dai, Qun and
          Dai, Yong and Liu, Jianzhuang and Wang, Jun and Huang, Shu-Dong and
          Liu, Xinwang and Xu, Zenglin and Kang, Zhao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  doi={10.1109/TPAMI.2026.3751756}
}
```

## 🙏 Acknowledgments

This codebase builds on [CoMM](https://github.com/Duplums/CoMM) and inherited components listed in [`third_party/README.md`](third_party/README.md). The repository and README organization follow the requested style of [HRIL](https://github.com/brightest66/HRIL); no HRIL training code was imported.

The authors have not assigned a project-wide license yet. Third-party notices remain in `third_party/`.
