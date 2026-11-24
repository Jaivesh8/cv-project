# P44 · Object Hallucination Audit of Multimodal LLMs

This UCS532 project asks whether three open vision-language checkpoints give consistent yes/no answers when the wording changes and the image and target object stay fixed. It measures both false affirmations of absent objects and misses of present objects.

## Dataset and checkpoints

The fixed main set contains 500 object-existence probes on 250 distinct [COCO val2014](https://cocodataset.org/#download) images. Each contributes one corrected present probe and one corrected absent probe. Random, popular, and adversarial negative strata contribute 84, 83, and 83 images, respectively. Positives come from the original random split's RePOPE counterpart, including annotation corrections. The disjoint pilot contains 12 probes on six images. Neither set is a training set.

The sampler joins [official POPE](https://github.com/RUCAIBox/POPE) and [RePOPE](https://github.com/YanNeu/RePOPE) by split and question ID and verifies the image and question match. RePOPE excludes ambiguous cases before sampling. The main analysis uses corrected labels and records a matched rescoring with original labels. `protocol.json` pins all revisions, the sampling seed (53244), the bootstrap seed (53245), and the full prompt templates. The log records hashes of the annotations, probe manifest, and image set.

All checkpoints receive a single image per probe with the longest edge capped at 512 pixels. Tiling is disabled and the runner checks the processor output size before generation.

The runner isolates checkpoints in separate Python processes and reuses the frozen vision features for identical image inputs within each checkpoint. The Windows compatibility path reads the pinned [Safetensors format](https://github.com/huggingface/safetensors#format) in buffered chunks to avoid a failure in the memory-mapped checkpoint loader.

| Name in tables | Open checkpoint |
|---|---|
| smol256 | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| smol500 | [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) |
| smol2video | [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct), used with still images |

## Reproduce

Use Python 3.12 and Bash (Git Bash on Windows). Install the pinned dependencies in an activated environment.

```bash
python -m pip install -r requirements.txt
bash run_all.sh
```

The command downloads the pinned annotations as needed. If the complete recorded prediction log and main table are present, it rebuilds the tables and plots from that log. Otherwise it downloads just the required COCO images and checkpoint files to `.cache/`, runs inference, and writes `results/main.md` and `results/main.csv`. CUDA uses float16 and CPU uses float32. The project machine has a 4 GB NVIDIA GeForce GTX 1650. On that Windows machine the CUDA packages can be installed using the command below. The runtime measured per model and arm is available in `results/timing.csv` after a complete run. Timing includes per-batch preprocessing and decoding and excludes download and model load.

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.7.1+cu118' 'torchvision==0.22.1+cu118' -r requirements.txt
```

Run fresh inference with `bash run_all.sh --out .cache/fresh-run`. The seed, two input manifests, model revisions, runtime hardware and software are printed during the run. The CSVs from the two runs can be compared directly for the same hardware and configuration. Numerical differences across devices or kernels may change some greedy decisions.

For a remote GPU, open the [Colab notebook](https://colab.research.google.com/github/Jaivesh8/cv-project/blob/main/colab/P44_Audit.ipynb), select a GPU runtime, and run all cells. It installs into an isolated venv, logs the actual compute device, and downloads a small archive after validating all 6,000 prediction rows.

For the offline viva, cache the separate pilot with `bash run_all.sh --pilot` while online. Demonstrate inference with networking disabled via `bash run_all.sh --pilot --offline --out .cache/viva` (use a fresh output directory for a fresh run).

## Results

The recorded run used **NVIDIA GeForce GTX 1650**, torch.float16, PyTorch 2.7.1+cu118, Transformers 4.53.3, and batch size 2. The 6,000 responses took **38.98 minutes** of inference batches, excluding downloads and checkpoint loading.

| Model | Prompt | Precision | Recall | F1 | False yes | False no | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| smol256 | W0 | 84.71% | 82.00% | 83.33% | 14.80% | 18.00% | 0.00% |
| smol256 | W1 | 90.91% | 72.00% | 80.36% | 7.20% | 28.00% | 0.00% |
| smol256 | W2 | 93.81% | 42.40% | 58.40% | 2.80% | 57.60% | 0.00% |
| smol500 | W0 | 88.03% | 82.40% | 85.12% | 11.20% | 17.60% | 0.00% |
| smol500 | W1 | 89.66% | 83.20% | 86.31% | 9.60% | 16.80% | 0.00% |
| smol500 | W2 | 95.57% | 77.60% | 85.65% | 3.60% | 22.40% | 0.00% |
| smol2video | W0 | 70.48% | 93.60% | 80.41% | 39.20% | 6.40% | 0.00% |
| smol2video | W1 | 79.12% | 94.00% | 85.92% | 24.80% | 6.00% | 0.00% |
| smol2video | W2 | 95.20% | 55.60% | 70.20% | 2.80% | 44.40% | 0.00% |

The [prediction log](results/predictions.jsonl) contains the exact response text and parsed answer. [Run metadata](results/run.json) records the fixed seeds, revisions, hardware and input hashes. [Main metrics and intervals](results/main.csv), [paired wording transitions](results/wording.csv), [gray-image ablation](results/ablation_effects.csv), [negative strata](results/negative_slices.csv), [annotation rescoring](results/label_rescoring.csv), and [gallery case IDs](results/gallery_cases.csv) are generated from the same complete log. To render the visual gallery using locally cached COCO images, run `python -m audit.gallery`.

## Implementation and credits

This project implements seeded probe selection, the shared evaluation harness, response parsing, paired statistics, gray-image ablation, paired label rescoring, and error selection. It reuses the POPE probing protocol and annotations, the RePOPE relabeling, COCO photographs and upstream licensing, and pretrained SmolVLM checkpoints with their model-specific Transformers processors. [Hugging Face Transformers](https://github.com/huggingface/transformers) implements model loading, vision preprocessing, and autoregressive decoding. [PyTorch](https://pytorch.org/) runs inference and [Matplotlib](https://matplotlib.org/) generates the plots.

Relevant papers and how they position the extension are in [RELATED_WORK.md](RELATED_WORK.md).

## AI declaration

OpenAI Codex assisted with literature discovery, protocol and code development, experiment execution, and drafting submission materials. Reported metrics come from logged model outputs.
