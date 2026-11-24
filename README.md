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

See [main table](results/main.md) after the main run finishes. `predictions.jsonl` is the raw prediction log. The aggregate output includes wording transitions, negative strata, paired real-vs-gray ablation, rescoring under original annotations, timing, and gallery probe IDs. `audit/reporting.py` validates log completeness before computing results. No dataset images, weights, or environment cache belong in Git.

## Implementation and credits

This project implements seeded probe selection, the shared evaluation harness, response parsing, paired statistics, gray-image ablation, paired label rescoring, and error selection. It reuses the POPE probing protocol and annotations, the RePOPE relabeling, COCO photographs and upstream licensing, and pretrained SmolVLM checkpoints with their model-specific Transformers processors. [Hugging Face Transformers](https://github.com/huggingface/transformers) implements model loading, vision preprocessing, and autoregressive decoding. [PyTorch](https://pytorch.org/) runs inference and [Matplotlib](https://matplotlib.org/) generates the plots.

Relevant papers and how they position the extension are in [RELATED_WORK.md](RELATED_WORK.md).

## AI declaration

OpenAI Codex assisted with literature discovery, protocol and code development, experiment execution, and drafting submission materials. Reported metrics come from logged model outputs.
