# Related work and contribution

| Paper | What it supplies | How P44 uses or extends it |
|---|---|---|
| [Lin et al., Microsoft COCO, ECCV 2014](https://arxiv.org/abs/1405.0312) | Natural images with common annotated object categories. | The audit uses COCO val2014 and inherits its domain limits. |
| [Rohrbach et al., CHAIR, EMNLP 2018](https://aclanthology.org/D18-1437/) | Measures hallucinated object mentions in free-form captions. | This audit uses closed questions with a fixed queried target and separates false yes answers from misses. |
| [Li et al., POPE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.20/) | Balanced yes/no probing, three negative strategies, and aggregate prompt stability on four templates for an early LLaVA model in Table 4. | Uses the probing protocol and records which individual answers change with phrasing, in addition to aggregate metrics. |
| [Neuhaus and Hein, RePOPE, 2025](https://arxiv.org/abs/2504.15707) | Corrects annotation mistakes, removes ambiguous cases, and shows that rankings can change after relabeling. | Freezes one common subset before inference, then scores the same predictions with both label versions. |
| [Augustin et al., DASH, ICCV 2025](https://arxiv.org/abs/2503.23573) | Studies systematic false positives and whether they persist across prompt formulations. | The balanced positive/negative audit measures whether corrected and introduced errors cancel and separates false yes changes from misses. |
| [He et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.349/) | Uses counterfactual object removal to build more relevant negatives. | Reuses natural images with independently revised labels. A uniform-gray control removes scene information in a separate ablation. |
| [Marafioti et al., SmolVLM, 2025](https://arxiv.org/abs/2504.05299) | Describes a family of compact still-image and video-language models. | Provides the three checkpoints selected for a limited-memory course setup. |
| [When Irrelevant Text Matters, 2026](https://arxiv.org/abs/2608.19208) | Tests how irrelevant added text shifts yes/no decision margins and includes template controls. | P44 uses short meaning-preserving variants and counts directional probe changes on a corrected-label subset. |

The extension is an empirical analysis of the same probes under different wordings and both label versions. It asks whether a stable aggregate score hides individual errors that switch in both directions, and whether the interpretation holds after label correction. Any ranking applies to these checkpoints, templates, and sampled probes. The negative strategies use different sampled images and object distributions, so their ranking is descriptive.
