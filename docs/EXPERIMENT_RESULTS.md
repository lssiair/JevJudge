# Jev as a GRPO reward: experiment results

This report summarizes the runs that actually produced held-out evaluations. GRPO is the policy optimization method in every trained arm; Jev is one possible reward source. The public, machine-readable aggregate is [public_results/metrics.json](../public_results/metrics.json). It contains no dataset prompts, model completions, API payloads or credentials.

## Experimental design

All arms start from the same local Qwen3.5-4B text backbone. The main runs use seed 42 on 8 × A100 80 GB GPUs: GPU 0 serves vLLM, and GPUs 1–7 train with TRL GRPO. The reference and Jev arms within each task use matched sampling and optimizer settings. A run's training steps and checkpoint status are stated below; not every cross-run comparison has equal training budget.

| Task | Reference reward | Jev reward | Held-out evaluator |
|---|---|---|---|
| GSM8K | Exact final numerical answer | Jev Noul judges the candidate **with the reference final answer supplied** | Local exact numerical answer parser |
| ToolRL tool calling | Local format plus graded tool-name/parameter match against ground truth | Local format plus scaled Jev Noul judgment **with parsed reference calls supplied** | Local structural exact call match |
| ToolRL no-answer ablation | Same ToolRL reference arm as control | Reference-independent local format plus scaled Jev Noul judgment using only dialogue, tool definitions and candidate response | Same local structural exact call match |

The ToolRL Jev reward uses `format + 6 × probability − 3` for call-required cases; the reference-conditioned arm gives no-call reference cases only the format reward. The no-answer arm scores *all* candidates, including no-call cases, with Jev and uses a format check independent of the ground truth. Ground-truth calls remain in local diagnostic logs and held-out evaluation; they are not sent to Jev or used to form the no-answer training reward.

## Jev reward validation

On the HelpSteer2 validation split, 1,038 responses from 519 prompts were scored without sending human labels to Jev. The returned model was `jev-1.13.0`. Scores and labels were normalized to [0, 1]; human ties were excluded from pairwise comparisons.

| Dimension | Pearson | Spearman | MAE | Same-prompt pairwise agreement (95% bootstrap CI) | Pairs |
|---|---:|---:|---:|---:|---:|
| Helpfulness | 0.456 | 0.391 | 0.251 | 74.5% (70.0–79.2%) | 373 |
| Correctness | 0.451 | 0.391 | 0.251 | 67.7% (62.6–72.5%) | 354 |
| Coherence | 0.428 | 0.315 | 0.138 | 77.0% (71.2–82.7%) | 196 |

This is evidence of a useful but noisy ranking signal, not a calibrated estimate of correctness. The [validation overview](../public_results/plots/reward_validation/overview.png) and [pairwise plot](../public_results/plots/reward_validation/pairwise.png) visualize these aggregates. The separate tool-calling Jev sanity check used only three handcrafted cases (correct call, wrong tool, wrong argument) and is not a reliability study for that task.

## GSM8K: two-step integration smoke

Before the pilot, the pipeline completed two optimization steps per trained arm (56 rollouts each) and evaluated the first 16 GSM8K test questions: Base 11/16 (68.75%), Reference 12/16 (75.00%), Jev 12/16 (75.00%). These rows verify training, checkpoint reload and evaluation wiring only. They are **not an efficacy comparison** and are not independent of the later full-test evaluation.

## GSM8K: 144-step pilot

All three models were evaluated on the same 1,319 GSM8K test questions with greedy decoding and a 512-token completion limit. The base model received no GRPO training. Both trained arms completed 144 steps.

| Model | Correct | Accuracy | 95% bootstrap CI | Invalid final answer |
|---|---:|---:|---:|---:|
| Base | 1,127/1,319 | 85.44% | 83.47–87.34% | 8.87% |
| GRPO-Reference | 1,147/1,319 | 86.96% | 84.99–88.78% | 6.52% |
| GRPO-Jev | 1,152/1,319 | 87.34% | 85.52–89.16% | 6.22% |

Jev minus Reference is +0.38 percentage points; the paired bootstrap 95% CI is **−0.53 to +1.29 points**. The interval includes zero, so this one-seed pilot does not establish a benefit from replacing the reference reward with Jev. The original [pilot accuracy plot](../public_results/plots/accuracy_vs_training_step.png) is supplied for context; online rollout metrics are not held-out accuracy.

## GSM8K: expanded runs

The same 1,319-question GSM8K test was evaluated with greedy decoding and a 512-token limit.

| Model | Evaluated step | Status | Correct | Accuracy | 95% bootstrap CI | Truncated |
|---|---:|---|---:|---:|---:|---:|
| GRPO-Reference | 1,602 | Completed | 1,203/1,319 | 91.21% | 89.61–92.72% | 55 |
| GRPO-Jev | 1,300 | Checkpoint; run later failed with HTTP 402 | 1,186/1,319 | 89.92% | 88.25–91.51% | 81 |

The observed Jev-minus-Reference difference is −1.29 points (paired bootstrap 95% CI −2.50 to −0.08 points). **Training budgets differ by 302 steps**, so this comparison cannot isolate the effect of the reward source. The Jev run continued beyond the evaluated checkpoint and stopped after step 1,344; the measured model is explicitly the saved step-1,300 checkpoint.

## ToolRL tool calling

ToolRL `rlla_4k` supplied 3,920 training rows; 21 prompts exceeded the 2,048-token prompt limit, leaving 3,899 eligible. Its test file has 80 rows; one was excluded by the same limit. The 79 evaluated questions include 71 requiring calls and 8 requiring no call. Evaluation uses greedy decoding, seed 42 and at most 1,024 output tokens. It parses call names and JSON parameters, ignores call order, and **does not execute tools**. This is **not the official BFCL-v3 benchmark**.

| Model | Evaluated step | Exact calls | Call-required exact | Format | Correct no-call abstentions |
|---|---:|---:|---:|---:|---:|
| GRPO-Reference | 1,300 | 47/79 (59.49%) | 43/71 (60.56%) | 75/79 (94.94%) | 6/8 |
| GRPO-Jev with reference calls | 1,300 | 44/79 (55.70%) | 40/71 (56.34%) | 75/79 (94.94%) | 4/8 |
| GRPO-Jev without reference calls | 1,250 | 43/79 (54.43%) | 39/71 (54.93%) | 74/79 (93.67%) | 7/8 |

At matched 1,300 steps, reference-conditioned Jev trails Reference by 3/79 questions (−3.80 points). The paired bootstrap 95% CI is −8.86 to 0.00 points, and exact McNemar p = 0.25. Jev had no uniquely correct questions; Reference had three. The no-answer arm's 1,250-step checkpoint was evaluated after its training stopped at step 1,286 with API HTTP 402. Its stronger no-call abstention count is based on only **eight** examples and does not validate the semantic quality of clarification text. Its training budget is also 50 steps shorter, so its 43/79 should not be treated as a matched-budget comparison.

An intermediate reference-conditioned Jev checkpoint at step 850 scored 45/79 (56.96%). It is retained as a historical point, not an additional independent run.

## Planned but not reported as completed

Hybrid reward and Jev Best-of-N have code paths, but no completed held-out result is included here. A three-seed main study was not run. The BFCL/gorilla code was downloaded for potential benchmarking but no official BFCL-v3 evaluation was executed. The local ToolRL result must not be labeled BFCL accuracy.

## What these experiments support

The pilot shows Jev can provide a trainable reward signal. The measured GSM8K pilot difference is inconclusive. The expanded GSM8K comparison favors Reference at the evaluated checkpoints but has unequal training steps. On this small local ToolRL test, Reference scores higher in exact call matching at matched 1,300 steps; the no-answer ablation has not completed its 1,300-step target. None of these numbers is a BFCL-v3 score, and no three-seed main experiment is reported.

Only one training seed is available for these comparisons. Example-level bootstrap intervals do not capture variation from training seeds. The strict GSM8K final-answer parser and 512-token limit can mark a mathematically correct but truncated response invalid. Jev's returned model version, rubric and API availability may change. The ToolRL test's 79 rows are too small for strong claims about broad tool-use capability.

## Provenance and reproduction

- [Aggregate JSON](../public_results/metrics.json) is generated from the private experiment outputs by `python scripts/export_public_results.py`.
- [Training configurations](../configs/) and [offline ToolRL evaluator](../src/evaluate_tools.py) are included in the repository. The public code uses relative default paths or environment variables; historical cluster path strings were omitted without changing the measured training settings.
- Datasets and model weights are **not redistributed**. Obtain [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), [ToolRL](https://github.com/qiancheng0/ToolRL), and the Qwen3.5-4B model from their respective publishers, observing their licenses.
- The 8-GPU training topology and exact package pins are documented in [README.md](../README.md), [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock). The public aggregate does not include raw prompts, completions, API responses, model checkpoints or credentials.
- This report reflects saved evaluation artifacts, not a new evaluation. The public commit freezes the release code, but the exact externally downloaded model and dataset revisions are still needed for byte-for-byte reproducibility.
