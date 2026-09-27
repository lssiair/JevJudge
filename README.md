# Jev as a reward source for GRPO

This repository compares **GRPO with a deterministic reference reward** against **GRPO with Jev-derived rewards** on GSM8K and ToolRL tool calling. GRPO is the optimizer in every trained arm; Jev is a reward source, not an alternative optimizer.

The complete [experiment report](docs/EXPERIMENT_RESULTS.md), [中文结果汇总](docs/EXPERIMENT_RESULTS.zh-CN.md) and [aggregate metrics](public_results/metrics.json) cover the actual runs, including interrupted runs and unequal training budgets. Raw dataset rows, generated responses, API payloads and model checkpoints are kept out of the public result bundle.

## Results at a glance

| Evaluation | Reference | Jev with reference | Jev without reference |
|---|---:|---:|---:|
| GSM8K pilot, both 144 steps, 1,319 test questions | 86.96% | 87.34% | — |
| GSM8K expanded, Reference 1,602 vs Jev 1,300 steps | 91.21% | 89.92% | — |
| ToolRL local test, Reference/Jev 1,300 steps, 79 questions | 59.49% | 55.70% | 54.43% at step 1,250 |

The pilot Jev-minus-Reference paired 95% interval includes zero. The expanded GSM8K arms have **unequal training budgets**. The no-answer ToolRL arm stopped after step 1,286 due to API HTTP 402 and was evaluated at its saved step-1,250 checkpoint. ToolRL results use a local structural exact-call evaluator; they are **not BFCL-v3 results**. See the report for confidence intervals, format accuracy, no-call cases and limitations.

## What the rewards see

- **GSM8K Reference:** exact final numerical answer.
- **GSM8K Jev:** the problem, candidate response **and reference final answer** are sent to Jev; its Noul correctness value is the reward.
- **ToolRL Reference:** local format plus tool-name/parameter similarity against the ground truth.
- **ToolRL Jev:** Jev sees the dialogue, available tools, candidate response **and parsed reference calls**. A local format term is added to its scaled score.
- **ToolRL Jev without reference:** Jev sees only the dialogue, available tools and candidate response. The local format term does not consult the reference. All candidates, including no-call responses, receive a Jev score. References are retained locally only for diagnostics and held-out evaluation.

The original ToolRL training set has 3,920 rows; 21 overlength prompts are excluded. The local test evaluates 79 of 80 rows after the same 2,048-token prompt filter. Calls are parsed but not executed.

## Repository map

- `configs/`: training configurations for pilot, expanded GSM8K and ToolRL arms.
- `src/rewards/`: reference, Jev and no-answer reward definitions.
- `src/evaluate_tools.py`: offline ToolRL structural evaluator.
- `scripts/train_resumable.sh`: one-arm launcher with full checkpoints.
- `docs/EXPERIMENT_RESULTS.md`: interpreted results and comparisons.
- `public_results/metrics.json`: aggregate-only, machine-readable release.
- `scripts/export_public_results.py`: regenerate aggregate metrics from the local `results/` directory.

## Data and environment

Obtain model weights and datasets from their original publishers and comply with their licenses:

- [Qwen3.5-4B](https://huggingface.co/Qwen)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [ToolRL](https://github.com/qiancheng0/ToolRL), directory `dataset/rlla_4k`

The original runs used 8 × A100 80 GB, Python 3.12 and the versions pinned in `pyproject.toml` / `uv.lock`. The project used GPU 0 for vLLM and GPUs 1–7 for distributed training. Reproducing a full run requires substantial GPU time and Jev API usage. The repository does not distribute datasets, weights or API credentials.

Create an environment compatible with your CUDA installation, for example with `uv sync --locked`. Export paths before running scripts:

```bash
export QWEN_SOURCE_PATH=/path/to/Qwen3.5-4B
export MODEL_PATH=/path/to/qwen3.5-4b-text
export GSM8K_PATH=/path/to/gsm8k
export HELPSTEER2_PATH=/path/to/HelpSteer2
export TOOLRL_PATH=/path/to/ToolRL/dataset/rlla_4k
export JEV_TMPDIR=/path/to/large/temp-directory
export TYPESAFE_KEY_FILE=/path/to/secure/typesafe-key-file
```

The release uses repository-relative defaults; the variables above override them. Keep credentials outside the repository. To export and verify the text backbone, run `python -m src.prepare_model`; then use `bash scripts/check_env.sh`. `scripts/check_env.sh` checks the Jev credential but makes no scoring request.

## Run an experiment

For Jev arms, first run the HelpSteer2 validation required by `src.train_grpo`:

```bash
python -m src.reward_validation
```

Each launcher starts its own local vLLM server. Use a fresh output directory per run, reserve an API request budget appropriate to the run, and keep GPU 0–7 available:

```bash
bash scripts/train_resumable.sh \
  --config configs/grpo_tool_reference.yaml \
  --output outputs/tool-reference-new --save-steps 50

JEV_MAX_REQUESTS=200000 JEV_BUDGET_ID=tool-jev-new \
bash scripts/train_resumable.sh \
  --config configs/grpo_tool_jev.yaml \
  --output outputs/tool-jev-new --save-steps 50

JEV_MAX_REQUESTS=200000 JEV_BUDGET_ID=tool-jev-no-answer-new \
bash scripts/train_resumable.sh \
  --config configs/grpo_tool_jev_no_answer.yaml \
  --output outputs/tool-jev-no-answer-new --save-steps 50
```

The configured `max_steps` is the total target, not an additional number of steps. To resume, pass `--resume outputs/previous-run/checkpoint-N` and choose a **new** `--output` directory. A full checkpoint includes model, optimizer, scheduler and seven RNG states. A failed Jev request stops training rather than fabricating a reward.

The ToolRL evaluator does not call Jev:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate_tools \
  --model outputs/tool-reference-new/final \
  --name tool-reference-new --training-step 1300
```

For the saved results in this repository, read [the report](docs/EXPERIMENT_RESULTS.md) rather than re-running costly training. To refresh the public aggregate after local evaluations: `python scripts/export_public_results.py`.

## Publication scope

`public_results/` contains aggregate metrics and plots. The working `results/`, `outputs/` and `cache/` directories may contain dataset text, model responses, API data, large checkpoints or credentials; they are excluded from Git. The result report records which arms completed, which stopped on HTTP 402, and where training steps differ. Exact reproduction also requires the same externally downloaded model and dataset revisions.

## Acknowledgments

Thanks to [wllzhang](https://github.com/wllzhang) for supporting the Jev API credits used in these experiments.
