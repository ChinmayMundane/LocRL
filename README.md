# When to Seek Landmarks?

Learning continuous robot control with sparse landmark localization. Includes
CAPQL, preference-conditioned SAC (PC-SAC), fixed-weight SAC, and three RRT* planners.

## Setup

Python 3.10, tested on Linux. Run commands from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required MORL-Baselines modules are included in `morl_import/`.


## Download all weights and evaluate

Download the [50 trained models (about 473 MB)](https://owncloud.ut.ee/owncloud/s/Aa8n3TsLmkAG32f).
Extract the download and place the **contents of `model_logs/` inside `models/`**,
keeping the downloaded folder names. For example:
`models/capql_recovery40_150k_seed_11/final_model.tar`.

Rerun the 200-map evaluation across all five seeds and configured preferences:

```bash
python experiments/run_conditioned.py final-test --execute-final-test
python experiments/run_conditioned.py final-test \
  --config configs/experiments/pc_sac.yaml --execute-final-test
python experiments/run_fixed_sac_preferences.py final-test --execute-final-test
python experiments/run_conditioned.py final-test \
  --config configs/experiments/ablation_no_landmark.yaml --execute-final-test
```

No retraining is needed. New evaluations go to `outputs/`; the table commands
below read the bundled paper records in `results/`.

## Train

```bash
# CAPQL
python experiments/run_conditioned.py train --seeds 11

# PC-SAC
python experiments/run_conditioned.py train \
  --config configs/experiments/pc_sac.yaml --seeds 11

# Fixed SAC: one policy per preference
python experiments/run_fixed_sac_preferences.py train --seeds 11
```

Replace `train` with `validate` to evaluate trained models on validation maps.
Omit `--seeds 11` to use all five seeds. Each model trains for 150,000 steps.
New models go to `models/`; TensorBoard logs go to `logs/`.

## Paper results

Results in the paper can be viewed by:

```bash
python experiments/summarize_final_study.py
python experiments/analyze_zero_contact_results.py
python experiments/plot_results.py --show
python experiments/summarize_final_study.py --table 3
python experiments/summarize_final_study.py --table 4
```

## Repository structure

```text
environment/    # Simulator, EKF, observations and rewards
capql/          # Preference-conditioned vector-critic controller
pc_sac/         # Preference-conditioned scalar SAC
fixed_sac/      # Fixed-weight SAC
planners/       # Dubins, landmark-aware and belief-space RRT*
configs/        # Environment defaults and experiment settings
experiments/    # Training/evaluation runners, tables and plots
data/           # Training, validation and test maps
models/         # Place downloaded weights here
results/        # Saved paper evaluations
outputs/        # New evaluations, reports and plots (Git-ignored)
morl_import/    # Required MORL-Baselines modules and license
```

Hardware integration code is not included yet.

Built on [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines) and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
