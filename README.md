# Fish Singulation via Deep Reinforcement Learning

Companion code for the Master's thesis:

> **[Thesis title]**
> [Author name], [University], [Year]

This repository contains the Isaac Lab environment and PPO training configurations used to train a UR10e robot arm to push fish from an infeed conveyor to an outfeed conveyor in simulation. The primary result is a single-fish push policy (Stage S3a) that achieves 96.8% of the theoretical delivery ceiling after approximately 765,000 training steps.

---

## Requirements

| Component | Version |
|-----------|---------|
| NVIDIA Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 |
| SKRL | 1.4.3 |
| PyTorch | CUDA-enabled build |
| Python | 3.10+ |
| GPU | CUDA-capable; tested on NVIDIA RTX 5070 Ti |

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) before proceeding.

---

## Installation

Clone this repository, then install the `Reach` extension in editable mode:

```bash
git clone https://github.com/<your-username>/fish-singulation-rl.git
cd fish-singulation-rl
pip install -e source/Reach
```

Verify that the environments are registered:

```bash
python scripts/list_envs.py
```

You should see `UR10e-Fish3-S1-v0`, `UR10e-Fish3-S3a-v0`, and the other fish singulation environments listed.

---

## Curriculum Overview

The training follows a four-stage curriculum. Each stage is initialised from the best checkpoint of the preceding stage.

| Stage | Environment ID | Fish | Gripper | Episode | Trained |
|-------|---------------|------|---------|---------|---------|
| S1 | `UR10e-Fish3-S1-v0` | 3 (approach only) | Open (fixed) | 8 s | Yes |
| S2 | `UR10e-Fish3-S2-v0` | 1 (pick-place) | Enabled | 15 s | No — push-only sufficient |
| S3a | `UR10e-Fish3-S3a-v0` | 1 (push) | Disabled | 10 s | **Yes — primary result** |
| Grasp | `UR10e-Fish3-Grasp-v0` | 1 (pick-place) | Enabled | 15 s | No |
| S3b | `UR10e-Fish3-S3b-v0` | 3 (push) | Disabled | 20 s | Yes (failed to converge) |
| S3ab | `UR10e-Fish3-S3ab-v0` | 2 (push) | Disabled | 15 s | Yes (failed to converge) |

---

## Training

All training uses 4096 parallel environments. Adjust `--num_envs` to fit your GPU memory.

**Stage S1** (fresh start):

```bash
python scripts/skrl/train.py --task UR10e-Fish3-S1-v0 --num_envs 4096
```

**Stage S3a** (transfer from S1 best checkpoint):

```bash
python scripts/skrl/train.py \
  --task UR10e-Fish3-S3a-v0 \
  --num_envs 4096 \
  --checkpoint logs/skrl/fish3_s1/<run_id>/checkpoints/best_agent.pt
```

**Stage S3b** (transfer from S3a best checkpoint):

```bash
python scripts/skrl/train.py \
  --task UR10e-Fish3-S3b-v0 \
  --num_envs 4096 \
  --checkpoint logs/skrl/fish3_s3a/<run_id>/checkpoints/best_agent.pt
```

**Stage S3ab** (two-fish intermediate; transfer from S3a best checkpoint):

```bash
python scripts/skrl/train.py \
  --task UR10e-Fish3-S3ab-v0 \
  --num_envs 4096 \
  --checkpoint logs/skrl/fish3_s3a/<run_id>/checkpoints/best_agent.pt
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/skrl/
```

---

## Evaluation (Play Mode)

Play mode uses 4 parallel environments with observation noise disabled and fixed mid-range physics.

```bash
python scripts/skrl/play.py \
  --task UR10e-Fish3-S3a-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/skrl/fish3_s3a/<run_id>/checkpoints/best_agent.pt
```

---

## Pretrained Checkpoint

The best S3a checkpoint (run `2026-05-27_12-04-59`, 765K steps, delivery = 0.387) is available as a release asset:

**[Download best_agent.pt](https://github.com/<your-username>/fish-singulation-rl/releases/tag/v1.0)**

Place it at `logs/skrl/fish3_s3a/2026-05-27_12-04-59/checkpoints/best_agent.pt` to use the exact path shown in the evaluation command above, or pass the path explicitly via `--checkpoint`.

---

## Repository Structure

```
fish-singulation-rl/
├── scripts/
│   └── skrl/
│       ├── train.py          # Training entry point
│       └── play.py           # Evaluation entry point
└── source/
    └── Reach/
        ├── setup.py
        └── Reach/
            ├── fish_rigid.usd            # Fish rigid-body proxy asset
            ├── UR-with-SuccGripper.usd   # UR10e robot asset
            └── tasks/manager_based/reach/
                ├── reach_env_cfg_fish3.py    # All environment configurations
                ├── ur_gripper_new.py         # Robot articulation config
                ├── mdp/
                │   ├── rewards_fish3.py
                │   ├── observations_fish3.py
                │   ├── terminations_fish3.py
                │   ├── events_fish3.py
                │   ├── actions_fish3.py
                │   └── events_conveyor.py
                └── agents/
                    ├── skrl_ppo_cfg_fish3_s1.yaml
                    ├── skrl_ppo_cfg_fish3_s3a.yaml
                    ├── skrl_ppo_cfg_fish3_s3b.yaml
                    ├── skrl_ppo_cfg_fish3_grasp.yaml
                    └── skrl_ppo_cfg_fish3_s3ab.yaml
```

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{[citekey],
  author  = {[Author name]},
  title   = {[Thesis title]},
  school  = {[University]},
  year    = {[Year]},
}
```

---

## License

BSD 3-Clause (inherited from the Isaac Lab extension template). See `source/Reach/pyproject.toml`.
