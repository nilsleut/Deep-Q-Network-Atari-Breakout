# Deep Q-Network — Atari Breakout

Paper-faithful implementation of DQN (Mnih et al. 2015) trained on Atari Breakout, achieving ~17 mean reward after 5M frames.

## Overview

Faithful reproduction of the original DQN paper, including all implementation details that matter for stable training: frame stacking, replay buffer, target network, reward clipping, and epsilon-greedy annealing.

## Architecture

```
4 stacked grayscale frames (84×84×4)
        ↓
Conv2d(4 → 32, kernel=8, stride=4)  + ReLU
        ↓
Conv2d(32 → 64, kernel=4, stride=2) + ReLU
        ↓
Conv2d(64 → 64, kernel=3, stride=1) + ReLU
        ↓
Linear(3136 → 512) + ReLU
        ↓
Linear(512 → n_actions)   [Q-values]
```

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| Replay buffer | 1M transitions, uniform sampling |
| Target network | Hard update every 10k steps |
| Frame stack | 4 consecutive grayscale frames |
| Reward clipping | [-1, +1] |
| ε-greedy | 1.0 → 0.1 over 1M steps, then 0.01 |
| Optimizer | RMSprop (lr=2.5e-4, momentum=0.95) |
| Loss | Huber loss on TD error |
| Training frames | 5M |

## Result

~17 mean episode reward after 5M frames on Breakout. The original paper reports ~168 after 200M frames — the gap is expected given the reduced training budget.

<img width="1112" height="236" alt="plot_frame5000000" src="https://github.com/user-attachments/assets/bb3c3f96-f1d1-451d-9b6b-ca955bc3aa5d" />

## Setup

```bash
pip install torch gymnasium[atari] ale-py opencv-python numpy matplotlib
# Install ROMs
ale-import-roms /path/to/roms
python DQN_Breakout_Lokal.py
```

A Colab version (`DQN_Breakout_Colab.ipynb`) is included for GPU training.

## Reference

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518, 529–533.
