# DQN Library

## Support
- ExperienceReplay
- PrioritizedExperienceReplay
- multi-step bootstrap
- Double DQN (default)
- ε-greedy method
- Dueling DQN
- User costomized network and env

## Quick start
### 1. Config setting
```
info:
  name: Carpole_dqn
  module_name: cartpole_dqn
# Replay
replay:
  type: PrioritizedExperienceReplay  # ExperienceReplay or PrioritizedExperienceReplay
  capacity: 10000
# Train
train: 
  train_mode: true # if False, eval mode
  num_episode: 1000
  batch_size: 32
  target_update_iter: 20
  multi_step_bootstrap: true  # multi-step bootstrap
  num_multi_step_bootstrap: 5 # multi-step bootstrap
  gamma: 0.97      # ε-greedy
  eps_start: 0.9   # ε-greedy
  eps_end: 0.05    # ε-greedy
  eps_decay: 200   # ε-greedy
  render: False
  # Save
  save_iter: 200
  save_filename: cartpole
# Eval
eval:
  num_episode: 100
  filename: cartpole_1000.pth
  render: True
```

### 2. Run DQN
```
python dqn.py -f config.yaml
```

### 3. View Result
You can result by tensorboard
```
tensorboard --logdir results/20210914165555
```
result is created at results/YYYYMMDDHHMMSS/


## for Docker
```
docker-compose up -d
docker exec -it dqn_breakout bash
```

## tmux usage
```
tmux new-session -s <name>  
tmux ls
# atatch
tmux a -t <name>
# detatch
ctl + b -> d  
```