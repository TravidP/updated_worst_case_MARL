

# 🚦 Distributionally Robust Multi-Agent Reinforcement Learning for Traffic Signal Control (DR-MARL)

This repository contains the official implementation of a **Distributionally Robust Multi-Agent Reinforcement Learning (DR-MARL)** framework for large-scale traffic signal control.

Unlike standard MARL approaches that optimize under nominal traffic conditions, this framework explicitly **optimizes for worst-case traffic scenarios** using an adversarial demand generator, significantly improving robustness under non-stationary and uncertain traffic patterns.

---

## 🧠 Key Idea

We formulate traffic signal control as a **two-timescale minimax optimization problem**:

* **MARL Controllers (Fast Timescale)**
  Learn decentralized policies to **minimize congestion** using local observations.

* **Contextual-Bandit Worst-Case Estimator (CB-WCE) (Slow Timescale)**
  Learns to **maximize network-wide congestion** by generating adversarial traffic demand mixtures.

This creates a **distributionally robust learning loop**:

[
\min_{\pi_{\theta}} \max_{w \in \Delta} \ \mathbb{E}[R(\tau) | w, \pi_{\theta}]
]

---

## 🏗️ System Architecture

```
+-----------------------------------+             +-----------------------------------+
|                                   |             |                                   |
|    MARL Traffic-Light Agents      |  Actions a  |       SUMO / Flow Environment     |
|    (PPO, IA2C, MA2C, IQL-LR)      |-----------> |       (Grid / Real Networks)      |
|                                   | <-----------|                                   |
+-----------------------------------+  Obs o, R r +-----------------------------------+
                                                                    ^  |
                                                      Mixed Demand  |  | Reward (Delay)
                                                                    |  v
                                                  +-----------------------------------+
                                                  |                                   |
                                                  | Contextual-Bandit Worst-Case      |
                                                  | Estimator (CB-WCE)                |
                                                  |                                   |
                                                  +-----------------------------------+
```

---

## ⚙️ Features

* ✅ **Algorithm-agnostic**: Works with PPO, IA2C, MA2C, IQL-LR
* ✅ **Adversarial demand generation** via contextual bandits
* ✅ **Two-timescale training** (fast control + slow adversary)
* ✅ **Scalable** from grid networks to real-world networks (e.g., Monaco)
* ✅ **Robustness-focused evaluation** (worst-case performance guarantees)

---

## 📂 Repository Structure

```
├── agents/                 # MARL algorithms and neural network models
│   ├── models.py
│   ├── policies.py
│   └── utils.py
│
├── config/                 # Training & evaluation configs
│   ├── config_ppo_large.ini
│   ├── config_ma2c_real.ini
│   └── ...
│
├── envs/                   # Environment wrappers (Flow/SUMO)
│   ├── env.py
│   └── adversarial_env.py  # CB-WCE integration
│
├── data_traffic/           # Traffic demand scenarios
│   ├── traffic_peak.csv
│   └── ...
│
├── large_grid/             # Synthetic 5×5 grid network
├── real_net/               # Real-world Monaco network
│
├── output_adversary/       # CB-WCE logs & checkpoints
├── output_result/          # MARL training logs & metrics
│
└── main.py                 # Training entry point
```

---

## 🧩 Methodology

### 1. Reward Design

To ensure stability in heterogeneous real-world networks, we use a **queue-based reward**:

[
r_{t}^{i} = -\sum_{l \in \mathcal{L}*{i}} queue*{t+\Delta t}[l]
]

```python
def compute_local_reward(self, intersection_id):
    incoming_lanes = self.get_incoming_lanes(intersection_id)
    queue_penalty = sum([self.get_queue_length(lane) for lane in incoming_lanes])
    return -queue_penalty
```

---

### 2. Two-Timescale Training Loop

```python
for episode in range(NUM_EPISODES):
    env.reset()
    s_tau = env.get_estimator_observation()

    for window in range(WINDOWS_PER_EPISODE):

        # 1. Adversarial demand generation
        w_tau = cb_wce_policy.sample(s_tau)
        mixed_demand = apply_mixed_demand(base_scenarios, w_tau)
        env.set_demand(mixed_demand)

        cb_reward = 0

        # 2. MARL control (fast timescale)
        for t in range(STEPS_PER_WINDOW):
            actions = {
                i: agents[i].act(env.get_local_obs(i))
                for i in env.intersections
            }

            rewards, next_obs = env.step(actions)
            store_marl_transitions(obs, actions, rewards, next_obs)

            cb_reward += env.get_network_total_waiting_time()

        # 3. Update adversary
        next_s_tau = env.get_estimator_observation()
        cb_wce_policy.update(s_tau, w_tau, cb_reward, next_s_tau)
        s_tau = next_s_tau

    # 4. Update MARL agents
    update_marl_agents()
```

---

## 📊 Experimental Design

### Benchmarks

* 🟦 Synthetic Grid: **3×3 → 5×5 scaling**
* 🟥 Real Network: **Monaco City**
* 🟨 Cross-validation: unseen demand scenarios (zero-shot)

---

### Algorithms Evaluated

* PPO (Primary)
* IA2C
* MA2C
* IQL-LR

---

### Metrics

* 📉 Average Queue Length
* 📉 Worst-Case Queue Length
* 📉 Network Delay
* 📊 Queue Variance (stability proxy)

---

## 📈 Robustness Objective

We explicitly optimize **worst-case performance**:

[
J_{worst}(\theta) = \min_{k \in \mathcal{K}} \mathbb{E}[R(\tau) | k, \pi_{\theta}]
]

---

## 🧪 Key Results (Monaco + PPO)

* 🚀 **74.39% reduction** in worst-case queue length
* 🚀 **75.45% improvement** in average network queue length

---

## 🔍 Practical Insights

* **Zero-shot generalization matters more than average reward**
* Avoid **state overloading** → hurts scalability
* Monitor **queue divergence** → early signal of instability
* Robust policies reduce **variance**, not just mean delay

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run Training

```bash
python main.py --config config/config_ppo_large.ini
```

---

### 3. Evaluate

```bash
python main.py --eval --model_path output_result/ppo_model
```

---

## 📌 Future Work

* Game-theoretic equilibrium analysis of MARL vs adversary
* Integration with **real-time traffic data streams**
* Scaling to **city-level digital twins**
* Safe RL constraints for deployment

---

## 📄 Citation

If you use this work, please cite:

```
@article{drmarl_traffic_control,
  title={Distributionally Robust Multi-Agent Reinforcement Learning for Traffic Signal Control},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first.

---

## ⭐ Acknowledgements

* SUMO / Flow simulation framework
* OpenAI Gym ecosystem
* MARL research community

---

If you want, I can next step help you:

* turn this into **NeurIPS / T-ITS style paper structure**
* or align README with **your exact repo code (function names, configs)**
* or even generate **figures (architecture diagram, training curve templates)**
