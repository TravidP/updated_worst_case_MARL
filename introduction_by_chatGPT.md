# 🛡️ Distributionally Robust MARL for Traffic Signal Control

> **Research Goal:** Transitioning from "Average-Case" optimality to "Worst-Case" robustness in traffic signal control.

This repository implements a **Distributionally Robust Multi-Agent Reinforcement Learning (DR-MARL)** framework. Unlike traditional methods (IA2C, MA2C) that optimize for average performance under nominal demand, this system trains a controller to remain stable under adversarial traffic perturbations. It utilizes a **Contextual-Bandit Worst-Case Estimator (CB-WCE)**  to identify and simulate the most challenging traffic distributions, forcing the traffic signal controller to learn robust policies.

---

## 📚 Theoretical Foundation

This project solves a **Minimax Game** between two agents:

1. **The Protagonist (Traffic Controller):** Minimizes the total network delay (or queue length).
2. **The Adversary (Worst-Case Estimator):** Maximizes the total network delay by dynamically mixing traffic demand patterns.

Mathematically, we seek the optimal policy  that satisfies:

* : The traffic signal control policy (e.g., MA2C).
* : The weight vector controlling the mixture of traffic scenarios (the "distribution").
* : The resulting traffic distribution (e.g., 30% Peak Morning + 70% Event Traffic).
* : The loss function (congestion/delay).

By training against the worst-case mixture , we ensure the controller performs reliably even when traffic conditions deviate significantly from the training mean.

---

## 🛠 GitHub Quick Start Guide

### 1. Prerequisites

* **OS:** Windows (via Git Bash), Linux (Ubuntu 18.04+), or macOS.
* **SUMO:** System-wide installation of SUMO (Simulation of Urban MObility) is required. Ensure `SUMO_HOME` is set in your environment variables.
* **Python:** 3.6 or 3.7 (TensorFlow 1.x requirement).

### 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/TravidP/updated_worst_case_MARL.git
cd updated_worst_case_MARL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify SUMO Installation
sumo --version
# Output should be version 1.0.0 or higher

```

### 3. Contribution Workflow

1. **Pull latest changes:** `git pull origin main`
2. **Stage your changes:** `git add .`
3. **Commit:** `git commit -m "Feat: Added new reward function for adversary"`
4. **Push:** `git push origin main`

---

# 📊 Traffic Distribution & Scenario Generation

Robustness is only as good as the diversity of the attacks. We define "attacks" as specific mixtures of traffic flows.

## 1. Traffic Data Structure

All traffic distributions are stored as **CSV files** in `./data_traffic/`. These serve as the "basis vectors" for our adversary.

* **Format:** `origin_edge, dest_edge, veh_per_hour`
* **Logic:** The Adversary outputs a weight vector . The actual traffic injected into the simulation is a linear combination:



## 2. 5x5 Synthetic Grid

Located in `./data_traffic/`, this dataset includes:

* `traffic_uniform_low.csv`: Low density, balanced traffic.
* `traffic_west_east_peak.csv`: High stress on horizontal arteries.
* `traffic_north_south_peak.csv`: High stress on vertical arteries.

**Debugging:**
Use `./data_traffic/generate_traffic.py` to create new basis scenarios if the current ones are too easy for the agent.

## 3. Monaco Real-World Network (Development)

Located in `./data_traffic_real/`.

* 
**Methodology:** Based on the *PNEUMA* dataset, we cluster the city into regions.


* **Regional Pairs:** Traffic is generated between specific region pairs (e.g., Port -> City Center) to simulate realistic commute waves.
* **Hard-coded Routes:** Unlike the grid, real-world drivers follow specific paths. These are pre-defined in `.rou.xml` templates to ensure topological consistency.

---

# 🚦 5x5 Large Grid: The Experimental Pipeline

To achieve the "Distributionally Robust" controller described in the conference paper, you must follow this 3-Stage Training Pipeline.

## 🔎 Debugging & Visualization

To watch the simulation in real-time:

1. Open `./envs/env.py`.
2. Set `gui=True` in `self._init_sim(seed, gui=True)`.
3. **Warning:** Always set `gui=False` for long training runs to avoid memory leaks.

---

## Phase I: Baseline Training (Nominal Policy)

First, we train a standard IA2C or MA2C agent on a standard, average traffic distribution. This gives us a decent "Nominal Policy".

**Command:**

```bash
python main.py --base-dir ./runs/ia2c_large --config-dir ./config/config_ia2c_large.ini --test-mode no_test train

```

* **Output:** Checkpoints saved in `./runs/ia2c_large/model/`.
* **Key Configs (`config_ia2c_large.ini`):**
* `total_step`: `1e6` (1 million steps).
* `batch_size`: `120` (trajectory length before update).



---

## Phase II: Training the Adversary (The "Worst Estimator")

Now we freeze the Phase I controller and train the **Adversary**. The adversary observes the global traffic state (waves, waits) and selects traffic weights  to **maximize** congestion.

**File:** `train_adversary.py` 
**Logic:**

1. **Frozen Victim:** The script loads the model from `FROZEN_MODEL_DIR`.
2. **Adversary Agent:** A CNN-based A2C agent that acts every 600s (10 mins).
3. **Action:** Continuous vector representing weights for flow groups.
4. **Reward:**  (Zero-sum game structure).

**Configuration (Inside `train_adversary.py`):**

```python
FROZEN_CONFIG_PATH = './config/config_iqll_large.ini'
FROZEN_MODEL_DIR = './runs/iqll_large' 
base_dir = './output_adversary/ia2c_large'

```

**Run:**

```bash
python train_adversary.py

```

---

## Phase III: Robust Co-evolution (DR-MARL)

Finally, we retrain the traffic controller. This time, instead of random traffic, the environment uses the **Adversary** trained in Phase II to generate traffic. This forces the controller to fix its weak spots.

**File:** `train_coevolution.py`

* **Input:** Requires both the `TRAFFIC_CHECKPOINT_DIR` (Phase I) and `ADVERSARY_CHECKPOINT_DIR` (Phase II).
* **Process:**
1. Adversary selects difficult traffic weights.
2. Controller runs for 10 minutes on this difficult traffic.
3. Controller updates its policy (PPO/A2C) to handle this stress.
4. Repeat.



**Run:**

```bash
python train_coevolution.py

```

---

# 🏙️ Monaco Real-World Network: Training Guide

The pipeline is identical to the 5x5 Grid but uses Graph Neural Network (GCN) architecture (if configured) and real-world topology.

## 1. Baseline

```bash
python main.py --base-dir ./runs/ia2c_real train --config-dir ./config/config_ia2c_real.ini --test-mode no_test

```

## 2. Adversarial Training

Use `train_adversary_real.py`. This script adapts the state space to the 30-intersection Monaco map.

* **Crucial:** Ensure `config_ia2c_real.ini` points to the correct `./data_traffic_real/` folder.

## 3. Co-evolution

Use `train_coevolution_real.py`.

* **Note on Hard-coded Routes:** Since Monaco routes are fixed, the adversary effectively chooses "Which OD pairs are active?" rather than generating random flows.

---

# 📈 Monitoring & Metrics

Use TensorBoard to validate your experiments.

```bash
tensorboard --logdir=./runs

```

**Key Metrics to Watch:**

1. **`Reward` (Controller):** Should increase (less negative) over time.
2. **`Total_Congestion_Reward` (Adversary):** In Phase II, this should *increase* (finding worse traffic). In Phase III, this should eventually *decrease* as the controller becomes robust.
3. 
**`Queue Length Variance`:** A robust controller will have lower variance in queue length across different episodes compared to the baseline.



---

# 📂 Code Structure Overview

* **`agents/`**: Contains the RL brains.
* `models.py`: Implementation of A2C, IA2C, MA2C, and the Adversary (CNN/GCN).
* 
`policies.py`: LSTM and Fully Connected layers definition.




* **`envs/`**: The simulation environments.
* `large_grid_env.py`: Standard 5x5 grid.
* `adversarial_large_grid_env.py`: Modified environment that accepts `weight` vectors to mix traffic CSVs.


* **`config/`**: Hyperparameters (`.ini` files).
* **`data_traffic/`**: The "Arsenal" of traffic scenarios for the adversary.

---

### Citation

If you use this code for your research, please cite the original papers:

1. *Multi-Agent Deep Reinforcement Learning for Large-scale Traffic Signal Control*, IEEE T-ITS 2019.
2. *Distributionally Robust Multi-Agent Reinforcement Learning for Intelligent Traffic Control*, 2025.
