# 🧠 Reinforcement Learning for Adaptive Learning (RL_for_AL)

This project applies **Reinforcement Learning** (specifically **Proximal Policy Optimization - PPO**) to an **Adaptive Learning environment**. The RL agent learns to personalize educational activities for students with different learning styles and velocities.

---

### ⚙️ Setup with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency and virtual environment management.

### 📁 Project Strucutre
RL_for_AL/
│
├── data/ # Curriculum datasets (JSON)
├── environments/ # Custom Gym environment
│ └── AdaptiveLearningEnv.py
├── models/ # PPO agent implementation
│ ├── checkpoints/ # Saved model weights
│ └── ppo_model/
│ ├── ActorCritic.py
│ ├── PPO.py
│ └── utils.py
├── tests/ # Unit and integration tests
│ ├── init.py
│ └── test.py
├── training/ # Training configuration & scripts
│ ├── config.yaml
│ └── train.py
├── utils/ # Data loading, logging, plotting
│ ├── config_loader.py
│ ├── data_generator.py
│ ├── load_dataset.py
│ ├── logger.py
│ └── plot_rewards.py
├── logs/ # Training and testing logs (.csv)
├── pyproject.toml # Poetry configuration
├── poetry.lock
└── README.md # Project description

### 1. Clone the repository

```bash
git clone https://github.com/Herriot65/A-Reinforcement-Learning-Algorithm-for-Adaptive-Learning.git
cd RL_for_AL
````

### 2. Install dependencies

```bash
poetry install
```

### 3. Activate the virtual environment

```bash
poetry shell
```

---

## 🚀 How to Train the Agent

Run the PPO training loop using:

```bash
python training/train.py
```

Training logs will be saved to:

```
logs/training_logs_XXXX.csv
```

Model checkpoints are saved in:

```
models/checkpoints/ppo_model.pt
```

---

## 🧪 How to Test the Agent

Evaluate the trained agent on synthetic student profiles:

```bash
python scripts/test.py
```

This will:

* Generate random students (style + velocity)
* Run the trained PPO model
* Save results to: `logs/test_results.csv`

---

## 📊 Plotting Episode Rewards

To visualize the reward evolution during training:

```python
from utils.plot_rewards import plot_rewards_from_csv

plot_rewards_from_csv("logs/training_logs_XXXX.csv")
```

---

## 📦 Configuration

All training hyperparameters are stored in `config.yaml`:

```yaml
n_episodes: 5000
LR: 0.0003
GAMMA: 0.99
LAMDA: 0.95
EPOCHS: 10
CLIP_EPSILON: 0.2
entropy_coef: 0.01
minibatch_size: 64
freq_update: 2048
```

This file is loaded automatically during training and testing. You can modify these parameters.

---

## 📚 Dataset

Dataset follows this structure:

```json
{
  "lessons": {...},
  "activities": {...},
  "sprints": {...}
}
```

