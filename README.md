# 🎓 Adaptive Learning Path Recommendation using Reinforcement Learning

This project explores the use of **Reinforcement Learning (RL)** to build an intelligent agent capable of recommending the most appropriate learning activities for a virtual student. The agent interacts with a simulated environment containing a curriculum of lessons, pedagogical constraints, and activities aligned with different cognitive learning styles (VARK).

-----

## 🚀 Project Overview

  - The problem is modeled as a **Markov Decision Process (MDP)**.
  - A student is simulated with a given **learning style profile (VARK)** and **velocity** (assimilation speed).
  - An RL agent learns to propose the right activity at each time step, based on the student's **current mastery state**.
  - The goal is to **maximize lesson mastery** using the **fewest and most suitable activities**.

-----

## 📂 Project Structure

```
RL_FOR_ADL/
├── data/                       # JSON files (curriculum, activities, etc.)
├── envs/                       # Custom Gymnasium environment definitions
│   ├── AdaptiveLearningEnv.py
│   ├── LessonMasteryTracker.py
│   └── Student.py
├── logs/                       # TensorBoard logs generated during training
├── models/                     # Saved trained RL agent models
├── tests/                      # Scripts and utilities for model testing and evaluation
│   ├── model_test_results/     # Directory for output of model tests
│   ├── model_vs_random_evaluation/ # Scripts for comparing trained model vs. random baseline
│   │   └── __init__.py
│   ├── __init__.py
│   ├── learning_progress_analyzer.py # Analyzes student learning progress from logs
│   ├── model_vs_random.py      # Script for running model vs. random baseline comparison
│   └── student_model_tester.py # Tests a trained student model
├── training/                   # Core training logic for RL agents
│   ├── agent_trainer.py        # Contains `train_single_student` and `train` functions
│   └── environment_setup.py    # Environment creation utilities and `mask_fn`
├── training_data/              # JSON logs of student interactions during training
├── utils/                      # Helper scripts and data generators
│   ├── CurriculumGenerator.py
│   ├── dataset_analysis.py
│   └── student_profile_generator.py
├── config.py                   # Centralized configuration for paths, hyperparameters, and constants
├── main.py                     # Main entry point to run the training process
├── poetry.lock                 # Poetry lock file for exact dependency versions
├── pyproject.toml              # Poetry project configuration and dependencies
└── README.md                   # This README file
```

-----

## 🧠 Key Concepts

  - **Learning Style (VARK):** Each student has a dominant learning style (Visual, Auditory, Read/Write, Kinesthetic), which influences their learning effectiveness.
  - **Velocity:** Describes how quickly the virtual student can assimilate knowledge and master lessons.
  - **Activities:** Each activity is designed to cover specific lessons, has a defined learning style distribution, and a point value indicating its pedagogical weight.
  - **Rewards:** The RL agent receives rewards based on how well the chosen activity aligns with the student’s learning style and velocity, leading to efficient mastery.
  - **Constraints:** The agent must adhere to pedagogical constraints, such as respecting lesson prerequisites, avoiding repeated activities, and ensuring activities contribute to unmastered lessons.

-----

## 🛠 Technologies Used

  - 🐍 **Python**: The primary programming language.
  - 🧪 **[Gymnasium](https://www.farama.org/Projects/Gymnasium/)**: For defining the adaptive learning environment as a standard RL environment.
  - 🤖 **[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/)**: A set of reliable implementations of RL algorithms in PyTorch.
  - 🧠 **[SB3-Contrib (Maskable PPO)](https://sb3-contrib.readthedocs.io/)**: An extended library for SB3, providing algorithms that support action masking (essential for handling pedagogical constraints).
  - 📊 **Pandas, NumPy**: For data manipulation and numerical operations.
  - 📈 **Matplotlib, Seaborn**: For data visualization and plotting results.
  - 📦 **[Poetry](https://python-poetry.org/)**: For dependency management and project packaging.
  - 💻 **Google Colab**: Used for development, training, and experimentation, leveraging GPU resources.

-----

## ⚙️ How to Run the Project

Follow these steps to set up and run the project:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Herriot65/A-Reinforcement-Learning-Algorithm-for-Adaptive-Learning.git
    cd RL_FOR_ADL
    ```

2.  **Install dependencies using Poetry:**
    If you don't have Poetry installed, follow their official installation guide: [Poetry Installation Guide](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation).

    Once Poetry is installed, navigate to the project root directory (`RL_FOR_ADL/`) and install the project dependencies:

    ```bash
    poetry install
    ```

    This command will create a virtual environment (if not already present) and install all packages specified in `pyproject.toml` and `poetry.lock`.

3.  **Activate the Poetry shell:**
    To execute Python scripts within the project's isolated virtual environment, activate the Poetry shell:

    ```bash
    poetry shell
    ```

4.  **Generate curriculum data:**
    Before training, ensure your curriculum data is generated:

    ```bash
    python utils/CurriculumGenerator.py
    ```

5.  **Train the agent(s):**
    Initiate the training process for the RL agent(s) with defined student profiles:

    ```bash
    python main.py
    ```

    Training logs and models will be saved in the `training_data/`, `logs/`, and `models/` directories respectively.

6.  **Evaluate the trained agent(s):**
    To test the performance of the trained models and generate detailed evaluation results:

    ```bash
    python tests/student_model_tester.py
    # or to compare against a random baseline
    python tests/model_vs_random.py
    ```

    Evaluation outputs, including mastery progression graphs and performance metrics, will be stored in `tests/model_test_results/`.

-----

## 📊 Outputs

Upon successful execution, the project will generate the following outputs:

  - **Training logs**: Detailed interaction data between the agent and the environment in `.json` format, stored in `/training_data/`.
  - **Mastery progression graphs**: Visualizations depicting the student's mastery level over time.
  - **Style-alignment and performance visualizations**: Graphs illustrating how well the agent's recommendations matched student styles and overall learning efficiency.
  - **Comparison with a random agent baseline**: Performance comparison to demonstrate the RL agent's effectiveness against a non-adaptive approach.
  - **Trained models**: Saved PPO models in `/models/`, ready for deployment or further analysis.

-----

## 📌 Future Work

Potential areas for future enhancement and research include:

  - **Introduce difficulty levels in activities:** Implement a more nuanced activity selection based on dynamic difficulty adjustment.
  - **Move from profile-specific models to a general pre-trained model:** Develop a single RL agent capable of adapting to diverse student profiles without retraining for each new student type.
  - **Test with real learner data for real-world validation:** Validate the agent's efficacy using actual student interaction data, moving beyond simulated environments.

-----

## 🧑‍💻 Author

[DAGOUDI Herriot Déo-gratias]

Student in Computer Science & Software Engineering

Email: [herriotdagoudi@gmail.com]

-----

## 📄 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

-----