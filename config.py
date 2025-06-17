# Base directories
MODELS_DIR = "models/"
LOGS_DIR = "logs/"
DATA_DIR = "training_data/"
CURRICULUM_FILE = "data/learning_curriculum.json"
MODEL_NAME = "student_model"
LOG_FILE_NAME_JSON = "student_data.json"
MODEL_TEST_RESULTS_BASE_DIR = "tests/model_test_results/"

# Default training hyperparameters
DEFAULT_TOTAL_TIMESTEPS = 200000 
N_ENVS = 4
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 10
CLIP_RANGE = 0.2
N_STEPS = 2048
GAMMA = 0.95
GAE_LAMBDA = 0.97
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
VERBOSE = 1
MAX_STEPS_PER_EPISODE = 1000

# Testing and Analysis
TEST_AFTER_TRAINING = True
TEST_EPISODES = 1
ANALYZE_PROGRESS = True
MASTERY_THRESHOLD = 1.0

# Specific override for slower students
SLOW_STUDENT_VELOCITY_THRESHOLD = 0.45
SLOW_STUDENT_TOTAL_TIMESTEPS = 250000