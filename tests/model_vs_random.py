import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv

def mask_fn(env):
    """
    Returns the action mask from the environment. This mask indicates which actions
    are currently valid (e.g., based on prerequisites or already completed activities).
    This function is used by the ActionMasker wrapper for Stable Baselines3.

    Args:
        env (gym.Env): The Gymnasium environment instance.

    Returns:
        np.ndarray: A binary array where 1 indicates a valid action and 0 an invalid one.
    """
    return env._get_available_actions_mask()

def create_test_env(lessons, activities, student, max_steps=1000):
    """
    Creates and returns an instance of the AdaptiveLearningEnv, specifically configured
    for testing. Logging of training data is disabled to ensure clean evaluation runs.
    The environment is wrapped with ActionMasker to automatically apply action masks.

    Args:
        lessons (List[Dict]): A list of lesson dictionaries.
        activities (List[Dict]): A list of activity dictionaries.
        student (Student): An instance of the Student class.
        max_steps (int, optional): The maximum number of steps allowed per episode. Defaults to 1000.

    Returns:
        ActionMasker: An ActionMasker-wrapped AdaptiveLearningEnv instance ready for testing.
    """
    env = AdaptiveLearningEnv(
        lessons,
        activities,
        student=student,
        log_training_data=False,  
        max_steps=max_steps
    )
    return ActionMasker(env, mask_fn) 

def get_activity_dominant_style(activity):
    """
    Determines the dominant learning style for a given activity.
    It identifies the style (e.g., 'visual', 'auditory') that has the highest
    relevance score within the activity's 'style' dictionary.

    Args:
        activity (Dict): A dictionary representing an activity, expected to have a 'style' key.

    Returns:
        str: The name of the dominant learning style for the activity.
    """
    return max(activity['style'], key=activity['style'].get)

def convert_student_style_to_activity_format(student_style):
    """
    Converts a student's dominant learning style string (e.g., "Read/Write")
    to a format consistent with the keys used in activity style dictionaries (e.g., "read_write").

    Args:
        student_style (str): The dominant learning style of the student.

    Returns:
        str: The converted style string.
    """
    style_mapping = {
        "Visual": "visual",
        "Auditory": "auditory",
        "Read/Write": "read_write",
        "Kinesthetic": "kinesthetic"
    }
    return style_mapping.get(student_style, student_style.lower())

def test_trained_model_with_style_tracking(model_path, lessons, activities, student, num_episodes=10):
    """
    Evaluates a pre-trained reinforcement learning model over multiple episodes,
    tracking performance metrics including steps to completion, total reward,
    final mastery, and alignment with the student's learning style.

    Args:
        model_path (str): The file path to the trained MaskablePPO model.
        lessons (List[Dict]): A list of lesson dictionaries.
        activities (List[Dict]): A list of activity dictionaries (used for indexing by action).
        student (Student): The student profile for the evaluation.
        num_episodes (int, optional): The number of episodes to run for evaluation. Defaults to 10.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains performance
                    metrics for one episode.
    """
    print(f"Loading model from: {model_path}")
    model = MaskablePPO.load(model_path) 
    print("Testing trained model...")

    results = []
    
    # Convert student's dominant style for consistent comparison with activity styles
    student_style_key = convert_student_style_to_activity_format(student.dominant_style)

    for _ in range(num_episodes):
        # Create a fresh environment for each episode to ensure independent runs
        env = create_test_env(lessons, activities, student)
        obs, info = env.reset() 

        total_reward = 0
        steps = 0
        done = False
        style_matches = 0 
        total_actions = 0 

        while not done:
            action_masks = info.get('available_actions', None)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            selected_activity = activities[action]
            
            activity_dominant_style = get_activity_dominant_style(selected_activity)
            
            if activity_dominant_style == student_style_key:
                style_matches += 1
            total_actions += 1 

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated 

        # Calculate final metrics for the episode
        final_mastery = np.mean(list(env.unwrapped.lesson_mastery.mastery.values()))
        completion = 1.0 if terminated else 0.0 
        style_alignment = (style_matches / total_actions) * 100 if total_actions > 0 else 0 
        
        # Store results for the current episode
        results.append({
            'reward': total_reward,
            'steps': steps,
            'completed': completion,
            'final_mastery': final_mastery,
            'style_alignment': style_alignment
        })

    return results

def test_random_model_with_style_tracking(lessons, activities, student, num_episodes=10):
    """
    Evaluates a 'random' baseline model over multiple episodes. This model
    selects actions randomly from the available (valid) actions. It tracks
    the same performance metrics as the trained model for comparison.

    Args:
        lessons (List[Dict]): A list of lesson dictionaries.
        activities (List[Dict]): A list of activity dictionaries (used for indexing by action).
        student (Student): The student profile for the evaluation.
        num_episodes (int, optional): The number of episodes to run for evaluation. Defaults to 10.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains performance
                    metrics for one episode.
    """
    print("Testing random model...")

    results = []
    student_style_key = convert_student_style_to_activity_format(student.dominant_style)

    for episode in range(num_episodes):
        env = create_test_env(lessons, activities, student)
        _, info = env.reset() 

        total_reward = 0
        steps = 0
        done = False
        style_matches = 0
        total_actions = 0

        while not done:
            action_masks = info.get('available_actions', None)
            if action_masks is not None:
                valid_actions = np.where(action_masks == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
            else:
                action = np.random.randint(0, len(activities))

            selected_activity = activities[action]
            activity_dominant_style = get_activity_dominant_style(selected_activity)
            
            if activity_dominant_style == student_style_key:
                style_matches += 1
            total_actions += 1

            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        # Calculate final metrics for the episode
        final_mastery = np.mean(list(env.unwrapped.lesson_mastery.mastery.values()))
        completion = 1.0 if terminated else 0.0
        style_alignment = (style_matches / total_actions) * 100 if total_actions > 0 else 0

        # Store results for the current episode
        results.append({
            'reward': total_reward,
            'steps': steps,
            'completed': completion,
            'final_mastery': final_mastery,
            'style_alignment': style_alignment
        })
        print(f"Episode {episode + 1}: Steps: {steps}, Completed: {completion}, Final Mastery: {final_mastery}, Style Alignment: {style_alignment:.2f}%")
    return results

def create_comparison_charts(trained_results, random_results, student, model_name):
    """
    Generates and saves comparative plots (box plots) and a summary table
    to visualize the performance of the trained model against a random baseline.
    The charts highlight efficiency (activities to completion) and personalization
    (style alignment).

    Args:
        trained_results (List[Dict]): Performance data from the trained model evaluation.
        random_results (List[Dict]): Performance data from the random model evaluation.
        student (Student): The student profile for which the evaluation was run.
        model_name (str): The name of the trained model (used for chart titles and directory naming).
    """
    def safe_annotate(ax, text, x, y, dy=5, color='gold', arrow_color='darkorange'):
        """
        Helper function to add annotations to matplotlib plots, adjusting their
        vertical position to avoid going off-chart.
        """
        y_values = [line.get_ydata() for line in ax.lines if hasattr(line, 'get_ydata')]
        y_non_empty = [y for y in y_values if len(y) > 0]
        ymax = max(y.max() for y in y_non_empty) if y_non_empty else y + dy
        ytext = y + dy if y + dy < ymax + 10 else y - dy # Adjust text position based on max y-value

        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x - 0.3, ytext), 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2)
        )

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    colors = ['#FF6B6B', '#4ECDC4']  # Red and Blue

    # Extract relevant data for plotting
    trained_steps = [r['steps'] for r in trained_results]
    random_steps = [r['steps'] for r in random_results]
    trained_style = [r['style_alignment'] for r in trained_results]
    random_style = [r['style_alignment'] for r in random_results]

    # Calculate improvement multipliers
    efficiency_multiplier = np.mean(random_steps) / max(np.mean(trained_steps), 1e-6)
    style_multiplier = np.mean(trained_style) / max(np.mean(random_style), 1e-6)
    completion_multiplier = np.mean([r["completed"] for r in trained_results]) / max(np.mean([r["completed"] for r in random_results]), 1e-6)

    # Sanitize model name for use in file paths (remove slashes)
    sanitized_model_name = model_name.replace("/", "_").replace("\\", "_")
    output_dir = f"tests/model_vs_random_evaluation/{sanitized_model_name}"
    os.makedirs(output_dir, exist_ok=True)
    display_velocity = int(student.velocity * 100)

    # --- 1. Activities to Completion Chart ---
    fig_steps = plt.figure(figsize=(9, 7))
    ax_steps = fig_steps.add_subplot(111)

    # Create box plot for activities to completion
    bp_steps = ax_steps.boxplot([random_steps, trained_steps],
                                 tick_labels=['Random\nBaseline', 'Trained\nModel'],
                                 patch_artist=True, notch=True, 
                                 boxprops=dict(linewidth=1.5),
                                 medianprops=dict(color='black', linewidth=1.5),
                                 whiskerprops=dict(linewidth=1.5),
                                 capprops=dict(linewidth=1.5))

    # Apply colors to the box plot patches
    for patch, color in zip(bp_steps['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set title and labels for the chart
    ax_steps.set_title(f'Performance Summary: Activities to Completion\nStudent Profile: {student.dominant_style} Learner ({student.dominant_percent}% dominant, velocity={display_velocity}%)',
                       fontsize=16, fontweight='bold', pad=20)
    ax_steps.set_ylabel('Number of Activities', fontsize=12)
    ax_steps.grid(True, linestyle='--', alpha=0.6) 
    
    safe_annotate(ax_steps,
                  f'{efficiency_multiplier:.1f}x more efficient',
                  x=1.8, 
                  y=np.mean(trained_steps), 
                  dy=25,
                  color='lightgreen',
                  arrow_color='darkgreen')

    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, 'activities_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_steps) 

    # --- 2. Style Alignment Chart ---
    fig_style = plt.figure(figsize=(9, 7))
    ax_style = fig_style.add_subplot(111)

    # Create box plot for style alignment
    bp_style = ax_style.boxplot([random_style, trained_style],
                                 tick_labels=['Random\nBaseline', 'Trained\nModel'],
                                 patch_artist=True, notch=True,
                                 boxprops=dict(linewidth=1.5),
                                 medianprops=dict(color='black', linewidth=1.5),
                                 whiskerprops=dict(linewidth=1.5),
                                 capprops=dict(linewidth=1.5))

    # Apply colors to the box plot patches
    for patch, color in zip(bp_style['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set title and labels for the chart
    ax_style.set_title(f'Personalized Learning: Style Alignment\nStudent Profile: {student.dominant_style} Learner ({student.dominant_percent}% dominant, velocity={display_velocity}%)',
                       fontsize=16, fontweight='bold', pad=20)
    ax_style.set_ylabel('Style Alignment (%)', fontsize=12)
    ax_style.grid(True, linestyle='--', alpha=0.6)

    # Add annotation for style alignment improvement
    safe_annotate(ax_style,
                  f'{style_multiplier:.1f}x better alignment',
                  x=1.5, 
                  y=np.mean(trained_style), 
                  dy=15,
                  color='lightgreen',
                  arrow_color='darkgreen')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_alignment.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_style)

    # --- 3. Summary Table ---
    summary_data = [
        ['Metric', 'Random Baseline', 'Trained Model', 'Improvement'],
        ['Avg. Activities', f'{np.mean(random_steps):.0f}', f'{np.mean(trained_steps):.0f}', f'{efficiency_multiplier:.1f}x more efficient'],
        ['Style Alignment', f'{np.mean(random_style):.1f}', f'{np.mean(trained_style):.1f}', f'{style_multiplier:.1f}x better'],
        ['Completion Rate', f'{np.mean([r["completed"] for r in random_results]):.2f}',
         f'{np.mean([r["completed"] for r in trained_results]):.2f}',
         f'{completion_multiplier:.1f}x better']
    ]

    fig_table = plt.figure(figsize=(12, 6))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off') 

    # Create the table
    table_separate = ax_table.table(cellText=summary_data[1:], colLabels=summary_data[0],
                                     cellLoc='center', loc='center',
                                     colWidths=[0.25, 0.25, 0.25, 0.25]) 
    table_separate.auto_set_font_size(False)
    table_separate.set_fontsize(12)
    table_separate.scale(1, 2.5) 

    # Style table header
    for i in range(len(summary_data[0])):
        table_separate[(0, i)].set_facecolor('#4ECDC4') 
        table_separate[(0, i)].set_text_props(weight='bold', color='white') 
    for i in range(1, len(summary_data)):
        table_separate[(i, 3)].set_facecolor('#E8F5E8') 

    # Set title for the table
    ax_table.set_title(f'Performance Summary Statistics\nStudent Profile: {student.dominant_style} Learner ({student.dominant_percent}% dominant, velocity={display_velocity}%)',
                       fontsize=18, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(f'tests/model_vs_random_evaluation/{model_name}/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig_table)

    print(f"Charts and summary table for {model_name} saved in the '{output_dir}' directory.")


def main(students, model_paths, curriculum_path="learning_curriculum.json", model_num_episodes=1, random_model_num_episodes=100):
    """
    Main function to orchestrate the evaluation process.
    It loads curriculum data, iterates through student profiles and their
    corresponding trained models, performs evaluations, and generates comparison charts.

    Args:
        students (List[Student]): A list of Student objects, each representing a student profile.
        model_paths (List[str]): A list of file paths to the trained models,
                                 corresponding to the student profiles.
        curriculum_path (str, optional): Path to the JSON file containing lesson and activity data.
                                         Defaults to "learning_curriculum.json".
        model_num_episodes (int, optional): Number of episodes to evaluate the trained model. Defaults to 1.
        random_model_num_episodes (int, optional): Number of episodes to evaluate the random baseline. Defaults to 100.
    """
    with open(curriculum_path) as f:
        data = json.load(f)
    lessons = data["lessons"]
    activities = data["activities"]

    # Iterate through each model path and corresponding student profile
    for model_path, student in zip(model_paths, students):
        model_name = os.path.basename(model_path) 
        print(f"\n--- Evaluating {model_name} ---")
        print(f"Student: {student.dominant_style}, {student.dominant_percent}% (vel={student.velocity})")

        # Test the trained model
        trained_results = test_trained_model_with_style_tracking(model_path, lessons, activities, student, model_num_episodes)
        
        # Test the random baseline model
        random_results = test_random_model_with_style_tracking(lessons, activities, student, random_model_num_episodes)
        
        # Generate and save comparison charts
        create_comparison_charts(trained_results, random_results, student, model_name)

if __name__ == "__main__":
    model_paths = [
        "models/student_model_Read_Write_85pct_045vel",
        "models/student_model_Read_Write_85pct_075vel",
        "models/student_model_Read_Write_85pct_09vel",
        "models/student_model_Visual_80pct_04vel",
        "models/student_model_Visual_80pct_065vel",
        "models/student_model_Visual_80pct_09vel"
    ]

    students = [
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.45),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.75),
        Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.9),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.4),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.65),
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.9)
    ]

    main(model_num_episodes=1, random_model_num_episodes=1000, curriculum_path="data/learning_curriculum.json",
         students=students, model_paths=model_paths)