import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # defaultdict is not strictly needed for the current logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming sb3_contrib and envs are correctly set up in the path
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv

from training.train import mask_fn, load_curriculum

def test_model_with_student(model, lessons, activities, student, num_episodes=2):
    """Test the model with a specific student profile and collect data."""
    # Convert activities list to dictionary for efficient lookup
    activities_dict = {a['id']: a for a in activities}

    test_env = AdaptiveLearningEnv(
        lessons,
        activities,
        student,
        log_training_data=True,
        log_file=f"test_data_{student.dominant_style.lower()}.json"
    )
    masked_env = ActionMasker(test_env, mask_fn)

    episode_data = []

    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}")

        obs, info = masked_env.reset()
        episode_steps = []
        step_count = 0

        while True:
            action_masks = info.get('available_actions', np.ones(len(activities)))
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = masked_env.step(action)

            activity_id = info['activity_id']
            activity = activities_dict[activity_id]

            # Determine dominant learning style for this activity
            activity_styles = activity.get('style', {})
            dominant_style = 'Unknown'
            dominant_value = 0.0
            if activity_styles:
                dominant_style = max(activity_styles, key=activity_styles.get)
                dominant_value = activity_styles[dominant_style]

            step_data = {
                'step': step_count,
                'activity_id': activity_id,
                'activity_style': dominant_style,
                'activity_style_strength': dominant_value,
                'activity_styles_full': activity_styles,
                'performance': info['performance'],
                'mastery_levels': test_env.lesson_mastery.mastery.copy(),
                'reward': reward
            }
            episode_steps.append(step_data)
            step_count += 1

            if terminated or truncated:
                break

        episode_data.append({
            'episode': episode,
            'steps': episode_steps,
            'total_steps': len(episode_steps),
            'final_mastery': test_env.lesson_mastery.mastery.copy()
        })

    test_env.finalize_logging()
    return episode_data, test_env.lessons

# --- Plotting Functions ---
def save_plot(fig, filename, plot_dir="plots"):
    """Saves the given matplotlib figure to a specified directory."""
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot saved to {filepath}")

def plot_mastery_evolution(episode_data, lessons, student_profile, plot_dir="plots"):
    """Plot individual lesson mastery evolution curves and save them."""
    lesson_names = list(lessons.keys())
    num_lessons = len(lesson_names)

    cols = 3
    rows = (num_lessons + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f'Individual Lesson Mastery Evolution - {student_profile["dominant_style"]} Student ({student_profile["dominant_percent"]}%)',
                     fontsize=16, fontweight='bold')

    axes = axes.flatten() if num_lessons > 1 else [axes]

    episode_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Consistent colors

    for lesson_idx, lesson_name in enumerate(lesson_names):
        ax = axes[lesson_idx]
        for episode_idx, episode in enumerate(episode_data):
            steps = episode['steps']
            step_numbers = [step['step'] for step in steps]
            mastery_values = [step['mastery_levels'][lesson_name] for step in steps]

            ax.plot(step_numbers, mastery_values,
                    color=episode_colors[episode_idx % len(episode_colors)],
                    marker='o', markersize=4, linewidth=2,
                    label=f'Episode {episode_idx + 1}')

            # Annotate activity IDs sparingly
            for i, (step_num, mastery, step_data) in enumerate(zip(step_numbers, mastery_values, steps)):
                if i % 5 == 0 or i == len(step_numbers) - 1:
                    ax.annotate(step_data['activity_id'],
                                (step_num, mastery),
                                textcoords="offset points", xytext=(0,10),
                                ha='center', fontsize=8, alpha=0.7)

        ax.set_title(f'Mastery Evolution for Lesson {lesson_name}', fontweight='bold')
        ax.set_xlabel('Step Number')
        ax.set_ylabel('Mastery Level')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots if any
    for i in range(num_lessons, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_plot(fig, f'mastery_evolution_{student_profile["dominant_style"].lower()}.png', plot_dir)


def plot_activity_style_distribution(episode_data, student_profile, plot_dir="plots"):
    """Plot the distribution of learning styles and save it."""
    dominant_styles = []
    style_strengths = []
    strong_dominance_threshold = 0.4

    for episode in episode_data:
        for step in episode['steps']:
            dominant_styles.append(step['activity_style'])
            style_strengths.append(step.get('activity_style_strength', 0.0))

    style_counts = Counter(dominant_styles)
    student_dominant_style = student_profile['dominant_style'].lower()
    total_activities = len(dominant_styles)

    student_style_activities = 0
    strong_student_style_activities = 0

    for i, (style, strength) in enumerate(zip(dominant_styles, style_strengths)):
        if style.lower() == student_dominant_style:
            student_style_activities += 1
            if strength >= strong_dominance_threshold:
                strong_student_style_activities += 1

    other_activities = total_activities - student_style_activities

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Pie Chart 1: Student's dominant style vs others
    sizes = [student_style_activities, other_activities]
    labels = [f'{student_profile["dominant_style"]}\n({student_style_activities} activities)',
              f'Other Styles\n({other_activities} activities)']
    colors = ['#ff6b6b', '#4ecdc4']

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                     colors=colors, startangle=90,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)

    ax1.set_title(f'Activities by Dominant Learning Style\n{student_profile["dominant_style"]} Student ({student_profile["dominant_percent"]}% preference)',
                  fontsize=14, fontweight='bold', pad=20)

    # Bar Chart: Detailed breakdown of all dominant styles
    style_names = list(style_counts.keys())
    style_values = list(style_counts.values())

    bar_colors = ['#ff6b6b' if style.lower() == student_dominant_style else '#95a5a6'
                  for style in style_names]

    bars = ax2.bar(style_names, style_values, color=bar_colors, alpha=0.8,
                    edgecolor='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    ax2.set_title('Count of Activities by Dominant Style', fontweight='bold')
    ax2.set_xlabel('Dominant Learning Style')
    ax2.set_ylabel('Number of Activities')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_plot(fig, f'activity_style_distribution_{student_profile["dominant_style"].lower()}.png', plot_dir)

    # Print detailed analysis
    print("\n" + "="*70)
    print("DOMINANT LEARNING STYLE ANALYSIS")
    print("="*70)
    print(f"Student Profile: {student_profile['dominant_style']} ({student_profile['dominant_percent']}% preference)")
    print(f"Total Activities Suggested: {total_activities}")
    print(f"Strong Dominance Threshold: {strong_dominance_threshold*100}%")
    print()

    print("Activities by Dominant Style:")
    for style, count in style_counts.most_common():
        percentage = (count / total_activities) * 100
        marker = "✓" if style.lower() == student_dominant_style else " "
        print(f"   {marker} {style.capitalize()}: {count} activities ({percentage:.1f}%)")

    print()
    print("Personalization Analysis:")
    student_percentage = (student_style_activities / total_activities) * 100
    strong_percentage = (strong_student_style_activities / total_activities) * 100

    print(f"   Activities with {student_profile['dominant_style']} dominance: {student_style_activities}/{total_activities} ({student_percentage:.1f}%)")
    print(f"   Activities with STRONG {student_profile['dominant_style']} dominance (>{strong_dominance_threshold*100}%): {strong_student_style_activities}/{total_activities} ({strong_percentage:.1f}%)")

    print(f"\nPersonalization Assessment:")
    if student_percentage >= 60:
        assessment = "EXCELLENT"
        icon = "✓✓"
    elif student_percentage >= 40:
        assessment = "GOOD"
        icon = "✓"
    elif student_percentage >= 25:
        assessment = "MODERATE"
        icon = "~"
    else:
        assessment = "LOW"
        icon = "⚠"

    print(f"   {icon} {assessment} - Model personalizes {student_percentage:.1f}% of activities to student's dominant style")

    if strong_percentage >= 30:
        print(f"   ✓ Strong personalization with {strong_percentage:.1f}% having strong style dominance")
    elif strong_percentage >= 15:
        print(f"   ~ Moderate strong personalization with {strong_percentage:.1f}% having strong style dominance")
    else:
        print(f"   ⚠ Weak strong personalization with only {strong_percentage:.1f}% having strong style dominance")

    print("="*70)

# --- Main Execution ---
def main():
    """Main testing function."""
    # Configuration
    curriculum_file = "data/adaptive_learning_curriculum.json"
    model_path = "models/adaptive_learning_model_50k_biased"
    plot_output_dir = "plots" # New directory for plots

    print("Loading curriculum and trained model...")

    lessons, activities = load_curriculum(curriculum_file)
    model = MaskablePPO.load(model_path)

    # Create a new student profile for testing
    test_student = Student(
        dominant_style="Visual", # Example: "Visual", "Auditory", "Reading/Writing", "Kinesthetic"
        dominant_percent=70,    # How dominant is this style (e.g., 70% Visual)
        velocity=0.9            # Learning velocity
    )

    print(f"\nTesting with new student profile:")
    print(f"   Dominant Style: {test_student.dominant_style}")
    print(f"   Dominant Percentage: {test_student.dominant_percent}%")
    print(f"   Learning Style Distribution: {test_student.learning_style}")
    print(f"   Velocity: {test_student.velocity}")

    print(f"\nRunning test episodes...")
    episode_data, lesson_info = test_model_with_student(
        model, lessons, activities, test_student, num_episodes=1 # Set to 1 for lighter testing
    )

    student_profile = {
        'dominant_style': test_student.dominant_style,
        'dominant_percent': test_student.dominant_percent,
        'learning_style': test_student.learning_style
    }

    print(f"\nGenerating and saving visualizations to '{plot_output_dir}' folder...")

    plot_mastery_evolution(episode_data, lesson_info, student_profile, plot_output_dir)
    plot_activity_style_distribution(episode_data, student_profile, plot_output_dir)

    # Print episode summary
    print("\n" + "="*50)
    print("EPISODE SUMMARY")
    print("="*50)
    for i, episode in enumerate(episode_data):
        print(f"Episode {i+1}:")
        print(f"   Total Steps: {episode['total_steps']}")
        final_mastery = episode['final_mastery']
        avg_mastery = np.mean(list(final_mastery.values()))
        print(f"   Average Final Mastery: {avg_mastery:.3f}")
        print(f"   Lessons Mastered: {sum(1 for m in final_mastery.values() if m >= 0.8)}/{len(final_mastery)}")
        print()

if __name__ == "__main__":
    main()