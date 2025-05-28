import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
TRAINING_DATA_FILE = "training_data/training_logs_new_dataset.json"
PLOTS_FOLDER = "test_plots_phases" # New folder for these plots
mastery_target = 1.0 # Define the target mastery level

# --- Create the plots folder if it doesn't exist ---
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# --- Load the full dataset ---
try:
    with open(TRAINING_DATA_FILE, "r") as f:
        full_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{TRAINING_DATA_FILE}' was not found.")
    print("Please ensure you have run your 'train.py' script successfully to generate this file.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{TRAINING_DATA_FILE}'. The file might be corrupted.")
    exit()

# Split data into phases: early (0-10%), middle (45-55%), late (90-100%) of episodes
episodes = full_data["episodes"]
total_episodes = len(episodes)

def extract_phase_indices(total, phase_percent):
    start = int(total * phase_percent[0])
    end = int(total * phase_percent[1])
    # Ensure end index does not exceed total_episodes
    return range(start, min(end, total))

# Define three phases
phases = {
    "Early": extract_phase_indices(total_episodes, (0.0, 0.1)),
    "Middle": extract_phase_indices(total_episodes, (0.45, 0.55)),
    "Late": extract_phase_indices(total_episodes, (0.9, 1.0))
}

# Dynamically get lessons from the data, similar to previous code
unique_lesson_ids_found = set()
for episode_data in episodes:
    for step_data in episode_data['steps']:
        for lesson_id in step_data['mastery_levels'].keys():
            unique_lesson_ids_found.add(lesson_id)
lessons = sorted(list(unique_lesson_ids_found))

# For each phase, collect the average mastery vs number of steps for each lesson
phase_curves = {lesson: {"Early": [], "Middle": [], "Late": []} for lesson in lessons}

for phase_name, indices in phases.items():
    for lesson in lessons:
        all_mastery = []

        # Collect mastery levels for all episodes in the current phase
        for idx in indices:
            if idx < len(episodes): # Ensure index is within bounds
                episode = episodes[idx]
                steps_data = episode["steps"]
                mastery_per_step_in_episode = []

                for step_entry in steps_data:
                    # Append mastery level for the current lesson, default to 0 if not present
                    mastery_per_step_in_episode.append(step_entry["mastery_levels"].get(lesson, 0.0))
                all_mastery.append(mastery_per_step_in_episode)

        # Determine the maximum number of steps across all episodes in this phase for this lesson
        max_steps = 0
        if all_mastery:
            max_steps = max(len(m) for m in all_mastery)
        
        avg_mastery_per_step = []

        # Calculate average mastery for each step
        for step_i in range(max_steps):
            step_values = [m[step_i] for m in all_mastery if len(m) > step_i]
            avg = np.mean(step_values) if step_values else np.nan # Use NaN for missing values
            avg_mastery_per_step.append(avg)

        phase_curves[lesson][phase_name] = avg_mastery_per_step

# Plot
fig, axs = plt.subplots(2, 2, figsize=(15, 12)) # Adjusted figure size for better readability
axs = axs.flatten()

colors = {"Early": "red", "Middle": "orange", "Late": "green"}

for i, lesson in enumerate(lessons):
    ax = axs[i]
    
    for phase in ["Early", "Middle", "Late"]:
        y_full = np.array(phase_curves[lesson][phase])
        
        if len(y_full) == 0:
            continue
            
        # Find the first step where mastery starts improving (> 0.01)
        improvement_threshold = 0.01
        first_improvement_idx = np.where(y_full > improvement_threshold)[0]
        
        if len(first_improvement_idx) == 0:
            # No improvement found, skip this phase
            continue
            
        start_idx = first_improvement_idx[0]
        
        # Find the first step where mastery reaches or exceeds the target AFTER improvement starts
        mastery_reached_idx = None
        
        # Debug: Print max mastery value for this phase/lesson combination
        max_mastery = np.max(y_full) if len(y_full) > 0 else 0
        print(f"Lesson {lesson}, Phase {phase}: Max mastery = {max_mastery:.3f}, Start idx = {start_idx}")
        
        # Look for mastery >= 0.99 first, then 0.95, then 0.90
        for threshold in [0.99, 0.95, 0.90]:
            for step_idx in range(start_idx, len(y_full)):
                if y_full[step_idx] >= threshold:
                    mastery_reached_idx = step_idx
                    print(f"  Found mastery at step {step_idx} with value {y_full[step_idx]:.3f} (threshold {threshold})")
                    break
            if mastery_reached_idx is not None:
                break
        
        if mastery_reached_idx is not None:
            # Mastery is reached - stop exactly at that point
            end_idx = mastery_reached_idx + 1  # +1 to include the mastery point
        else:
            # Mastery not reached, use all available data
            end_idx = len(y_full)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(y_full))
        
        # Create the plot data
        x_plot = np.arange(start_idx, end_idx)
        y_plot = y_full[start_idx:end_idx]
        
        # Only plot if there's meaningful data
        if len(x_plot) > 0 and len(y_plot) > 0:
            ax.plot(x_plot, y_plot, label=phase, color=colors[phase], linewidth=2)
            
            # Add a marker at the point of mastery if it was reached
            if mastery_reached_idx is not None:
                ax.plot(mastery_reached_idx, y_full[mastery_reached_idx], 'o', 
                       color=colors[phase], markersize=8, alpha=0.8,
                       markeredgecolor='black', markeredgewidth=1)

    ax.set_title(f"Lesson {lesson}", fontsize=14)
    ax.set_xlabel("Steps (within episode)", fontsize=12)
    ax.set_ylabel("Average Mastery", fontsize=12)
    ax.set_ylim(-0.05, 1.05) # Adjust y-lim to show 0 clearly and slightly above 1
    ax.grid(True, alpha=0.3)
    ax.legend(title="Training Phase", fontsize=10, title_fontsize=11)
    
    # Set x-axis ticks dynamically based on the actual data range shown
    all_x_values = []
    for phase in ["Early", "Middle", "Late"]:
        y_data = np.array(phase_curves[lesson][phase])
        if len(y_data) > 0:
            first_improvement_idx = np.where(y_data > 0.01)[0]
            if len(first_improvement_idx) > 0:
                start_idx = first_improvement_idx[0]
                
                # Find mastery point after improvement starts
                mastery_step = None
                for threshold in [0.99, 0.95, 0.90]:
                    for step_idx in range(start_idx, len(y_data)):
                        if y_data[step_idx] >= threshold:
                            mastery_step = step_idx
                            break
                    if mastery_step is not None:
                        break
                
                if mastery_step is not None:
                    end_idx = mastery_step
                else:
                    end_idx = len(y_data) - 1
                
                all_x_values.extend(range(start_idx, end_idx + 1))
    
    if all_x_values:
        min_x = min(all_x_values)
        max_x = max(all_x_values)
        # Create ticks at intervals of 5, starting from the nearest multiple of 5 <= min_x
        tick_start = (min_x // 5) * 5
        tick_end = ((max_x // 5) + 1) * 5
        ax.set_xticks(np.arange(tick_start, tick_end + 1, 5))
        ax.set_xlim(min_x - 1, max_x + 1)
    else:
        ax.set_xticks(np.arange(0, 50, 5))

plt.suptitle("Learning Progress Over Time for Each Lesson (Mastery vs Steps)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot before showing it
fig_filename = os.path.join(PLOTS_FOLDER, "all_lessons_learning_progress_phases.png")
plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to '{fig_filename}'")

# plt.show()