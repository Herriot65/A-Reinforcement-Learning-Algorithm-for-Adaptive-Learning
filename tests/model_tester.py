import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib import patheffects

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv

class ModelTester:
    def __init__(self, model_path: str, curriculum_path: str, output_dir: str = "test_results/", mastery_threshold: float = 0.8):
        """Initialize the model tester.
        
        Args:
            model_path: Path to the trained model
            curriculum_path: Path to curriculum JSON file
            output_dir: Directory to save test results
            mastery_threshold: Threshold to consider a lesson as mastered
        """
        self.model_path = model_path
        self.curriculum_path = curriculum_path
        self.output_dir = output_dir
        self.mastery_threshold = mastery_threshold
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load curriculum
        self.lessons, self.activities = self._load_curriculum()
        
        # Load trained model
        self.model = MaskablePPO.load(model_path)
        
    def _load_curriculum(self) -> Tuple[List[Dict], List[Dict]]:
        """Load curriculum from JSON file."""
        with open(self.curriculum_path) as f:
            data = json.load(f)
        return data["lessons"], data["activities"]
    
    def _mask_fn(self, env):
        """Action masking function for environment."""
        return env._get_available_actions_mask()
    
    def test_student(self, student: Student, num_episodes: int = 5, student_name: str = None) -> Dict:
        """Test the model with a specific student profile.
        
        Args:
            student: Student instance to test with
            num_episodes: Number of episodes to run
            student_name: Name for the student (for file naming)
            
        Returns:
            Dictionary containing test results
        """
        if student_name is None:
            student_name = f"{student.dominant_style}_{student.dominant_percent}pct"
        
        print(f"\nTesting student: {student_name}")
        print(f"Profile: {student.dominant_style} ({student.dominant_percent}%), Velocity: {student.velocity}")
        
        # Store results for all episodes
        all_episodes_data = []
        episode_summaries = []
        all_learning_paths = []
        
        for episode in range(num_episodes):
            print(f"Running episode {episode + 1}/{num_episodes}...")
            
            # Create environment for this episode
            env = AdaptiveLearningEnv(
                lessons=self.lessons,
                activities=self.activities,
                student=student,
                log_training_data=False  # We'll handle logging ourselves
            )
            masked_env = ActionMasker(env, self._mask_fn)
            
            # Run episode
            episode_data = self._run_episode(masked_env, episode + 1)
            all_episodes_data.extend(episode_data)
            
            # Generate learning paths for this episode
            episode_learning_paths = self._generate_learning_paths(episode_data, episode + 1, student_name)
            all_learning_paths.extend(episode_learning_paths)
            
            # Calculate episode summary
            episode_summary = self._calculate_episode_summary(episode_data, episode + 1, student)
            episode_summaries.append(episode_summary)
            
            masked_env.close()
        
        # Save detailed CSV data
        csv_path = os.path.join(self.output_dir, f"{student_name}_detailed_log.csv")
        self._save_to_csv(all_episodes_data, csv_path)
        
        # Generate and save learning path files
        self._save_learning_paths(all_learning_paths, student_name)
        
        # Generate global learning path
        self._generate_global_learning_path(all_learning_paths, student_name)
        
        # Generate visualizations
        self._generate_mastery_plots(all_episodes_data, student_name)
        self._generate_style_analysis(all_episodes_data, student, student_name)
                
        # Compile results
        results = {
            'student_profile': {
                'name': student_name,
                'dominant_style': student.dominant_style,
                'dominant_percent': student.dominant_percent,
                'learning_style': student.learning_style,
                'velocity': student.velocity
            },
            'episode_summaries': episode_summaries,
            'avg_steps_to_completion': float(np.mean([ep['steps_to_completion'] for ep in episode_summaries])),
            'avg_final_mastery': float(np.mean([ep['final_avg_mastery'] for ep in episode_summaries])),
            'avg_dominant_style_proportion': float(np.mean([ep['dominant_style_proportion'] for ep in episode_summaries])),
            'csv_file': csv_path
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"{student_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
        return results
    
    def _run_episode(self, env, episode_num: int) -> List[Dict]:
        """Run a single episode and collect data."""
        episode_data = []
        obs, info = env.reset()
        step = 0
        
        # Log initial state
        initial_state = {
            'episode': episode_num,
            'step': step,
            'activity_id': None,
            'activity_style': None,
            'performance': 0.0,
            'reward': 0.0,
            'terminated': False
        }
        
        # Add mastery levels and gains for each lesson (initial gains are 0)
        for i, lesson_id in enumerate([l['id'] for l in self.lessons]):
            initial_state[f'mastery_{lesson_id}'] = float(obs[i])
            initial_state[f'mastery_gain_{lesson_id}'] = 0.0
        
        episode_data.append(initial_state)
        
        terminated = False
        truncated = False
        previous_obs = obs.copy()
        
        while not terminated and not truncated:
            # Get action from model
            action_masks = info.get('available_actions', None)
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Get activity info
            activity_id = info.get('activity_id')
            activity = self.activities[action]
            
            # Log step data
            step_data = {
                'episode': episode_num,
                'step': step,
                'activity_id': activity_id,
                'activity_style': json.dumps(activity.get('style', {})),
                'performance': round(info.get('performance', 0.0), 2) if info.get('performance', 0.0) is not None else info.get('performance', 0.0),
                'reward': round(reward, 2) if reward is not None else reward,
                'terminated': terminated
            }
            
            # Add mastery levels and gains for each lesson
            for i, lesson_id in enumerate([l['id'] for l in self.lessons]):
                current_mastery = obs[i]
                previous_mastery = previous_obs[i]
                mastery_gain = current_mastery - previous_mastery
                
                step_data[f'mastery_{lesson_id}'] = round(current_mastery, 3) if current_mastery is not None else float(current_mastery)
                step_data[f'mastery_gain_{lesson_id}'] = round(mastery_gain, 3) if mastery_gain is not None else float(mastery_gain)
            
            episode_data.append(step_data)
            previous_obs = obs.copy()
        
        return episode_data
    
    def _generate_learning_paths(self, episode_data: List[Dict], episode_num: int, student_name: str) -> List[Dict]:
        """Generate learning paths for each lesson in this episode."""
        lesson_ids = [l['id'] for l in self.lessons]
        learning_paths = []
        
        # Filter out initial state (step 0)
        action_steps = [step for step in episode_data if step['step'] > 0]
        
        for lesson_id in lesson_ids:
            lesson_path = {
                'student_name': student_name,
                'episode': episode_num,
                'lesson_id': lesson_id,
                'steps': [],
                'final_mastery': 0.0,
                'mastered': False,
                'total_gain': 0.0
            }
            
            total_gain = 0.0
            
            for step_data in action_steps:
                mastery_value = step_data[f'mastery_{lesson_id}']
                mastery_gain = step_data[f'mastery_gain_{lesson_id}']
                total_gain += mastery_gain
                
                step_info = {
                    'step': step_data['step'],
                    'activity_id': step_data['activity_id'],
                    'mastery_value': mastery_value,
                    'mastery_gain': mastery_gain,
                    'performance': step_data['performance']
                }
                
                lesson_path['steps'].append(step_info)
            
            if action_steps:
                lesson_path['final_mastery'] = action_steps[-1][f'mastery_{lesson_id}']
                lesson_path['mastered'] = lesson_path['final_mastery'] >= self.mastery_threshold
                lesson_path['total_gain'] = round(total_gain, 3)
            
            learning_paths.append(lesson_path)
        
        return learning_paths
    
    def _save_learning_paths(self, learning_paths: List[Dict], student_name: str):
        """Save individual learning paths for each lesson to text files."""
        lesson_ids = [l['id'] for l in self.lessons]
        
        for lesson_id in lesson_ids:
            # Filter paths for this lesson
            lesson_paths = [path for path in learning_paths if path['lesson_id'] == lesson_id]
            
            if not lesson_paths:
                continue
            
            # Create learning path file for this lesson
            filename = f"{student_name}_{lesson_id}_learning_path.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"LEARNING PATH FOR LESSON: {lesson_id}\n")
                f.write(f"Student: {student_name}\n")
                f.write("=" * 60 + "\n\n")
                
                for episode_path in lesson_paths:
                    f.write(f"EPISODE {episode_path['episode']}:\n")
                    f.write(f"Final Mastery: {episode_path['final_mastery']:.3f}\n")
                    f.write(f"Mastered: {'YES' if episode_path['mastered'] else 'NO'} (threshold: {self.mastery_threshold})\n")
                    f.write(f"Total Mastery Gain: {episode_path['total_gain']:.3f}\n")
                    f.write("-" * 40 + "\n")
                    
                    f.write("Step-by-step progression:\n")
                    for step_info in episode_path['steps']:
                        f.write(f"  Step {step_info['step']:2d}: "
                               f"Activity {step_info['activity_id']:15s} -> "
                               f"Mastery: {step_info['mastery_value']:6.3f} "
                               f"(+{step_info['mastery_gain']:+6.3f}) "
                               f"Performance: {step_info['performance']:5.2f}\n")
                    
                    f.write("\n")
                
                # Summary across all episodes
                avg_final_mastery = np.mean([path['final_mastery'] for path in lesson_paths])
                mastery_rate = np.mean([path['mastered'] for path in lesson_paths])
                avg_total_gain = np.mean([path['total_gain'] for path in lesson_paths])
                
                f.write("SUMMARY ACROSS ALL EPISODES:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Final Mastery: {avg_final_mastery:.3f}\n")
                f.write(f"Mastery Rate: {mastery_rate:.1%}\n")
                f.write(f"Average Total Gain: {avg_total_gain:.3f}\n")
        
        print(f"Individual learning paths saved for each lesson in {self.output_dir}")
    
    def _generate_global_learning_path(self, learning_paths: List[Dict], student_name: str):
        """Generate a global learning path combining all lessons."""
        # Group by episode and step to reconstruct the full sequence
        episodes_data = {}
        
        for path in learning_paths:
            episode = path['episode']
            if episode not in episodes_data:
                episodes_data[episode] = {}
            
            for step_info in path['steps']:
                step = step_info['step']
                if step not in episodes_data[episode]:
                    episodes_data[episode][step] = {
                        'activity_id': step_info['activity_id'],
                        'performance': step_info['performance'],
                        'lesson_impacts': {}
                    }
                
                # Add this lesson's mastery change for this step
                episodes_data[episode][step]['lesson_impacts'][path['lesson_id']] = {
                    'mastery_value': step_info['mastery_value'],
                    'mastery_gain': step_info['mastery_gain']
                }
        
        # Create global learning path file
        filename = f"{student_name}_global_learning_path.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"GLOBAL LEARNING PATH\n")
            f.write(f"Student: {student_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPLETE LEARNING SEQUENCE:\n")
            f.write("(To master all subjects, follow this activity sequence)\n")
            f.write("-" * 60 + "\n\n")
            
            # Collect all activities across episodes to create optimal sequence
            all_activities = []
            activity_impacts = {}
            
            for episode in sorted(episodes_data.keys()):
                episode_data = episodes_data[episode]
                
                f.write(f"EPISODE {episode}:\n")
                
                for step in sorted(episode_data.keys()):
                    step_data = episode_data[step]
                    activity_id = step_data['activity_id']
                    performance = step_data['performance']
                    
                    f.write(f"  {len(all_activities)+1:2d}. {activity_id} (Performance: {performance:.2f})\n")
                    
                    # Show impact on each lesson
                    lesson_impacts = step_data['lesson_impacts']
                    significant_impacts = [(lesson, impact) for lesson, impact in lesson_impacts.items() 
                                         if abs(impact['mastery_gain']) > 0.001]
                    
                    if significant_impacts:
                        f.write("      Mastery impacts:\n")
                        for lesson, impact in significant_impacts:
                            f.write(f"        {lesson}: {impact['mastery_value']:.3f} "
                                   f"({impact['mastery_gain']:+.3f})\n")
                    
                    all_activities.append(activity_id)
                    
                    # Track activity effectiveness
                    if activity_id not in activity_impacts:
                        activity_impacts[activity_id] = {
                            'count': 0,
                            'total_performance': 0.0,
                            'lesson_gains': {}
                        }
                    
                    activity_impacts[activity_id]['count'] += 1
                    activity_impacts[activity_id]['total_performance'] += performance
                    
                    for lesson, impact in lesson_impacts.items():
                        if lesson not in activity_impacts[activity_id]['lesson_gains']:
                            activity_impacts[activity_id]['lesson_gains'][lesson] = []
                        activity_impacts[activity_id]['lesson_gains'][lesson].append(impact['mastery_gain'])
                
                f.write("\n")
            
            # Generate recommended learning sequence
            f.write("RECOMMENDED LEARNING SEQUENCE:\n")
            f.write("-" * 40 + "\n")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sequence = []
            for activity in all_activities:
                if activity not in seen:
                    seen.add(activity)
                    unique_sequence.append(activity)
            
            f.write("Optimal activity sequence (deduplicated):\n")
            for i, activity in enumerate(unique_sequence, 1):
                avg_performance = activity_impacts[activity]['total_performance'] / activity_impacts[activity]['count']
                f.write(f"{i:2d}. {activity} (Avg Performance: {avg_performance:.2f})\n")
            
            f.write(f"\nTotal unique activities: {len(unique_sequence)}\n")
            
            # Activity effectiveness analysis
            f.write("\nACTIVITY EFFECTIVENESS ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Sort activities by average performance
            sorted_activities = sorted(activity_impacts.items(), 
                                     key=lambda x: x[1]['total_performance'] / x[1]['count'], 
                                     reverse=True)
            
            f.write("Most effective activities (by average performance):\n")
            for activity, impact_data in sorted_activities[:10]:  # Top 10
                avg_perf = impact_data['total_performance'] / impact_data['count']
                f.write(f"  {activity}: {avg_perf:.3f} (used {impact_data['count']} times)\n")
                
                # Show which lessons this activity helps most
                best_lessons = []
                for lesson, gains in impact_data['lesson_gains'].items():
                    avg_gain = np.mean([g for g in gains if g > 0])  # Only positive gains
                    if not np.isnan(avg_gain) and avg_gain > 0.001:
                        best_lessons.append((lesson, avg_gain))
                
                if best_lessons:
                    best_lessons.sort(key=lambda x: x[1], reverse=True)
                    f.write(f"    Best for: {', '.join([f'{lesson} ({gain:.3f})' for lesson, gain in best_lessons[:3]])}\n")
        
        print(f"Global learning path saved to: {filepath}")
    
    def _calculate_episode_summary(self, episode_data: List[Dict], episode_num: int, student: Student) -> Dict:
        """Calculate summary statistics for an episode."""
        # Filter out initial state (step 0)
        action_steps = [step for step in episode_data if step['step'] > 0]
        
        if not action_steps:
            return {
                'episode': episode_num,
                'steps_to_completion': 0,
                'final_avg_mastery': 0.0,
                'dominant_style_proportion': 0.0,
                'activities_used': []
            }
        
        # Steps to completion
        steps_to_completion = len(action_steps)
        
        # Final mastery levels
        final_step = action_steps[-1]
        lesson_ids = [l['id'] for l in self.lessons]
        final_masteries = [final_step[f'mastery_{lesson_id}'] for lesson_id in lesson_ids]
        final_avg_mastery = float(np.mean(final_masteries))
        
        # Dominant style analysis
        dominant_style_count = 0
        total_activities = 0
        activities_used = []
        
        for step in action_steps:
            if step['activity_id']:
                activities_used.append(step['activity_id'])
                total_activities += 1
                
                # Parse activity style
                try:
                    activity_style = json.loads(step['activity_style'])
                    dominant_style_value = activity_style.get(student.dominant_style.lower(), 0.0)
                    
                    # Consider it dominant-style focused if the value is above threshold (e.g., 0.4)
                    if dominant_style_value >= 0.4:
                        dominant_style_count += 1
                except:
                    pass
        
        dominant_style_proportion = dominant_style_count / total_activities if total_activities > 0 else 0.0
        
        return {
            'episode': episode_num,
            'steps_to_completion': int(steps_to_completion),
            'final_avg_mastery': final_avg_mastery,
            'dominant_style_proportion': float(dominant_style_proportion),
            'activities_used': activities_used,
            'final_mastery_per_lesson': {lesson_id: float(final_step[f'mastery_{lesson_id}']) for lesson_id in lesson_ids}
        }
    
    def _save_to_csv(self, data: List[Dict], filepath: str):
        """Save episode data to CSV file."""
        if not data:
            return
        
        fieldnames = data[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Detailed log saved to: {filepath}")
    
    def _generate_mastery_plots(self, data: List[Dict], student_name: str):
        """Generate mastery evolution plots for each lesson."""
        df = pd.DataFrame(data)
        lesson_ids = [l['id'] for l in self.lessons]
        
        # Create subplot for each lesson
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, lesson_id in enumerate(lesson_ids):
            if i < len(axes):
                ax = axes[i]
                
                # Plot mastery evolution for each episode
                episodes = df['episode'].unique()
                episode_lines = []
                
                for ep in episodes:
                    ep_data = df[df['episode'] == ep].copy()
                    
                    # Filter out steps where mastery is 0 at the beginning (not started yet)
                    mastery_col = f'mastery_{lesson_id}'
                    
                    # Find first step where mastery starts increasing (> 0.001 to account for floating point)
                    first_increase_idx = None
                    for idx, row in ep_data.iterrows():
                        if row[mastery_col] > 0.001:
                            first_increase_idx = idx
                            break
                    
                    if first_increase_idx is not None:
                        # Filter data from first increase onwards
                        ep_data_filtered = ep_data[ep_data.index >= first_increase_idx].copy()
                        
                        # Find last step before mastery reaches 1.0 (or very close to it)
                        last_meaningful_idx = None
                        for idx, row in ep_data_filtered.iterrows():
                            if row[mastery_col] >= 0.999:  # Close to 1.0
                                last_meaningful_idx = idx
                                break
                        
                        if last_meaningful_idx is not None:
                            # Include only up to the point where mastery reaches 1.0
                            ep_data_final = ep_data_filtered[ep_data_filtered.index <= last_meaningful_idx]
                        else:
                            # If never reaches 1.0, use all filtered data
                            ep_data_final = ep_data_filtered
                        
                        if not ep_data_final.empty:
                            line = ax.plot(ep_data_final['step'], ep_data_final[mastery_col], 
                                        color=colors[i], alpha=0.6, linewidth=1)
                            episode_lines.extend(line)
                
                # Calculate and plot average across episodes (with same filtering logic)
                avg_mastery_data = []
                all_steps = sorted(df['step'].unique())
                
                for step in all_steps:
                    step_data = df[df['step'] == step]
                    
                    # Get mastery values for this step across all episodes
                    mastery_values = []
                    for _, row in step_data.iterrows():
                        mastery_val = row[f'mastery_{lesson_id}']
                        # Only include if mastery has started (> 0.001) or if we've already started tracking this lesson
                        if mastery_val > 0.001 or (avg_mastery_data and avg_mastery_data[-1]['mastery'] > 0.001):
                            mastery_values.append(mastery_val)
                    
                    if mastery_values:
                        avg_mastery = np.mean(mastery_values)
                        avg_mastery_data.append({'step': step, 'mastery': avg_mastery})
                        
                        # Stop if we've reached mastery
                        if avg_mastery >= 0.999:
                            break
                
                # Plot average line
                if avg_mastery_data:
                    avg_steps = [point['step'] for point in avg_mastery_data]
                    avg_masteries = [point['mastery'] for point in avg_mastery_data]
                    
                    avg_line = ax.plot(avg_steps, avg_masteries, 
                                    color=colors[i], linewidth=3, label=f'{lesson_id} (avg)')
                
                # Add mastery threshold line (only in the relevant range)
                if avg_mastery_data:
                    min_step = min([point['step'] for point in avg_mastery_data])
                    max_step = max([point['step'] for point in avg_mastery_data])
                    ax.axhline(y=self.mastery_threshold, color='red', linestyle='--', 
                            alpha=0.7, label=f'Mastery Threshold ({self.mastery_threshold})',
                            xmin=(min_step / max(all_steps)) if all_steps else 0,
                            xmax=(max_step / max(all_steps)) if all_steps else 1)
                
                ax.set_title(f'Mastery Evolution - {lesson_id}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Mastery Level')
                ax.set_ylim([-0.05, 1.05])
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set x-axis limits to focus on relevant range
                if avg_mastery_data:
                    min_step = min([point['step'] for point in avg_mastery_data])
                    max_step = max([point['step'] for point in avg_mastery_data])
                    # Add some padding
                    padding = max(1, (max_step - min_step) * 0.05)
                    ax.set_xlim([min_step - padding, max_step + padding])
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{student_name}_mastery_evolution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mastery evolution plots saved to: {plot_path}")
    
    def _generate_style_analysis(self, data: List[Dict], student: Student, student_name: str):
        """Generate learning style proportion analysis with enhanced visuals."""
        df = pd.DataFrame(data)
        
        # Filter out initial states and get only action steps
        action_df = df[df['step'] > 0].copy()
        
        if action_df.empty:
            return
        
        # Analyze style proportions
        style_counts = {}
        total_activities = 0
        
        for _, row in action_df.iterrows():
            if row['activity_id']:
                total_activities += 1
                
                try:
                    activity_style = json.loads(row['activity_style'])
                    
                    # Find the dominant style for this activity
                    dominant_activity_style = max(activity_style.items(), key=lambda x: x[1])[0]
                    
                    style_counts[dominant_activity_style] = style_counts.get(dominant_activity_style, 0) + 1
                except:
                    style_counts['unknown'] = style_counts.get('unknown', 0) + 1
        
        # Create enhanced pie chart
        if style_counts:
            # Set up the figure with a clean, modern look
            plt.style.use('default')  # Reset to clean style
            fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
            ax.set_facecolor('white')
            
            # Define modern, visually appealing colors
            style_colors = {
                'visual': '#4ECDC4',      # Teal
                'auditory': '#45B7D1',    # Blue
                'kinesthetic': '#F7DC6F', # Yellow
                'read_write': '#96CEB4',  # Mint green
                'unknown': '#D5DBDB'      # Light gray
            }
            
            # Prepare data
            labels = list(style_counts.keys())
            sizes = list(style_counts.values())
            colors = [style_colors.get(label.lower(), '#95A5A6') for label in labels]
            
            # Create explode effect for student's dominant style
            explode = [0.08 if label.lower() == student.dominant_style.lower() else 0 for label in labels]
            
            # Create the pie chart with modern styling
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=None,  # We'll add custom labels
                autopct='%1.1f%%',
                colors=colors,
                explode=explode,
                startangle=90,
                pctdistance=0.85,
                wedgeprops={
                    'edgecolor': 'white',
                    'linewidth': 2,
                    'antialiased': True
                },
                textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'}
            )
            
            # Enhance the wedge appearance
            for wedge in wedges:
                wedge.set_linewidth(2)
                wedge.set_edgecolor('white')
                
            # Create custom legend with better positioning
            legend_labels = [f'{label.replace("_", " ").title()}' for label in labels]
            legend = ax.legend(
                wedges, 
                legend_labels,
                title="Learning Styles",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=11,
                title_fontsize=12,
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9
            )
            legend.get_title().set_fontweight('bold')
            
            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
                # Add subtle shadow effect to text
                autotext.set_path_effects([
                    patheffects.withStroke(linewidth=3, foreground='black', alpha=0.3)
                ])
            
            # Add a sophisticated title
            title_text = f'Learning Style Distribution\nStudent Profile: {student.dominant_style.replace("_", " ").title()} Learner ({student.dominant_percent}%)'
            ax.set_title(
                title_text,
                fontsize=16,
                fontweight='bold',
                pad=30,
                color='#2C3E50'
            )
            
            # Add a subtle circle in the center for donut effect
            centre_circle = plt.Circle((0, 0), 0.60, fc='white', ec='#ECF0F1', linewidth=2)
            ax.add_artist(centre_circle)
            
            # Add center text with key statistics
            center_text = f'Total\nActivities\n{total_activities}'
            ax.text(0, 0, center_text, horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontweight='bold', color='#34495E')
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add a subtle border around the entire plot
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Adjust layout to prevent clipping
            plt.tight_layout()
            
            # Save with high quality
            plot_path = os.path.join(self.output_dir, f"{student_name}_style_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', 
                    pad_inches=0.2)
            plt.close()
            
            print(f"Enhanced style analysis plot saved to: {plot_path}")

def main():
    """Main testing function with example usage."""
    
    # Configuration
    # model_path = "models/adaptive_learning_model_200k"  # Update this path
    model_path = "models/adaptive_learning_model_50k_performance_new_dtst"  # Update this path
    curriculum_path = "data/learning_curriculum.json"  # Update this path
    output_dir = "test_results_perf/"
    mastery_threshold = 1.0  # Threshold to consider a lesson as mastered
    
    # Initialize tester
    tester = ModelTester(model_path, curriculum_path, output_dir, mastery_threshold)
    
    # Test with different student profiles
    test_students = [
        Student(dominant_style="Visual", dominant_percent=80, velocity=0.9),
        # Student(dominant_style="Auditory", dominant_percent=75, velocity=0.8),
        # Student(dominant_style="Kinesthetic", dominant_percent=70, velocity=0.85),
        # Student(dominant_style="Read/Write", dominant_percent=85, velocity=0.95),
    ]
    
    student_names = [
        "Visual_80pct_Student",
        # "Auditory_75pct_Student", 
        # "Kinesthetic_70pct_Student",
        # "Reading_85pct_Student"
    ]
    
    all_results = {}
    
    for student, name in zip(test_students, student_names):
        results = tester.test_student(student, num_episodes=1, student_name=name)
        all_results[name] = results
        
        print(f"\n--- Results for {name} ---")
        print(f"Average steps to completion: {results['avg_steps_to_completion']:.1f}")
        print(f"Average final mastery: {results['avg_final_mastery']:.3f}")
        print(f"Dominant style proportion: {results['avg_dominant_style_proportion']:.3f}")
    
    # Save combined results
    combined_path = os.path.join(output_dir, "combined_test_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Combined results: {combined_path}")
    print(f"\nNew files generated:")
    print(f"- Individual lesson learning paths: *_learning_path.txt")
    print(f"- Global learning paths: *_global_learning_path.txt")

if __name__ == "__main__":
    main()