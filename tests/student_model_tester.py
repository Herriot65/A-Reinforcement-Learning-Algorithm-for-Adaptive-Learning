import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib import patheffects
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from envs.Student import Student
from envs.AdaptiveLearningEnv import AdaptiveLearningEnv
from training.environment_setup import mask_fn
class ModelTester:
    def __init__(self, model_path: str, curriculum_path: str, output_dir: str = "tests/", mastery_threshold: float = 1.0):
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
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.lessons, self.activities = self._load_curriculum()
        
        self.model = MaskablePPO.load(model_path)
        
    def _load_curriculum(self) -> Tuple[List[Dict], List[Dict]]:
        """Load curriculum from JSON file."""
        with open(self.curriculum_path) as f:
            data = json.load(f)
        return data["lessons"], data["activities"]
    
    def test_student(self, student: Student, num_episodes: int = 1, student_name: str = None, deterministic: bool = True) -> Dict:
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
        
        all_episodes_data = []
        episode_summaries = []
        all_learning_paths = []
        
        for episode in range(num_episodes):
            print(f"Running episode {episode + 1}/{num_episodes}...")
            
            env = AdaptiveLearningEnv(
                lessons=self.lessons,
                activities=self.activities,
                student=student,
                log_training_data=False,
                
            )
            masked_env = ActionMasker(env, mask_fn)
            
            episode_data = self._run_episode(masked_env, episode + 1, deterministic=deterministic)
            all_episodes_data.extend(episode_data)
            
            episode_learning_paths = self._generate_learning_paths(episode_data, episode + 1, student_name)
            all_learning_paths.extend(episode_learning_paths)
            
            episode_summary = self._calculate_episode_summary(episode_data, episode + 1, student)
            episode_summaries.append(episode_summary)
            
            masked_env.close()
        
        csv_path = os.path.join(self.output_dir, f"{student_name}_detailed_log.csv")
        self._save_to_csv(all_episodes_data, csv_path)
        
        self._save_learning_paths(all_learning_paths, student_name)
        self._generate_global_learning_path(all_learning_paths, student_name)
        self._generate_mastery_plots(all_episodes_data, student_name)
        self._generate_style_analysis(all_episodes_data, student, student_name)
                
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
            'csv_file': csv_path
        }
        
        summary_path = os.path.join(self.output_dir, f"{student_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def _run_episode(self, env, episode_num: int, deterministic: bool = True) -> List[Dict]:
        """Run a single episode and collect data."""
        episode_data = []
        obs, info = env.reset()
        step = 0
        
        initial_state = {
            'episode': episode_num,
            'step': step,
            'activity_id': None,
            'activity_style': None,
            'performance': 0.0,
            'reward': 0.0,
            'terminated': False
        }
        
        for i, lesson_id in enumerate([l['id'] for l in self.lessons]):
            initial_state[f'mastery_{lesson_id}'] = float(obs[i])
            initial_state[f'mastery_gain_{lesson_id}'] = 0.0
        
        episode_data.append(initial_state)
        
        terminated = False
        truncated = False
        previous_obs = obs.copy()
        
        while not terminated and not truncated:
            action_masks = info.get('available_actions', None)
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            activity_id = info.get('activity_id')
            activity = self.activities[action]
            
            step_data = {
                'episode': episode_num,
                'step': step,
                'activity_id': activity_id,
                'activity_style': json.dumps(activity.get('style', {})),
                'performance': round(info.get('performance', 0.0), 2) if info.get('performance', 0.0) is not None else info.get('performance', 0.0),
                'reward': round(reward, 2) if reward is not None else reward,
                'terminated': terminated
            }
            
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
            lesson_paths = [path for path in learning_paths if path['lesson_id'] == lesson_id]
            
            if not lesson_paths:
                continue
            
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
                    
                    f.write("Step-by-step progression (only showing steps with mastery gain):\n")
                    for step_info in episode_path['steps']:
                        if step_info['mastery_gain'] > 0.001:  
                            f.write(f"  Step {step_info['step']:2d}: "
                                   f"Activity {step_info['activity_id']:15s} -> "
                                   f"Mastery: {step_info['mastery_value']:6.3f} "
                                   f"(+{step_info['mastery_gain']:+6.3f}) "
                                   f"Performance: {step_info['performance']:5.2f}\n")
                    
                    f.write("\n")
                
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
        """Generate a global learning path combining all lessons"""
        
        activity_performances = {}
        activity_counts = {}
        
        for path in learning_paths:
            for step_info in path['steps']:
                activity_id = step_info['activity_id']
                performance = step_info['performance']
                
                if activity_id not in activity_performances:
                    activity_performances[activity_id] = 0.0
                    activity_counts[activity_id] = 0
                
                activity_performances[activity_id] += performance
                activity_counts[activity_id] += 1
        
        avg_activity_performance = {
            act_id: activity_performances[act_id] / activity_counts[act_id]
            for act_id in activity_performances
        }
        
        sorted_activities = sorted(avg_activity_performance.items(), key=lambda item: item[1], reverse=True)
        
        filename = f"{student_name}_global_learning_path.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"GLOBAL LEARNING PATH - RECOMMENDED ACTIVITIES\n")
            f.write(f"Student: {student_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Activities and their Average Performance:\n")
            f.write("-" * 50 + "\n")
            for activity_id, avg_perf in sorted_activities:
                f.write(f"{activity_id:30s} : {avg_perf:.3f}\n")
            f.write("-" * 50 + "\n")
            f.write(f"\nTotal unique activities encountered: {len(sorted_activities)}\n")
        
        print(f"Global learning path saved to: {filepath}")
    
    def _calculate_episode_summary(self, episode_data: List[Dict], episode_num: int, student: Student) -> Dict:
        """Calculate summary statistics for an episode."""
        
        action_steps = [step for step in episode_data if step['step'] > 0]
        
        if not action_steps:
            return {
                'episode': episode_num,
                'steps_to_completion': 0,
                'final_avg_mastery': 0.0,
                'activities_used': []
            }
        
        steps_to_completion = len(action_steps)
        
        final_step = action_steps[-1]
        lesson_ids = [l['id'] for l in self.lessons]
        final_masteries = [final_step[f'mastery_{lesson_id}'] for lesson_id in lesson_ids]
        final_avg_mastery = float(np.mean(final_masteries))
        
        activities_used = []
        
        for step in action_steps:
            if step['activity_id']:
                activities_used.append(step['activity_id'])
        
        return {
            'episode': episode_num,
            'steps_to_completion': int(steps_to_completion),
            'final_avg_mastery': final_avg_mastery,
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
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, lesson_id in enumerate(lesson_ids):
            if i < len(axes):
                ax = axes[i]
                
                episodes = df['episode'].unique()
                episode_lines = []
                
                for ep in episodes:
                    ep_data = df[df['episode'] == ep].copy()
                    
                    mastery_col = f'mastery_{lesson_id}'
                    
                    first_increase_idx = None
                    for idx, row in ep_data.iterrows():
                        if row[mastery_col] > 0.001:
                            first_increase_idx = idx
                            break
                    
                    if first_increase_idx is not None:
                        ep_data_filtered = ep_data[ep_data.index >= first_increase_idx].copy()
                        
                        last_meaningful_idx = None
                        for idx, row in ep_data_filtered.iterrows():
                            if row[mastery_col] >= 0.999:  
                                last_meaningful_idx = idx
                                break
                        
                        if last_meaningful_idx is not None:
                            ep_data_final = ep_data_filtered[ep_data_filtered.index <= last_meaningful_idx]
                        else:
                            ep_data_final = ep_data_filtered
                        
                        if not ep_data_final.empty:
                            line = ax.plot(ep_data_final['step'], ep_data_final[mastery_col], 
                                        color=colors[i], alpha=0.6, linewidth=1)
                            episode_lines.extend(line)
                
                avg_mastery_data = []
                all_steps = sorted(df['step'].unique())
                
                for step in all_steps:
                    step_data = df[df['step'] == step]
                    
                    mastery_values = []
                    for _, row in step_data.iterrows():
                        mastery_val = row[f'mastery_{lesson_id}']
                        if mastery_val > 0.001 or (avg_mastery_data and avg_mastery_data[-1]['mastery'] > 0.001):
                            mastery_values.append(mastery_val)
                    
                    if mastery_values:
                        avg_mastery = np.mean(mastery_values)
                        avg_mastery_data.append({'step': step, 'mastery': avg_mastery})
                        
                        if avg_mastery >= 0.999:
                            break
                
                if avg_mastery_data:
                    avg_steps = [point['step'] for point in avg_mastery_data]
                    avg_masteries = [point['mastery'] for point in avg_mastery_data]
                    
                    avg_line = ax.plot(avg_steps, avg_masteries, 
                                    color=colors[i], linewidth=3, label=f'{lesson_id} (avg)')
                
                if avg_mastery_data:
                    min_step = min([point['step'] for point in avg_mastery_data])
                    max_step = max([point['step'] for point in avg_mastery_data])
                    padding = max(1, (max_step - min_step) * 0.05)
                    ax.set_xlim([min_step - padding, max_step + padding])
                
                ax.set_title(f'Mastery Evolution - {lesson_id}')
                ax.set_xlabel('Activiies')
                ax.set_ylabel('Mastery Level')
                ax.set_ylim([-0.05, 1.05])
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{student_name}_mastery_evolution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mastery evolution plots saved to: {plot_path}")
    
    def _generate_style_analysis(self, data: List[Dict], student: Student, student_name: str):
        """Generate learning style proportion analysis with enhanced visuals."""
        df = pd.DataFrame(data)
        
        action_df = df[df['step'] > 0].copy()
        
        if action_df.empty:
            return
        
        style_counts = {}
        total_activities = 0
        
        for _, row in action_df.iterrows():
            if row['activity_id']:
                total_activities += 1
                
                try:
                    activity_style = json.loads(row['activity_style'])
                    
                    dominant_activity_style = max(activity_style.items(), key=lambda x: x[1])[0]
                    
                    style_counts[dominant_activity_style] = style_counts.get(dominant_activity_style, 0) + 1
                except:
                    style_counts['unknown'] = style_counts.get('unknown', 0) + 1
        
        if style_counts:
            plt.style.use('default')
            _, ax = plt.subplots(figsize=(12, 9), facecolor='white')
            ax.set_facecolor('white')
            
            style_colors = {
                'visual': '#4ECDC4',      
                'auditory': '#45B7D1',    
                'kinesthetic': '#F7DC6F', 
                'read_write': '#96CEB4',  
                'unknown': '#D5DBDB'     
            }
            
            labels = list(style_counts.keys())
            sizes = list(style_counts.values())
            colors = [style_colors.get(label.lower(), '#95A5A6') for label in labels]
            
            explode = [0.08 if label.lower() == student.dominant_style.lower() else 0 for label in labels]
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=None,  
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
        
            for wedge in wedges:
                wedge.set_linewidth(2)
                wedge.set_edgecolor('white')
                
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
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
                autotext.set_path_effects([
                    patheffects.withStroke(linewidth=3, foreground='black', alpha=0.3)
                ])
            
            title_text = f'Learning Style Distribution\nStudent Profile: {student.dominant_style.replace("_", " ").title()} Learner ({student.dominant_percent}% dominant, velocity={int(student.velocity*100)}%)'
            ax.set_title(
                title_text,
                fontsize=16,
                fontweight='bold',
                pad=30,
                color='#2C3E50'
            )
            
            centre_circle = plt.Circle((0, 0), 0.60, fc='white', ec='#ECF0F1', linewidth=2)
            ax.add_artist(centre_circle)
        
            center_text = f'Total\nActivities\n{total_activities}'
            ax.text(0, 0, center_text, horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontweight='bold', color='#34495E')
            
            ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, f"{student_name}_style_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', 
                    pad_inches=0.2)
            plt.close()
            
            print(f"Enhanced style analysis plot saved to: {plot_path}")

def main(model_path: str , output_dir: str, test_students: str, mastery_threshold: float = 1.0, 
         curriculum_path: str = "data/learning_curriculum.json", num_episodes: int = 1, deterministic: bool = True):
    """Main testing function with example usage."""
    
    tester = ModelTester(model_path, curriculum_path, output_dir, mastery_threshold)
    
    test_students = test_students
    
    student_names = [
        f"{student.dominant_style.replace('/', '_')}_{student.dominant_percent}pct_{student.velocity}vel_Student"
        for student in test_students
    ]
 
    for student, name in zip(test_students, student_names):
         tester.test_student(student, num_episodes = num_episodes, student_name = name, deterministic = deterministic)

if __name__ == "__main__":
    model_path="models/student_model_Read_Write_85pct_09vel"
    output_dir="tests/model_test_results/student_model_Read_Write_85pct_09vel_1" #add _1 just for testing
    main(model_path=model_path,output_dir = output_dir,
         test_students=[Student(dominant_style = "Read/Write", dominant_percent = 85, velocity = 0.9)], deterministic = True
         )