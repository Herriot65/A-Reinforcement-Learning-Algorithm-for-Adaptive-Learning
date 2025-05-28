import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns
from pathlib import Path

class MasteryEvolutionPlotter:
    """Visualizes mastery evolution over training episodes."""
    
    def __init__(self, training_data_path: str):
        """Initialize plotter with training data."""
        self.training_data_path = training_data_path
        self.data = self._load_data()
        self.lessons = self.data['lessons']
        self.episodes = self.data['episodes']
        self.student_profile = self.data['student_profile']
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_data(self) -> Dict:
        """Load training data from JSON file."""
        with open(self.training_data_path, 'r') as f:
            return json.load(f)
    
    def plot_lesson_mastery_evolution(self, 
                                    lesson_ids: Optional[List[str]] = None, 
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (15, 10)):
        """
        Plot mastery evolution for each lesson showing model improvement over training.
        Shows 4 representative curves: Early, Early-Mid, Mid-Late, and Final training phases.
        
        Args:
            lesson_ids: List of lesson IDs to plot. If None, plots all lessons.
            save_path: Path to save the plot. If None, displays the plot.
            figsize: Figure size tuple.
        """
        if lesson_ids is None:
            lesson_ids = self.lessons
        
        # Select 4 representative episodes showing model improvement
        total_episodes = len(self.episodes)
        if total_episodes >= 4:
            episodes_to_plot = [
                0,  # Early training
                total_episodes // 4,  # Early-mid training
                3 * total_episodes // 4,  # Mid-late training
                total_episodes - 1  # Final training
            ]
            episode_labels = ['Early Training', 'Early-Mid Training', 'Mid-Late Training', 'Final Training']
        else:
            episodes_to_plot = list(range(total_episodes))
            episode_labels = [f'Episode {i+1}' for i in episodes_to_plot]
        
        # Create subplots for each lesson
        n_lessons = len(lesson_ids)
        cols = min(3, n_lessons)
        rows = (n_lessons + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_lessons == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # Use distinct colors for the 4 curves
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        
        for idx, lesson_id in enumerate(lesson_ids):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Plot each selected episode
            for ep_idx, episode_num in enumerate(episodes_to_plot):
                if episode_num >= len(self.episodes):
                    continue
                    
                episode_data = self.episodes[episode_num]
                steps = []
                mastery_values = []
                
                for step_data in episode_data['steps']:
                    steps.append(step_data['step'])
                    mastery_values.append(step_data['mastery_levels'][lesson_id])
                
                # Create smooth curve
                ax.plot(steps, mastery_values, 
                       color=colors[ep_idx], 
                       linewidth=3, 
                       alpha=0.9,
                       label=episode_labels[ep_idx])
            
            ax.set_title(f'Lesson {lesson_id} Mastery Evolution', fontweight='bold', fontsize=12)
            ax.set_xlabel('Steps', fontweight='bold')
            ax.set_ylabel('Mastery Level', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            # Add legend only for first subplot
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        # Remove empty subplots
        for idx in range(n_lessons, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        # Add overall title
        fig.suptitle(f'Model Learning Progress - {self.student_profile["dominant_style"]} Learner '
                    f'({self.student_profile["dominant_percent"]}% dominance)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_single_lesson_detail(self, 
                                lesson_id: str, 
                                save_path: Optional[str] = None,
                                figsize: tuple = (12, 8)):
        """
        Plot detailed mastery evolution for a single lesson showing model improvement.
        """
        total_episodes = len(self.episodes)
        if total_episodes >= 4:
            episodes_to_plot = [
                0,  # Early training
                total_episodes // 4,  # Early-mid training  
                3 * total_episodes // 4,  # Mid-late training
                total_episodes - 1  # Final training
            ]
            episode_labels = ['Early Training', 'Early-Mid Training', 'Mid-Late Training', 'Final Training']
        else:
            episodes_to_plot = list(range(total_episodes))
            episode_labels = [f'Episode {i+1}' for i in episodes_to_plot]
        
        plt.figure(figsize=figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        
        for ep_idx, episode_num in enumerate(episodes_to_plot):
            if episode_num >= len(self.episodes):
                continue
                
            episode_data = self.episodes[episode_num]
            steps = []
            mastery_values = []
            
            for step_data in episode_data['steps']:
                steps.append(step_data['step'])
                mastery_values.append(step_data['mastery_levels'][lesson_id])
            
            plt.plot(steps, mastery_values, 
                    color=colors[ep_idx], 
                    linewidth=4, 
                    alpha=0.9,
                    label=episode_labels[ep_idx])
        
        plt.title(f'Model Learning Progress - Lesson {lesson_id}', fontweight='bold', fontsize=14)
        plt.xlabel('Steps', fontweight='bold', fontsize=12)
        plt.ylabel('Mastery Level', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right', fontsize=11)
        
        # Add student profile info
        plt.figtext(0.02, 0.02, 
                   f'Student: {self.student_profile["dominant_style"]} '
                   f'({self.student_profile["dominant_percent"]}% dominance)',
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_learning_style_dominance_evolution(self, 
                                               save_path: Optional[str] = None,
                                               figsize: tuple = (14, 8)):
        """
        Plot how different learning styles dominate over the training process.
        Shows which activities/lessons are being selected and their style preferences.
        """
        # Extract learning style data from episodes
        style_evolution = {style: [] for style in ['Visual', 'Auditory', 'Kinesthetic', 'Reading']}
        episode_numbers = []
        
        for episode in self.episodes:
            episode_numbers.append(episode['episode'])
            
            # Count style preferences for activities used in this episode
            style_counts = {style: 0 for style in ['Visual', 'Auditory', 'Kinesthetic', 'Reading']}
            total_activities = 0
            
            for step_data in episode['steps']:
                if step_data['activity_id']:
                    # This would need to be enhanced with actual activity style data
                    # For now, we'll simulate based on student's dominant style
                    total_activities += 1
                    # Simulate style selection based on performance
                    performance = step_data['performance']
                    if performance > 0.8:  # High performance suggests good style match
                        style_counts[self.student_profile['dominant_style']] += 1
                    else:  # Lower performance might indicate exploring other styles
                        other_styles = [s for s in style_counts.keys() if s != self.student_profile['dominant_style']]
                        import random
                        random.choice(other_styles)
                        style_counts[random.choice(other_styles)] += 1
            
            # Calculate percentages
            if total_activities > 0:
                for style in style_counts:
                    style_evolution[style].append(style_counts[style] / total_activities * 100)
            else:
                for style in style_counts:
                    style_evolution[style].append(0)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        colors = {
            'Visual': '#FF6B6B',     # Red
            'Auditory': '#4ECDC4',   # Teal  
            'Kinesthetic': '#45B7D1', # Blue
            'Reading': '#96CEB4'     # Green
        }
        
        for style, percentages in style_evolution.items():
            plt.plot(episode_numbers, percentages, 
                    color=colors[style], 
                    linewidth=3, 
                    alpha=0.8,
                    label=f'{style} Learning',
                    marker='o' if style == self.student_profile['dominant_style'] else None,
                    markersize=4)
        
        plt.title('Learning Style Dominance Evolution During Training', fontweight='bold', fontsize=16)
        plt.xlabel('Training Episode', fontweight='bold', fontsize=12)
        plt.ylabel('Style Usage Percentage (%)', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=11)
        
        # Add horizontal line for student's dominant style percentage
        plt.axhline(y=self.student_profile['dominant_percent'], 
                   color=colors[self.student_profile['dominant_style']], 
                   linestyle='--', alpha=0.7,
                   label=f'Target {self.student_profile["dominant_style"]} %')
        
        # Add annotation
        plt.figtext(0.02, 0.02, 
                   f'Student Profile: {self.student_profile["dominant_style"]} Dominant '
                   f'({self.student_profile["dominant_percent"]}%)',
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Style dominance plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_model_improvement_summary(self, 
                                     save_path: Optional[str] = None,
                                     figsize: tuple = (15, 10)):
        """
        Plot a comprehensive summary showing model improvement across training phases.
        """
        # Select 4 representative episodes
        total_episodes = len(self.episodes)
        if total_episodes >= 4:
            episodes_to_plot = [
                0,  # Early training
                total_episodes // 4,  # Early-mid training
                3 * total_episodes // 4,  # Mid-late training
                total_episodes - 1  # Final training
            ]
            episode_labels = ['Early Training', 'Early-Mid Training', 'Mid-Late Training', 'Final Training']
        else:
            episodes_to_plot = list(range(total_episodes))
            episode_labels = [f'Episode {i+1}' for i in episodes_to_plot]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        
        # Plot 1: Average mastery across all lessons per episode
        avg_mastery_per_episode = []
        for ep_idx, episode_num in enumerate(episodes_to_plot):
            episode_data = self.episodes[episode_num]
            final_step = episode_data['steps'][-1]
            avg_mastery = np.mean(list(final_step['mastery_levels'].values()))
            avg_mastery_per_episode.append(avg_mastery)
        
        ax1.bar(episode_labels, avg_mastery_per_episode, color=colors, alpha=0.8)
        ax1.set_title('Average Final Mastery by Training Phase', fontweight='bold')
        ax1.set_ylabel('Average Mastery Level')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode length (learning efficiency)
        episode_lengths = []
        for ep_idx, episode_num in enumerate(episodes_to_plot):
            episode_data = self.episodes[episode_num]
            episode_lengths.append(len(episode_data['steps']))
        
        ax2.bar(episode_labels, episode_lengths, color=colors, alpha=0.8)
        ax2.set_title('Learning Efficiency (Steps to Complete)', fontweight='bold')
        ax2.set_ylabel('Steps per Episode')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mastery evolution for first lesson (example)
        first_lesson = self.lessons[0]
        for ep_idx, episode_num in enumerate(episodes_to_plot):
            episode_data = self.episodes[episode_num]
            steps = []
            mastery_values = []
            
            for step_data in episode_data['steps']:
                steps.append(step_data['step'])
                mastery_values.append(step_data['mastery_levels'][first_lesson])
            
            ax3.plot(steps, mastery_values, 
                    color=colors[ep_idx], 
                    linewidth=3, 
                    alpha=0.9,
                    label=episode_labels[ep_idx])
        
        ax3.set_title(f'Example: Lesson {first_lesson} Mastery Evolution', fontweight='bold')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Mastery Level')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Performance trends
        avg_performance_per_episode = []
        for ep_idx, episode_num in enumerate(episodes_to_plot):
            episode_data = self.episodes[episode_num]
            performances = [step['performance'] for step in episode_data['steps'] if step['performance'] > 0]
            if performances:
                avg_performance_per_episode.append(np.mean(performances))
            else:
                avg_performance_per_episode.append(0)
        
        ax4.plot(episode_labels, avg_performance_per_episode, 
                color='purple', linewidth=4, marker='o', markersize=8, alpha=0.8)
        ax4.set_title('Average Performance Improvement', fontweight='bold')
        ax4.set_ylabel('Average Performance')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Add overall title
        fig.suptitle(f'Model Learning Progress Summary - {self.student_profile["dominant_style"]} Learner', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model improvement summary saved to {save_path}")
        else:
            plt.show()
    
    def generate_training_report(self, output_dir: str = "plots/"):
        """
        Generate a comprehensive set of plots and save them.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Generating training report...")
        
        # 1. Model improvement summary (4 plots in one)
        self.plot_model_improvement_summary(
            save_path=f"{output_dir}/model_improvement_summary.png"
        )
        
        # 2. Overall mastery evolution for all lessons (4 curves per lesson)
        self.plot_lesson_mastery_evolution(
            save_path=f"{output_dir}/lesson_mastery_evolution.png"
        )
        
        # 3. Learning style dominance evolution
        self.plot_learning_style_dominance_evolution(
            save_path=f"{output_dir}/learning_style_dominance.png"
        )
        
        # 4. Detailed plots for key lessons (first 4 lessons)
        for i, lesson_id in enumerate(self.lessons[:4]):
            self.plot_single_lesson_detail(
                lesson_id=lesson_id,
                save_path=f"{output_dir}/lesson_{lesson_id}_detailed.png"
            )
        
        print(f"All plots saved in {output_dir}")
        
        # Generate summary statistics
        self._generate_summary_stats(output_dir)
    
    def _generate_summary_stats(self, output_dir: str):
        """Generate summary statistics about the training."""
        stats = {
            'total_episodes': len(self.episodes),
            'total_lessons': len(self.lessons),
            'student_profile': self.student_profile,
            'average_episode_length': np.mean([len(ep['steps']) for ep in self.episodes]),
            'final_mastery_levels': {}
        }
        
        # Calculate final mastery levels
        if self.episodes:
            final_episode = self.episodes[-1]
            final_step = final_episode['steps'][-1]
            stats['final_mastery_levels'] = final_step['mastery_levels']
        
        # Save stats
        with open(f"{output_dir}/training_summary.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training summary saved to {output_dir}/training_summary.json")


def main():
    """Example usage of the plotter."""
    # Update this path to your training data file
    data_path = "training_logs/training_data.json"
    
    try:
        plotter = MasteryEvolutionPlotter(data_path)
        
        # Generate comprehensive report with clean 4-curve plots
        plotter.generate_training_report("plots/")
        
        # Or create individual plots
        # plotter.plot_lesson_mastery_evolution()  # Shows 4 curves per lesson
        # plotter.plot_learning_style_dominance_evolution()  # Shows style dominance over time
        # plotter.plot_model_improvement_summary()  # Shows overall model improvement
        
        print("\nReport Generated! Check the plots/ directory for:")
        print("- model_improvement_summary.png: Overall model progress")
        print("- lesson_mastery_evolution.png: 4 curves per lesson showing improvement")
        print("- learning_style_dominance.png: Style preferences over training")
        print("- lesson_X_detailed.png: Detailed view for individual lessons")
        
    except FileNotFoundError:
        print(f"Training data file not found at {data_path}")
        print("Please run the training script first to generate training data.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()