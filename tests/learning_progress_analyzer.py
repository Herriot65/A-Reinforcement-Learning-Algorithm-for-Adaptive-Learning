import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


class LearningProgressConfig:
    """Configuration class for learning progress analysis."""
    
    def __init__(self, 
                 training_data_file: str = "training_data/",
                 plots_folder: str = "tests/",
                 mastery_target: float = 1.0,
                 improvement_threshold: float = 0.01,
                 mastery_thresholds: List[float] = None):
        self.training_data_file = training_data_file
        self.plots_folder = plots_folder
        self.mastery_target = mastery_target
        self.improvement_threshold = improvement_threshold
        self.mastery_thresholds = mastery_thresholds or [0.99, 0.95, 0.90]
        
        # Phase definitions (start%, end%)
        self.phases = {
            "Early": (0.0, 0.1),
            "Middle": (0.45, 0.55), 
            "Late": (0.9, 1.0)
        }
        
        # Plotting configuration
        self.colors = {"Early": "red", "Middle": "orange", "Late": "green"}
        self.figure_size = (15, 12)


class DataLoader:
    """Handles loading and basic validation of training data."""
    
    @staticmethod
    def load_training_data(file_path: str) -> Dict:
        """Load training data from JSON file with error handling."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data file '{file_path}' not found. "
                                  "Please run your train.py script to generate this file.")
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from '{file_path}'. "
                           "The file might be corrupted.")
    
    @staticmethod
    def extract_lessons(episodes: List[Dict]) -> List[str]:
        """Extract unique lesson IDs from episode data."""
        unique_lesson_ids = set()
        for episode_data in episodes:
            for step_data in episode_data['steps']:
                for lesson_id in step_data['mastery_levels'].keys():
                    unique_lesson_ids.add(lesson_id)
        return sorted(list(unique_lesson_ids))


class PhaseAnalyzer:
    """Analyzes learning progress across different training phases."""
    
    def __init__(self, config: LearningProgressConfig):
        self.config = config
    
    def extract_phase_indices(self, total_episodes: int, phase_percent: Tuple[float, float]) -> range:
        """Extract episode indices for a given phase percentage."""
        start = int(total_episodes * phase_percent[0])
        end = int(total_episodes * phase_percent[1])
        return range(start, min(end, total_episodes))
    
    def get_phase_indices(self, total_episodes: int) -> Dict[str, range]:
        """Get episode indices for all defined phases."""
        return {
            phase_name: self.extract_phase_indices(total_episodes, phase_range)
            for phase_name, phase_range in self.config.phases.items()
        }
    
    def calculate_phase_curves(self, episodes: List[Dict], lessons: List[str]) -> Dict:
        """Calculate mastery curves for each lesson across all phases."""
        total_episodes = len(episodes)
        phases_indices = self.get_phase_indices(total_episodes)
        
        # Initialize data structure
        phase_curves = {
            lesson: {phase: [] for phase in self.config.phases.keys()} 
            for lesson in lessons
        }
        
        for phase_name, indices in phases_indices.items():
            for lesson in lessons:
                all_mastery = []
                
                # Collect mastery data for current phase
                for idx in indices:
                    if idx < len(episodes):
                        episode = episodes[idx]
                        mastery_per_step = [
                            step_entry["mastery_levels"].get(lesson, 0.0)
                            for step_entry in episode["steps"]
                        ]
                        all_mastery.append(mastery_per_step)
                
                # Calculate average mastery per step
                if all_mastery:
                    max_steps = max(len(m) for m in all_mastery)
                    avg_mastery_per_step = []
                    
                    for step_i in range(max_steps):
                        step_values = [m[step_i] for m in all_mastery if len(m) > step_i]
                        avg = np.mean(step_values) if step_values else np.nan
                        avg_mastery_per_step.append(avg)
                    
                    phase_curves[lesson][phase_name] = avg_mastery_per_step
        
        return phase_curves


class MasteryAnalyzer:
    """Analyzes mastery achievement patterns."""
    
    def __init__(self, config: LearningProgressConfig):
        self.config = config
    
    def find_improvement_start(self, mastery_data: np.ndarray) -> Optional[int]:
        """Find the first step where mastery starts improving."""
        improvement_indices = np.where(mastery_data > self.config.improvement_threshold)[0]
        return improvement_indices[0] if len(improvement_indices) > 0 else None
    
    def find_mastery_achievement(self, mastery_data: np.ndarray, start_idx: int) -> Optional[int]:
        """Find the first step where mastery target is reached after improvement starts."""
        for threshold in self.config.mastery_thresholds:
            for step_idx in range(start_idx, len(mastery_data)):
                if mastery_data[step_idx] >= threshold:
                    return step_idx
        return None
    
    def get_plot_data(self, mastery_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Get x and y data for plotting, along with mastery achievement point."""
        if len(mastery_data) == 0:
            return None, None, None
        
        start_idx = self.find_improvement_start(mastery_data)
        if start_idx is None:
            return None, None, None
        
        mastery_idx = self.find_mastery_achievement(mastery_data, start_idx)
        end_idx = (mastery_idx + 1) if mastery_idx is not None else len(mastery_data)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(mastery_data))
        
        if start_idx >= end_idx:
            return None, None, None
        
        x_plot = np.arange(start_idx, end_idx)
        y_plot = mastery_data[start_idx:end_idx]
        
        return x_plot, y_plot, mastery_idx

class LearningProgressVisualizer:
    """Handles visualization of learning progress data."""
    def __init__(self, config: LearningProgressConfig):
        self.config = config
        self.mastery_analyzer = MasteryAnalyzer(config)
    
    def create_plots_folder(self):
        """Create plots folder if it doesn't exist."""
        os.makedirs(self.config.plots_folder, exist_ok=True)
    
    def plot_lesson_progress(self, ax, lesson: str, phase_curves: Dict, verbose: bool = False):
        """Plot progress for a single lesson across all phases."""
        all_x_values = []
        
        for phase in self.config.phases.keys():
            mastery_data = np.array(phase_curves[lesson][phase])
            
            if len(mastery_data) == 0:
                continue
            
            x_plot, y_plot, mastery_idx = self.mastery_analyzer.get_plot_data(mastery_data)
            
            if x_plot is None or len(x_plot) == 0:
                continue
            
            # Debug information
            if verbose:
                max_mastery = np.max(mastery_data)
                start_idx = self.mastery_analyzer.find_improvement_start(mastery_data)                
                if mastery_idx is not None:
                    print(f"  Found mastery at step {mastery_idx} with value {mastery_data[mastery_idx]:.3f}")
            
            ax.plot(x_plot, y_plot, label=phase, color=self.config.colors[phase], linewidth=2)
            
            # Add mastery achievement marker
            if mastery_idx is not None:
                ax.plot(mastery_idx, mastery_data[mastery_idx], 'o',
                       color=self.config.colors[phase], markersize=8, alpha=0.8,
                       markeredgecolor='black', markeredgewidth=1)
            
            all_x_values.extend(x_plot)
        
        self._configure_lesson_axes(ax, lesson, all_x_values)
    
    def _configure_lesson_axes(self, ax, lesson: str, all_x_values: List[int]):
        """Configure axes for a lesson plot."""
        ax.set_title(f"Lesson {lesson}", fontsize=14)
        ax.set_xlabel("Activities (within episode)", fontsize=12)
        ax.set_ylabel("Average Mastery", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Training Phase", fontsize=10, title_fontsize=11)
        
        if all_x_values:
            min_x, max_x = min(all_x_values), max(all_x_values)
            tick_start = (min_x // 25) * 25
            tick_end = ((max_x // 25) + 1) * 25
            ax.set_xticks(np.arange(tick_start, tick_end + 1, 25)) #
            ax.set_xlim(min_x - 1, max_x + 1)
        else:
            ax.set_xticks(np.arange(0, 25, 25)) #
    
    def create_combined_plot(self, lessons: List[str], phase_curves: Dict, verbose: bool = False) -> plt.Figure:
        """Create combined plot for all lessons."""
        fig, axs = plt.subplots(2, 2, figsize=self.config.figure_size)
        axs = axs.flatten()
        
        for i, lesson in enumerate(lessons):
            if i < len(axs):
                self.plot_lesson_progress(axs[i], lesson, phase_curves, verbose)
        
        plt.suptitle("Learning Progress Over Time for Each Lesson (Mastery vs Activities)", 
                    fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str = "all_lessons_learning_progress_phases.png"):
        """Save plot to file."""
        self.create_plots_folder()
        fig_path = os.path.join(self.config.plots_folder, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{fig_path}'")
        return fig_path

class LearningProgressAnalyzer:
    """Main analyzer class that orchestrates the analysis workflow."""
    
    def __init__(self, config: LearningProgressConfig = None):
        self.config = config or LearningProgressConfig()
        self.phase_analyzer = PhaseAnalyzer(self.config)
        self.visualizer = LearningProgressVisualizer(self.config)
        
        self.data = None
        self.lessons = None
        self.phase_curves = None
    
    def load_data(self) -> Dict:
        """Load and cache training data."""
        if self.data is None:
            self.data = DataLoader.load_training_data(self.config.training_data_file)
        return self.data
    
    def get_lessons(self) -> List[str]:
        """Get and cache lesson list."""
        if self.lessons is None:
            data = self.load_data()
            self.lessons = DataLoader.extract_lessons(data["episodes"])
        return self.lessons
    
    def analyze_phases(self) -> Dict:
        """Analyze learning progress across phases."""
        if self.phase_curves is None:
            data = self.load_data()
            lessons = self.get_lessons()
            self.phase_curves = self.phase_analyzer.calculate_phase_curves(
                data["episodes"], lessons
            )
        return self.phase_curves
    
    def generate_report(self, verbose: bool = False) -> Dict:
        """Generate analysis report."""
        lessons = self.get_lessons()
        phase_curves = self.analyze_phases()
        
        report = {
            "total_lessons": len(lessons),
            "lessons": lessons,
            "phases": list(self.config.phases.keys()),
            "lesson_stats": {}
        }
        
        mastery_analyzer = MasteryAnalyzer(self.config)
        
        for lesson in lessons:
            lesson_stats = {}
            for phase in self.config.phases.keys():
                mastery_data = np.array(phase_curves[lesson][phase])
                if len(mastery_data) > 0:
                    max_mastery = np.max(mastery_data)
                    start_idx = mastery_analyzer.find_improvement_start(mastery_data)
                    mastery_idx = None
                    if start_idx is not None:
                        mastery_idx = mastery_analyzer.find_mastery_achievement(mastery_data, start_idx)
                    
                    lesson_stats[phase] = {
                        "max_mastery": float(max_mastery),
                        "improvement_start": int(start_idx) if start_idx is not None else None,
                        "mastery_achieved_at": int(mastery_idx) if mastery_idx is not None else None,
                        "total_steps": len(mastery_data)
                    }
            
            report["lesson_stats"][lesson] = lesson_stats
        
        return report
    
    def create_visualization(self, verbose: bool = False, save: bool = True, 
                           filename: str = None) -> plt.Figure:
        """Create and optionally save visualization."""
        lessons = self.get_lessons()
        phase_curves = self.analyze_phases()
        
        fig = self.visualizer.create_combined_plot(lessons, phase_curves, verbose)
        
        if save:
            filename = filename or "all_lessons_learning_progress_phases.png"
            self.visualizer.save_plot(fig, filename)
        
        return fig
    
    def run_full_analysis(self, verbose: bool = False, save_plot: bool = True, 
                         show_plot: bool = False) -> Tuple[Dict, plt.Figure]:
        """Run complete analysis workflow."""
        print("Loading data...")
        self.load_data()
        
        print("Analyzing phases...")
        self.analyze_phases()
        
        print("Generating report...")
        report = self.generate_report(verbose)
        
        print("Creating visualization...")
        fig = self.create_visualization(verbose, save_plot)
        
        if show_plot:
            plt.show()
        
        return report, fig

def analyze_learning_progress(training_data_file: str = None, 
                            plots_folder: str = None,
                            verbose: bool = False,
                            save_plot: bool = True,
                            show_plot: bool = False) -> Tuple[Dict, plt.Figure]:
    """
    Convenience function to run learning progress analysis.
    
    Args:
        training_data_file: Path to training data JSON file
        plots_folder: Folder to save plots
        verbose: Whether to print debug information
        save_plot: Whether to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        Tuple of (analysis_report, matplotlib_figure)
    """
    config = LearningProgressConfig()
    
    if training_data_file:
        config.training_data_file = training_data_file
    if plots_folder:
        config.plots_folder = plots_folder
    
    analyzer = LearningProgressAnalyzer(config)
    return analyzer.run_full_analysis(verbose, save_plot, show_plot)

if __name__ == "__main__":
    log_file = "training_data/student_data_Read_Write_85pct_09vel.json"
    unified_output_dir = "tests/model_test_results/student_data_Read_Write_85pct_09vel"
    report, _= analyze_learning_progress(log_file, unified_output_dir)