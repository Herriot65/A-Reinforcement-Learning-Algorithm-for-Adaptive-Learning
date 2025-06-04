import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd

class CurriculumAnalyzer:
    """
    Analyzes curriculum datasets for balance, bias, and learning style distribution.
    Evaluates strong dominance patterns and provides comprehensive statistics.
    """
    
    def __init__(self, curriculum_path: str):
        """Initialize analyzer with curriculum JSON file."""
        with open(curriculum_path, 'r', encoding='utf-8') as f:
            self.curriculum = json.load(f)
        
        self.lessons = self.curriculum['lessons']
        self.activities = self.curriculum['activities']
        self.learning_styles = ['visual', 'auditory', 'read_write', 'kinesthetic']
        self.dominance_threshold = 0.4
        
    def analyze_learning_style_distribution(self) -> Dict:
        """Analyze distribution of learning styles across all activities."""
        style_stats = {style: [] for style in self.learning_styles}
        
        for activity in self.activities:
            for style in self.learning_styles:
                style_stats[style].append(activity['style'][style])
        
        distribution_stats = {}
        for style in self.learning_styles:
            values = style_stats[style]
            distribution_stats[style] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        return distribution_stats, style_stats
    
    def analyze_dominance_patterns(self) -> Dict:
        """Analyze strong dominance patterns (>=0.4 threshold)."""
        dominance_analysis = {
            'strong_dominant_activities': {style: 0 for style in self.learning_styles},
            'multi_dominant_activities': 0,
            'no_dominant_activities': 0,
            'dominance_combinations': defaultdict(int),
            'dominance_details': []
        }
        
        for activity in self.activities:
            dominant_styles = []
            for style in self.learning_styles:
                if activity['style'][style] >= self.dominance_threshold:
                    dominant_styles.append(style)
                    dominance_analysis['strong_dominant_activities'][style] += 1
            
            if len(dominant_styles) > 1:
                dominance_analysis['multi_dominant_activities'] += 1
                combo_key = '+'.join(sorted(dominant_styles))
                dominance_analysis['dominance_combinations'][combo_key] += 1
            elif len(dominant_styles) == 0:
                dominance_analysis['no_dominant_activities'] += 1
            else:
                dominance_analysis['dominance_combinations'][dominant_styles[0]] += 1
            
            dominance_analysis['dominance_details'].append({
                'activity_id': activity['id'],
                'dominant_styles': dominant_styles,
                'style_values': activity['style']
            })
        
        return dominance_analysis
    
    def analyze_lesson_coverage(self) -> Dict:
        """Analyze how lessons are covered across activities."""
        lesson_coverage = {lesson['id']: {'total_coverage': 0, 'activity_count': 0, 'coverages': []} 
                           for lesson in self.lessons}
        
        for activity in self.activities:
            for lesson_id, coverage in activity['lessons'].items():
                if lesson_id in lesson_coverage:
                    lesson_coverage[lesson_id]['total_coverage'] += coverage
                    lesson_coverage[lesson_id]['activity_count'] += 1
                    lesson_coverage[lesson_id]['coverages'].append(coverage)
        
        # Calculate statistics for each lesson
        for lesson_id in lesson_coverage:
            coverages = lesson_coverage[lesson_id]['coverages']
            if coverages:
                lesson_coverage[lesson_id]['mean_coverage'] = np.mean(coverages)
                lesson_coverage[lesson_id]['std_coverage'] = np.std(coverages)
                lesson_coverage[lesson_id]['min_coverage'] = np.min(coverages)
                lesson_coverage[lesson_id]['max_coverage'] = np.max(coverages)
        
        return lesson_coverage
    
    def analyze_prerequisites_complexity(self) -> Dict:
        """Analyze prerequisite complexity and distribution."""
        prereq_stats = {
            'lessons_with_prerequisites': 0,
            'total_prerequisite_relationships': 0,
            'prerequisite_strengths': [],
            'lessons_by_prereq_count': defaultdict(int)
        }
        
        for lesson in self.lessons:
            prereq_count = len(lesson['prerequisites'])
            prereq_stats['lessons_by_prereq_count'][prereq_count] += 1
            
            if prereq_count > 0:
                prereq_stats['lessons_with_prerequisites'] += 1
                prereq_stats['total_prerequisite_relationships'] += prereq_count
                
                for strength in lesson['prerequisites'].values():
                    prereq_stats['prerequisite_strengths'].append(strength)
        
        if prereq_stats['prerequisite_strengths']:
            prereq_stats['mean_prereq_strength'] = np.mean(prereq_stats['prerequisite_strengths'])
            prereq_stats['std_prereq_strength'] = np.std(prereq_stats['prerequisite_strengths'])
        
        return prereq_stats
    
    def check_balance_and_bias(self) -> Dict:
        """Comprehensive balance and bias analysis."""
        distribution_stats, style_stats = self.analyze_learning_style_distribution()
        dominance_stats = self.analyze_dominance_patterns()
        
        # Statistical tests for balance
        balance_analysis = {
            'style_balance_score': 0,
            'coefficient_of_variation': {},
            'style_correlation_matrix': {},
            'balance_verdict': '',
            'bias_indicators': []
        }
        
        # Calculate coefficient of variation for each style
        total_activities = len(self.activities)
        style_means = [distribution_stats[style]['mean'] for style in self.learning_styles]
        overall_mean = np.mean(style_means)
        overall_std = np.std(style_means)
        
        balance_analysis['style_balance_score'] = 1 - (overall_std / overall_mean) if overall_mean > 0 else 0
        
        for style in self.learning_styles:
            cv = distribution_stats[style]['std'] / distribution_stats[style]['mean']
            balance_analysis['coefficient_of_variation'][style] = cv
        
        # Correlation analysis
        style_matrix = np.array([style_stats[style] for style in self.learning_styles]).T
        correlation_matrix = np.corrcoef(style_matrix.T)
        balance_analysis['style_correlation_matrix'] = {
            self.learning_styles[i]: {
                self.learning_styles[j]: correlation_matrix[i][j] 
                for j in range(len(self.learning_styles))
            } for i in range(len(self.learning_styles))
        }
        
        # Bias detection
        dominance_percentages = {
            style: (dominance_stats['strong_dominant_activities'][style] / total_activities) * 100
            for style in self.learning_styles
        }
        
        max_dominance = max(dominance_percentages.values())
        min_dominance = min(dominance_percentages.values())
        dominance_range = max_dominance - min_dominance
        
        if dominance_range > 15:  # More than 15% difference
            balance_analysis['bias_indicators'].append(f"High dominance range: {dominance_range:.1f}%")
        
        if balance_analysis['style_balance_score'] > 0.8:
            balance_analysis['balance_verdict'] = "Well Balanced"
        elif balance_analysis['style_balance_score'] > 0.6:
            balance_analysis['balance_verdict'] = "Moderately Balanced"
        else:
            balance_analysis['balance_verdict'] = "Potentially Biased"
        
        balance_analysis['dominance_percentages'] = dominance_percentages
        
        return balance_analysis
    
    def create_visualizations(self, save_path: str = "curriculum_analysis"):
        """Create comprehensive visualizations."""
        distribution_stats, style_stats = self.analyze_learning_style_distribution()
        dominance_stats = self.analyze_dominance_patterns()
        balance_analysis = self.check_balance_and_bias()
        
        # Adjusting subplots to only show the plots you are actively filling
        # There are 4 plots you're using: axes[0,0], axes[0,1], axes[1,0], axes[1,1]
        # So, we need a 2x2 grid.
        fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Changed from 2,3 to 2,2 and adjusted figsize
        fig.suptitle('Curriculum Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning Style Distribution
        ax1 = axes[0, 0]
        style_data = [style_stats[style] for style in self.learning_styles]
        ax1.boxplot(style_data, tick_labels=[s.capitalize() for s in self.learning_styles])
        ax1.set_title('Learning Style Value Distribution')
        ax1.set_ylabel('Style Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Dominance Analysis
        ax2 = axes[0, 1]
        dominance_counts = [dominance_stats['strong_dominant_activities'][style] for style in self.learning_styles]
        bars = ax2.bar([s.capitalize() for s in self.learning_styles], dominance_counts, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title(f'Strong Dominance Distribution (≥{self.dominance_threshold})')
        ax2.set_ylabel('Number of Activities')
        
        # Add percentage labels on bars
        total = len(self.activities)
        for bar, count in zip(bars, dominance_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{(count/total)*100:.1f}%', ha='center', va='bottom')
        
        # 4. Balance Score Visualization
        ax4 = axes[1, 0] # This was originally axes[1,0] and remains the same
        categories = ['Balance\nScore', 'Multi-Dom\n%', 'No-Dom\n%']
        values = [
            balance_analysis['style_balance_score'],
            (dominance_stats['multi_dominant_activities'] / len(self.activities)),
            (dominance_stats['no_dominant_activities'] / len(self.activities))
        ]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_title('Balance Metrics')
        ax4.set_ylabel('Score/Percentage')
        ax4.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom')
        
        # 5. Lesson Coverage Distribution
        ax5 = axes[1, 1] # This was originally axes[1,1] and remains the same
        lesson_coverage = self.analyze_lesson_coverage()
        lesson_ids = list(lesson_coverage.keys())
        activity_counts = [lesson_coverage[lid]['activity_count'] for lid in lesson_ids]
        ax5.bar(lesson_ids, activity_counts, color='skyblue')
        ax5.set_title('Activities per Lesson')
        ax5.set_ylabel('Number of Activities')
        ax5.set_xlabel('Lesson ID')
        
        
        plt.tight_layout()
        plt.savefig(f'{save_path}_visualizations.png', dpi=300, bbox_inches='tight')
        
        return fig

def get_activity_dominant_style(activity):
    """Get the dominant learning style for an activity."""
    style_dict = activity['style']
    return max(style_dict, key=style_dict.get)

def analyze_style_distribution_in_activities(activities):
    """Analyze the distribution of dominant styles in the activity set."""
    style_counts = {'visual': 0, 'auditory': 0, 'read_write': 0, 'kinesthetic': 0}
    
    for activity in activities:
        dominant_style = get_activity_dominant_style(activity)
        style_counts[dominant_style] += 1
    
    total_activities = len(activities)
    print(f"\nActivity Style Distribution:")
    print(f"  Total activities: {total_activities}")
    for style, count in style_counts.items():
        percentage = (count / total_activities) * 100
        print(f"  {style.replace('_', '/').title()}: {count} ({percentage:.1f}%)")
    
    return style_counts


def main(curriculum_file: str = "data/balanced_curriculum.json"):
    """Main function to run the analysis."""  
    try:
        analyzer = CurriculumAnalyzer(curriculum_file)

        # Create visualizations
        analyzer.create_visualizations()
        
        print(f" Visualizations saved as 'curriculum_analysis_visualizations.png'")
        
    except FileNotFoundError:
        print(f" Error: Curriculum file '{curriculum_file}' not found.")
        print("Please run the curriculum generator first to create the dataset.")
    except Exception as e:
        print(f" Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()