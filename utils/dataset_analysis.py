import os
import json
import seaborn as sns
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from collections import defaultdict, Counter

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StreamlinedDatasetAnalyzer:
    """
    Streamlined analyzer focusing on:
    1. Primary activities per lesson distribution
    2. Dominant learning style distribution
    """
    
    def __init__(self, curriculum_path: str):
        """Load and initialize the curriculum data."""
        with open(curriculum_path, 'r', encoding='utf-8') as f:
            self.curriculum = json.load(f)
        
        self.lessons = self.curriculum['lessons']
        self.activities = self.curriculum['activities']
        self.learning_styles = ['visual', 'auditory', 'read_write', 'kinesthetic']
        
        os.makedirs('data', exist_ok=True)
        
    def analyze_primary_activities_per_lesson(self) -> Dict:
        """Analyze primary activities distribution across lessons."""
        lesson_stats = defaultdict(lambda: {'primary_activities': 0})
        
        for activity in self.activities:
            lessons_coverage = activity['lessons']
            
            # Find the lesson with highest coverage (primary lesson)
            primary_lesson = max(lessons_coverage.items(), key=lambda x: x[1])
            primary_lesson_id, primary_coverage = primary_lesson
            
            # Count as primary if coverage > 0.5 (highly dedicated to this lesson)
            if primary_coverage > 0.5:
                lesson_stats[primary_lesson_id]['primary_activities'] += 1
        
        return dict(lesson_stats)
    
    def analyze_dominant_learning_styles(self) -> Dict:
        """Analyze dominant learning style distribution."""
        dominant_styles = []
        
        for activity in self.activities:
            style_values = activity['style']
            dominant_style = max(style_values.items(), key=lambda x: x[1])
            dominant_styles.append(dominant_style[0])
        
        dominant_counts = Counter(dominant_styles)
        return dominant_counts
    
    def create_combined_analysis_plot(self, lesson_stats: Dict, dominant_counts: Counter):
        """Create a beautiful combined visualization for the report."""

        fig = plt.figure(figsize=(16, 8))
        fig.patch.set_facecolor('white')
        
        # Create subplots with custom spacing
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Define color palettes
        lesson_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        style_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Plot 1: Primary Activities per Lesson
        lessons = list(lesson_stats.keys())
        primary_activities = [lesson_stats[lesson]['primary_activities'] for lesson in lessons]
        
        bars = ax1.bar(lessons, primary_activities, 
                      color=lesson_colors[:len(lessons)], 
                      alpha=0.8, 
                      edgecolor='white', 
                      linewidth=2,
                      capsize=5)
        
        # Enhance the first plot
        ax1.set_xlabel('Lesson ID', fontsize=14, fontweight='600', color='#2C3E50')
        ax1.set_ylabel('Number of Primary Activities', fontsize=14, fontweight='600', color='#2C3E50')
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_facecolor('#FAFAFA')
        
        # Add value labels on bars with style
        for bar, value in zip(bars, primary_activities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(primary_activities) * 0.02,
                    f'{int(value)}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12, color='#2C3E50')
        
        # Add subtle shadow effect
        for bar in bars:
            bar.set_path_effects([path_effects.SimplePatchShadow(offset=(1, -1), 
                                                                shadow_rgbFace='gray', 
                                                                alpha=0.3)])
        
        # Plot 2: Dominant Learning Styles Distribution
        styles = list(dominant_counts.keys())
        counts = list(dominant_counts.values())
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax2.pie(counts, 
                                          labels=[style.replace('_', ' ').title() for style in styles],
                                          autopct='%1.1f%%',
                                          colors=style_colors[:len(styles)],
                                          startangle=90,
                                          explode=[0.05] * len(styles),
                                          shadow=True,
                                          textprops={'fontsize': 11, 'fontweight': '600'})
        
        # Enhance pie chart appearance
        ax2.set_title('Dominant Learning Styles\nDistribution', 
                     fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        
        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # Style the labels
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('600')
            text.set_color('#2C3E50')
        
        circle = plt.Circle((0, 0), 1.1, fill=False, edgecolor='#BDC3C7', linewidth=2, alpha=0.5)
        ax2.add_patch(circle)
        
        fig.suptitle('Learning Dataset Analysis: Key Distributions', 
                    fontsize=20, fontweight='bold', y=0.95, color='#2C3E50')
        
        fig.patch.set_facecolor('#FEFEFE')
        
        # Add footer text
        fig.text(0.5, 0.02, 'Generated for Educational Dataset Quality Assessment', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#7F8C8D')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08)
        
        plt.savefig('data/dataset_analysis_report.png', 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.2)
        plt.show()
    
    def generate_quick_summary(self, lesson_stats: Dict, dominant_counts: Counter):
        """Generate a quick summary of the key findings."""
        print("=" * 60)
        print("DATASET ANALYSIS SUMMARY")
        print("=" * 60)
        
        primary_activities = [lesson_stats[lesson]['primary_activities'] for lesson in lesson_stats.keys()]
        total_primary = sum(primary_activities)
        
        print(f"\n Primary Activities Distribution:")
        print(f"   Total Primary Activities: {total_primary}")
        for lesson_id, stats in lesson_stats.items():
            percentage = (stats['primary_activities'] / total_primary) * 100 if total_primary > 0 else 0
            print(f"   Lesson {lesson_id}: {stats['primary_activities']} ({percentage:.1f}%)")
        
        # Learning styles summary
        total_activities = sum(dominant_counts.values())
        print(f"\n Dominant Learning Styles:")
        print(f"   Total Activities Analyzed: {total_activities}")
        for style, count in dominant_counts.most_common():
            percentage = (count / total_activities) * 100
            print(f"   {style.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def run_analysis(self):
        """Run the streamlined analysis and generate visualization."""
        print("Running streamlined dataset analysis...")
        
        lesson_stats = self.analyze_primary_activities_per_lesson()
        dominant_counts = self.analyze_dominant_learning_styles()
        
        # Create visualization
        print("Generating combined visualization...")
        self.create_combined_analysis_plot(lesson_stats, dominant_counts)
        
        # Generate summary
        self.generate_quick_summary(lesson_stats, dominant_counts)
        
        print("Analysis complete!")
        print("Report visualization saved as: data/dataset_analysis_report.png")

if __name__ == "__main__":
    curriculum_path = "data/learning_curriculum.json"
    
    if not os.path.exists(curriculum_path):
        print("Curriculum file not found. Please generate it first using CurriculumGenerator.")
        print("Expected path: data/learning_curriculum.json")
    else:
        analyzer = StreamlinedDatasetAnalyzer(curriculum_path)
        analyzer.run_analysis()