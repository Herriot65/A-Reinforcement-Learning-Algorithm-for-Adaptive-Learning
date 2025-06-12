import random
from typing import Dict
from utils.student_profile_generator import generate_profile

class Student:
    """Handles student profile generation and performance simulation based on VARK's empirical data."""
    
    def __init__(
        self, 
        dominant_style: str = None, 
        dominant_percent: float = None, 
        velocity: float = None 
    ):
        """
        Initialize student with either:
        - Random profile (default) or
        - Specific characteristics
        
        Args:
            dominant_style: If None, random style is chosen
            dominant_percent: If None, random percentage is chosen
            velocity: If None, random velocity is chosen
        """
        # Generate random values if not specified
        self.dominant_style = dominant_style or random.choice(
            ["Visual", "Auditory", "Read/Write", "Kinesthetic"]
        )
        self.dominant_percent = dominant_percent or random.choice([80, 70, 45])
        
        # Generate and normalize profile
        self.profile = generate_profile(
            self.dominant_style, 
            self.dominant_percent,
            velocity=velocity 
        )
        self._normalize_profile()
    
    def _normalize_profile(self) -> None:
        """Ensure style weights sum to 1.0."""
        styles = self.profile['style']
        total = sum(styles.values())
        for style in styles:
            styles[style] /= total
    
    @property
    def learning_style(self) -> Dict[str, float]:
        """Get normalized style weights."""
        return self.profile['style']
    
    @property
    def velocity(self) -> float:
        """Get velocity value."""
        # This property correctly accesses the velocity from the generated profile
        return self.profile['velocity']
    
    def calculate_performance(self, activity: Dict) -> float:
        """
        Calculate performance on an activity.
        
        Args:
            activity: Dict with "style" and "nb_points" keys
            
        Returns:
            Performance score 
        """
        if not all(k in activity for k in ["style", "nb_points"]):
            raise ValueError("Activity must contain 'style' and 'nb_points'")
            
        performance = 0.0
        for style, weight in self.learning_style.items():
            activity_style = activity["style"].get(style, 0.0)
            performance += weight * self.velocity * activity_style * activity["nb_points"]
        
        return round(performance, 2) 