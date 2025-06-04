from typing import Dict
import random
from collections import defaultdict

COMBINATIONS = {
    "AK": 11.9, "AR": 0.7, "RK": 2.2, "VA": 0.3, 
    "VK": 4.6, "VR": 0.4, "ARK": 4.5, "VAK": 7.5,
    "VAR": 0.2, "VRK": 2.8, "VARK Selective": 2.7,
    "VARK": 2.8, "VARK Integrative": 25.4
}

def calculate_conditional_weights(dominant_style: str) -> Dict[str, float]:
    """Calculate style weights given a dominant style."""
    style_map = {"V": "Visual", "A": "Auditory", "R": "Read/Write", "K": "Kinesthetic"}
    dominant_letter = next(k for k, v in style_map.items() if v == dominant_style)
    
    co_occurrence_counts = defaultdict(float)
    total_weight = 0.0

    for combo, freq in COMBINATIONS.items():
        clean_combo = "".join(c for c in combo if c in "VARK")
        if dominant_letter in clean_combo:
            for letter in clean_combo:
                if letter != dominant_letter:
                    style = style_map[letter]
                    co_occurrence_counts[style] += freq
                    total_weight += freq

    return {style: round(freq / total_weight, 2) for style, freq in co_occurrence_counts.items()}

def generate_profile(
    dominant_style: str, 
    dominant_percent: float, 
    velocity: float = None
) -> Dict[str, Dict]:
    """
    Generate a VARK profile with nested structure.
    
    Args:
        dominant_style: One of ["Visual", "Auditory", "Read/Write", "Kinesthetic"]
        dominant_percent: Percentage strength (0-100)
        velocity: Optional performance multiplier
        
    Returns:
        Dictionary with {'style': {style_weights}, 'velocity': value}
    """
    if dominant_style not in ["Visual", "Auditory", "Read/Write", "Kinesthetic"]:
        raise ValueError(f"Invalid dominant style: {dominant_style}")
    
    if not 0 <= dominant_percent <= 100:
        raise ValueError(f"dominant_percent must be 0-100, got {dominant_percent}")

    remaining = 100 - dominant_percent
    weights = calculate_conditional_weights(dominant_style)
    
    style_profile = {
        style.lower().replace("/", "_"): round(remaining * weights.get(style, 0), 2)
        for style in ["Visual", "Auditory", "Read/Write", "Kinesthetic"]
    }
    style_profile[dominant_style.lower().replace("/", "_")] = dominant_percent
    
    return {
        'style': style_profile,
        'velocity': velocity if velocity is not None else random.choice([0.9, 0.65, 0.4])
    }