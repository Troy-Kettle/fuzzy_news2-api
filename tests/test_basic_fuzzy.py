"""
Basic test for the custom fuzzy logic implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fuzzy_news2.custom_fuzzy import (
    CustomFuzzyLogic,
    create_trimf,
    create_trapmf,
    create_gaussmf
)


def test_basic_fuzzy_system():
    """Test a simple fuzzy system using our custom implementation."""
    # Create a fuzzy system
    fuzzy_system = CustomFuzzyLogic()
    
    # Create temperature variable
    temp_universe = np.arange(0, 100, 1)
    temp = fuzzy_system.create_variable("temperature", temp_universe)
    
    # Add terms to the temperature variable
    temp.add_term("cold", create_trimf((0, 20, 40)))
    temp.add_term("warm", create_trimf((30, 50, 70)))
    temp.add_term("hot", create_trimf((60, 80, 100)))
    
    # Create comfort variable
    comfort_universe = np.arange(0, 10, 0.1)
    comfort = fuzzy_system.create_variable("comfort", comfort_universe)
    
    # Add terms to the comfort variable
    comfort.add_term("low", create_trimf((0, 2, 4)))
    comfort.add_term("medium", create_trimf((3, 5, 7)))
    comfort.add_term("high", create_trimf((6, 8, 10)))
    
    # Define rules
    fuzzy_system.add_rule({"temperature": "cold"}, ("comfort", "low"))
    fuzzy_system.add_rule({"temperature": "warm"}, ("comfort", "medium"))
    fuzzy_system.add_rule({"temperature": "hot"}, ("comfort", "high"))
    
    # Test with cold temperature
    result = fuzzy_system.compute({"temperature": 15})
    assert "comfort" in result
    assert result["comfort"] < 4.0  # Should be in the "low" range
    
    # Test with warm temperature
    result = fuzzy_system.compute({"temperature": 50})
    assert result["comfort"] >= 4.0 and result["comfort"] <= 6.0  # Should be in the "medium" range
    
    # Test with hot temperature
    result = fuzzy_system.compute({"temperature": 85})
    assert result["comfort"] > 6.0  # Should be in the "high" range


if __name__ == "__main__":
    test_basic_fuzzy_system()
    print("All tests passed!")
