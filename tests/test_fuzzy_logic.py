"""
Tests for the fuzzy logic implementation.
"""

import pytest
import numpy as np

from fuzzy_news2.fuzzy_logic import FuzzyLogic, MockAntecedent, MockConsequent


@pytest.fixture
def fuzzy_system():
    """Create a simple fuzzy logic system for testing."""
    fl = FuzzyLogic()
    
    # Create a simple fuzzy system for testing
    # Define universes
    temperature_universe = np.arange(0, 100, 1)
    humidity_universe = np.arange(0, 100, 1)
    comfort_universe = np.arange(0, 10, 0.1)
    
    # Define membership functions
    temperature_mfs = {
        "cold": {"type": "trimf", "params": [0, 0, 30]},
        "warm": {"type": "trimf", "params": [20, 50, 80]},
        "hot": {"type": "trimf", "params": [70, 100, 100]}
    }
    
    humidity_mfs = {
        "dry": {"type": "trimf", "params": [0, 0, 40]},
        "comfortable": {"type": "trimf", "params": [30, 50, 70]},
        "humid": {"type": "trimf", "params": [60, 100, 100]}
    }
    
    comfort_mfs = {
        "uncomfortable": {"type": "trimf", "params": [0, 0, 5]},
        "moderate": {"type": "trimf", "params": [3, 5, 7]},
        "comfortable": {"type": "trimf", "params": [5, 10, 10]}
    }
    
    # Create antecedents and consequent
    temperature = fl._create_antecedent("temperature", temperature_universe, temperature_mfs)
    humidity = fl._create_antecedent("humidity", humidity_universe, humidity_mfs)
    comfort = fl._create_consequent("comfort", comfort_universe, comfort_mfs)
    
    # Create rules
    rules = [
        fl._add_rule(temperature["cold"] & humidity["dry"], comfort["moderate"]),
        fl._add_rule(temperature["cold"] & humidity["comfortable"], comfort["moderate"]),
        fl._add_rule(temperature["cold"] & humidity["humid"], comfort["uncomfortable"]),
        
        fl._add_rule(temperature["warm"] & humidity["dry"], comfort["comfortable"]),
        fl._add_rule(temperature["warm"] & humidity["comfortable"], comfort["comfortable"]),
        fl._add_rule(temperature["warm"] & humidity["humid"], comfort["moderate"]),
        
        fl._add_rule(temperature["hot"] & humidity["dry"], comfort["moderate"]),
        fl._add_rule(temperature["hot"] & humidity["comfortable"], comfort["uncomfortable"]),
        fl._add_rule(temperature["hot"] & humidity["humid"], comfort["uncomfortable"])
    ]
    
    # Build the control system
    fl.build_control_system(rules)
    
    return fl


def test_create_antecedent():
    """Test creating an antecedent variable."""
    fl = FuzzyLogic()
    
    # Define universe and membership functions
    universe = np.arange(0, 100, 1)
    mfs = {
        "low": {"type": "trimf", "params": [0, 0, 50]},
        "medium": {"type": "trimf", "params": [25, 50, 75]},
        "high": {"type": "trimf", "params": [50, 100, 100]}
    }
    
    # Create antecedent
    antecedent = fl._create_antecedent("test", universe, mfs)
    
    # Verify
    assert antecedent.name == "test"
    assert "low" in antecedent.terms
    assert "medium" in antecedent.terms
    assert "high" in antecedent.terms


def test_create_consequent():
    """Test creating a consequent variable."""
    fl = FuzzyLogic()
    
    # Define universe and membership functions
    universe = np.arange(0, 10, 0.1)
    mfs = {
        "low": {"type": "trimf", "params": [0, 0, 5]},
        "medium": {"type": "trimf", "params": [3, 5, 7]},
        "high": {"type": "trimf", "params": [5, 10, 10]}
    }
    
    # Create consequent
    consequent = fl._create_consequent("test", universe, mfs)
    
    # Verify
    assert consequent.name == "test"
    assert "low" in consequent.terms
    assert "medium" in consequent.terms
    assert "high" in consequent.terms


def test_add_rule():
    """Test adding a rule to the fuzzy system."""
    fl = FuzzyLogic()
    
    # Create antecedents and consequent
    universe = np.arange(0, 100, 1)
    mfs = {"low": {"type": "trimf", "params": [0, 0, 50]},
           "high": {"type": "trimf", "params": [50, 100, 100]}}
    
    antecedent1 = fl._create_antecedent("test1", universe, mfs)
    antecedent2 = fl._create_antecedent("test2", universe, mfs)
    
    output_universe = np.arange(0, 10, 0.1)
    output_mfs = {"low": {"type": "trimf", "params": [0, 0, 5]},
                  "high": {"type": "trimf", "params": [5, 10, 10]}}
    
    consequent = fl._create_consequent("output", output_universe, output_mfs)
    
    # Create rule
    rule = fl._add_rule(antecedent1["low"] & antecedent2["high"], consequent["low"])
    
    # Verify
    assert rule is not None


def test_computation(fuzzy_system):
    """Test fuzzy system computation with different inputs."""
    # Test cold and dry scenario
    result = fuzzy_system.compute({"temperature": 10, "humidity": 20})
    assert "comfort" in result
    assert result["comfort"] > 3 and result["comfort"] < 7  # Should be moderate
    
    # Test warm and comfortable scenario
    result = fuzzy_system.compute({"temperature": 50, "humidity": 50})
    assert result["comfort"] > 5  # Should be comfortable
    
    # Test hot and humid scenario
    result = fuzzy_system.compute({"temperature": 90, "humidity": 90})
    assert result["comfort"] < 5  # Should be uncomfortable


def test_invalid_computation():
    """Test computation with invalid inputs."""
    fl = FuzzyLogic()
    
    with pytest.raises(Exception):
        # Should raise an exception because we haven't defined any rules
        fl.compute({"nonexistent_input": 50})


def test_unsupported_mf_type():
    """Test unsupported membership function type."""
    fl = FuzzyLogic()
    
    # Define universe and invalid membership function type
    universe = np.arange(0, 100, 1)
    mfs = {"invalid": {"type": "unknown_type", "params": [0, 50, 100]}}
    
    with pytest.raises(ValueError, match="Unsupported membership function type"):
        fl._create_antecedent("test", universe, mfs)


def test_gaussmf():
    """Test Gaussian membership function."""
    fl = FuzzyLogic()
    
    # Define universe and Gaussian membership function
    universe = np.arange(0, 100, 1)
    mfs = {"gaussian": {"type": "gaussmf", "params": [50, 10]}}  # mean=50, sigma=10
    
    # Create antecedent
    antecedent = fl._create_antecedent("test", universe, mfs)
    
    # Verify
    assert antecedent is not None
    assert "gaussian" in antecedent.terms


def test_trapmf():
    """Test trapezoidal membership function."""
    fl = FuzzyLogic()
    
    # Define universe and trapezoidal membership function
    universe = np.arange(0, 100, 1)
    mfs = {"trapezoid": {"type": "trapmf", "params": [20, 40, 60, 80]}}
    
    # Create antecedent
    antecedent = fl._create_antecedent("test", universe, mfs)
    
    # Verify
    assert antecedent is not None
    assert "trapezoid" in antecedent.terms
