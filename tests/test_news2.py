"""
Tests for the NEWS-2 implementation.
"""

import pytest
from fuzzy_news2 import FuzzyNEWS2


def test_crisp_score():
    """Test the crisp NEWS-2 score calculation."""
    fuzzy_news = FuzzyNEWS2()
    
    # Test case 1: All parameters normal
    result = fuzzy_news.calculate(
        respiratory_rate=12,
        oxygen_saturation=98,
        systolic_bp=120,
        pulse=70,
        consciousness="A",
        temperature=37.0,
        supplemental_oxygen=False
    )
    
    assert result.crisp_score == 0
    assert result.risk_category == "Low"
    
    # Test case 2: Some abnormal parameters
    result = fuzzy_news.calculate(
        respiratory_rate=25,
        oxygen_saturation=92,
        systolic_bp=95,
        pulse=120,
        consciousness="A",
        temperature=38.2,
        supplemental_oxygen=True
    )
    
    assert result.crisp_score > 0
    assert result.risk_category in ["Low-Medium", "Medium", "High"]
    
    # Test case 3: Severe abnormalities
    result = fuzzy_news.calculate(
        respiratory_rate=30,
        oxygen_saturation=88,
        systolic_bp=85,
        pulse=140,
        consciousness="V",
        temperature=39.5,
        supplemental_oxygen=True
    )
    
    assert result.crisp_score >= 7
    assert result.risk_category == "High"


def test_consciousness_validation():
    """Test validation of consciousness level."""
    fuzzy_news = FuzzyNEWS2()
    
    # Valid consciousness levels
    for level in ["A", "V", "P", "U"]:
        result = fuzzy_news.calculate(
            respiratory_rate=12,
            oxygen_saturation=98,
            systolic_bp=120,
            pulse=70,
            consciousness=level,
            temperature=37.0,
            supplemental_oxygen=False
        )
        
        if level == "A":
            assert result.parameter_scores["consciousness"] == 0
        else:
            assert result.parameter_scores["consciousness"] == 3
    
    # Invalid consciousness level
    with pytest.raises(ValueError):
        fuzzy_news.calculate(
            respiratory_rate=12,
            oxygen_saturation=98,
            systolic_bp=120,
            pulse=70,
            consciousness="X",  # Invalid
            temperature=37.0,
            supplemental_oxygen=False
        )


def test_fuzzy_vs_crisp():
    """Test the relationship between fuzzy and crisp scores."""
    fuzzy_news = FuzzyNEWS2()
    
    # Test cases at the boundary between risk categories
    result = fuzzy_news.calculate(
        respiratory_rate=21,  # Just above normal
        oxygen_saturation=95,  # Lower end of normal
        systolic_bp=110,  # Lower end of normal
        pulse=91,  # Just above normal
        consciousness="A",
        temperature=38.1,  # Just above normal
        supplemental_oxygen=False
    )
    
    # Fuzzy score should be more nuanced than crisp score
    # The exact relationship depends on the specific fuzzy logic implementation
    assert isinstance(result.fuzzy_score, float)
    
    # The fuzzy score should not be dramatically different from the crisp score
    assert abs(result.fuzzy_score - result.crisp_score) < 5.0


def test_recommended_response():
    """Test that the recommended response matches the risk category."""
    fuzzy_news = FuzzyNEWS2()
    
    # Low risk
    result = fuzzy_news.calculate(
        respiratory_rate=12,
        oxygen_saturation=98,
        systolic_bp=120,
        pulse=70,
        consciousness="A",
        temperature=37.0,
        supplemental_oxygen=False
    )
    
    assert "routine" in result.recommended_response.lower()
    
    # High risk
    result = fuzzy_news.calculate(
        respiratory_rate=30,
        oxygen_saturation=88,
        systolic_bp=85,
        pulse=140,
        consciousness="V",
        temperature=39.5,
        supplemental_oxygen=True
    )
    
    assert "urgent" in result.recommended_response.lower()
    assert "critical care" in result.recommended_response.lower()

