"""
Implementation of the NEWS-2 score using fuzzy logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

from .fuzzy_logic import FuzzyLogic


@dataclass
class NEWS2Result:
    """Result of NEWS-2 calculation."""
    crisp_score: int
    fuzzy_score: float
    risk_category: str
    recommended_response: str
    parameter_scores: Dict[str, Union[int, float]]
    fuzzy_memberships: Dict[str, Dict[str, float]]


class FuzzyNEWS2:
    """
    Fuzzy logic implementation of the NEWS-2 score.
    """
    
    # Constants for NEWS-2 risk categories
    LOW_RISK = "Low"
    LOW_MEDIUM_RISK = "Low-Medium"
    MEDIUM_RISK = "Medium" 
    HIGH_RISK = "High"
    
    # Constants for consciousness levels
    CONSCIOUSNESS_LEVELS = {
        "A": 0,  # Alert
        "V": 3,  # Voice
        "P": 3,  # Pain
        "U": 3,  # Unresponsive
    }
    
    def __init__(self):
        """Initialize the FuzzyNEWS2 system."""
        self.fuzzy_system = FuzzyLogic()
        self._setup_fuzzy_system()
    
    def _setup_fuzzy_system(self):
        """Set up the fuzzy logic system for NEWS-2 calculation."""
        # Define universes for input variables
        respiratory_rate_universe = np.arange(0, 50, 1)
        oxygen_saturation_universe = np.arange(70, 101, 1)
        systolic_bp_universe = np.arange(50, 250, 1)
        pulse_universe = np.arange(20, 180, 1)
        temperature_universe = np.arange(33, 43, 0.1)
        
        # Define universe for output variable (NEWS-2 score)
        score_universe = np.arange(0, 21, 0.1)
        
        # Define membership functions for respiratory rate
        resp_rate_mfs = {
            "normal": {"type": "trimf", "params": [8, 12, 20]},
            "low": {"type": "trapmf", "params": [0, 0, 8, 12]},
            "high": {"type": "trapmf", "params": [20, 24, 50, 50]},
            "very_high": {"type": "trapmf", "params": [24, 30, 50, 50]}
        }
        
        # Define membership functions for oxygen saturation
        oxygen_sat_mfs = {
            "normal": {"type": "trapmf", "params": [96, 98, 100, 100]},
            "low": {"type": "trimf", "params": [94, 95, 96]},
            "very_low": {"type": "trapmf", "params": [70, 70, 92, 94]}
        }
        
        # Define membership functions for systolic blood pressure
        systolic_bp_mfs = {
            "normal": {"type": "trimf", "params": [111, 130, 219]},
            "low": {"type": "trimf", "params": [101, 105, 111]},
            "very_low": {"type": "trapmf", "params": [50, 50, 90, 101]},
            "high": {"type": "trapmf", "params": [219, 230, 250, 250]}
        }
        
        # Define membership functions for pulse rate
        pulse_mfs = {
            "normal": {"type": "trimf", "params": [51, 70, 90]},
            "low": {"type": "trapmf", "params": [20, 20, 40, 51]},
            "high": {"type": "trimf", "params": [90, 110, 130]},
            "very_high": {"type": "trapmf", "params": [130, 140, 180, 180]}
        }
        
        # Define membership functions for temperature
        temperature_mfs = {
            "normal": {"type": "trimf", "params": [36.1, 37.0, 38.0]},
            "low": {"type": "trimf", "params": [35.1, 35.5, 36.1]},
            "very_low": {"type": "trapmf", "params": [33.0, 33.0, 35.0, 35.1]},
            "high": {"type": "trimf", "params": [38.0, 38.5, 39.0]},
            "very_high": {"type": "trapmf", "params": [39.0, 39.5, 43.0, 43.0]}
        }
        
        # Define membership functions for NEWS-2 score
        score_mfs = {
            "low": {"type": "trimf", "params": [0, 1, 4]},
            "medium": {"type": "trimf", "params": [4, 6, 7]},
            "high": {"type": "trapmf", "params": [7, 9, 20, 20]}
        }
        
        # Create antecedents
        resp_rate = self.fuzzy_system._create_antecedent(
            "respiratory_rate", respiratory_rate_universe, resp_rate_mfs
        )
        
        oxygen_sat = self.fuzzy_system._create_antecedent(
            "oxygen_saturation", oxygen_saturation_universe, oxygen_sat_mfs
        )
        
        systolic_bp = self.fuzzy_system._create_antecedent(
            "systolic_bp", systolic_bp_universe, systolic_bp_mfs
        )
        
        pulse = self.fuzzy_system._create_antecedent(
            "pulse", pulse_universe, pulse_mfs
        )
        
        temperature = self.fuzzy_system._create_antecedent(
            "temperature", temperature_universe, temperature_mfs
        )
        
        # Create consequent
        score = self.fuzzy_system._create_consequent(
            "score", score_universe, score_mfs
        )
        
        # Define rules
        rules = [
            # Respiratory rate rules
            self.fuzzy_system._add_rule(resp_rate["normal"], score["low"]),
            self.fuzzy_system._add_rule(resp_rate["low"], score["medium"]),
            self.fuzzy_system._add_rule(resp_rate["high"], score["medium"]),
            self.fuzzy_system._add_rule(resp_rate["very_high"], score["high"]),
            
            # Oxygen saturation rules
            self.fuzzy_system._add_rule(oxygen_sat["normal"], score["low"]),
            self.fuzzy_system._add_rule(oxygen_sat["low"], score["medium"]),
            self.fuzzy_system._add_rule(oxygen_sat["very_low"], score["high"]),
            
            # Systolic BP rules
            self.fuzzy_system._add_rule(systolic_bp["normal"], score["low"]),
            self.fuzzy_system._add_rule(systolic_bp["low"], score["medium"]),
            self.fuzzy_system._add_rule(systolic_bp["very_low"], score["high"]),
            self.fuzzy_system._add_rule(systolic_bp["high"], score["medium"]),
            
            # Pulse rules
            self.fuzzy_system._add_rule(pulse["normal"], score["low"]),
            self.fuzzy_system._add_rule(pulse["low"], score["medium"]),
            self.fuzzy_system._add_rule(pulse["high"], score["medium"]),
            self.fuzzy_system._add_rule(pulse["very_high"], score["high"]),
            
            # Temperature rules
            self.fuzzy_system._add_rule(temperature["normal"], score["low"]),
            self.fuzzy_system._add_rule(temperature["low"], score["medium"]),
            self.fuzzy_system._add_rule(temperature["very_low"], score["high"]),
            self.fuzzy_system._add_rule(temperature["high"], score["medium"]),
            self.fuzzy_system._add_rule(temperature["very_high"], score["high"]),
            
            # Combination rules
            self.fuzzy_system._add_rule(
                (resp_rate["high"] & oxygen_sat["low"]), score["high"]
            ),
            self.fuzzy_system._add_rule(
                (pulse["high"] & resp_rate["high"]), score["high"]
            ),
            self.fuzzy_system._add_rule(
                (systolic_bp["low"] & pulse["high"]), score["high"]
            ),
        ]
        
        # Build the control system
        self.fuzzy_system.build_control_system(rules)
    
    def _calculate_crisp_score(
        self,
        respiratory_rate: int,
        oxygen_saturation: int,
        systolic_bp: int,
        pulse: int,
        consciousness: str,
        temperature: float,
        supplemental_oxygen: bool
    ) -> Dict[str, int]:
        """
        Calculate the traditional NEWS-2 score.
        
        Args:
            respiratory_rate: Breaths per minute
            oxygen_saturation: O2 saturation (%)
            systolic_bp: Systolic blood pressure (mmHg)
            pulse: Pulse rate (beats per minute)
            consciousness: Level of consciousness (A, V, P, or U)
            temperature: Body temperature (°C)
            supplemental_oxygen: Whether supplemental oxygen is being used
        
        Returns:
            Dictionary of parameter scores and total score
        """
        # Respiratory rate score
        if respiratory_rate <= 8:
            resp_score = 3
        elif respiratory_rate <= 11:
            resp_score = 1
        elif respiratory_rate <= 20:
            resp_score = 0
        elif respiratory_rate <= 24:
            resp_score = 2
        else:  # > 24
            resp_score = 3
        
        # Oxygen saturation score
        if oxygen_saturation <= 91:
            o2_score = 3
        elif oxygen_saturation <= 93:
            o2_score = 2
        elif oxygen_saturation <= 95:
            o2_score = 1
        else:  # >= 96
            o2_score = 0
        
        # Supplemental oxygen score
        if supplemental_oxygen:
            o2_therapy_score = 2
        else:
            o2_therapy_score = 0
        
        # Systolic BP score
        if systolic_bp <= 90:
            bp_score = 3
        elif systolic_bp <= 100:
            bp_score = 2
        elif systolic_bp <= 110:
            bp_score = 1
        elif systolic_bp <= 219:
            bp_score = 0
        else:  # >= 220
            bp_score = 3
        
        # Pulse score
        if pulse <= 40:
            pulse_score = 3
        elif pulse <= 50:
            pulse_score = 1
        elif pulse <= 90:
            pulse_score = 0
        elif pulse <= 110:
            pulse_score = 1
        elif pulse <= 130:
            pulse_score = 2
        else:  # > 130
            pulse_score = 3
        
        # Consciousness score
        if consciousness in self.CONSCIOUSNESS_LEVELS:
            consciousness_score = self.CONSCIOUSNESS_LEVELS[consciousness]
        else:
            raise ValueError(f"Invalid consciousness level: {consciousness}. "
                           f"Must be one of {list(self.CONSCIOUSNESS_LEVELS.keys())}.")
        
        # Temperature score
        if temperature <= 35.0:
            temp_score = 3
        elif temperature <= 36.0:
            temp_score = 1
        elif temperature <= 38.0:
            temp_score = 0
        elif temperature <= 39.0:
            temp_score = 1
        else:  # > 39.0
            temp_score = 2
        
        # Calculate total score
        total_score = (
            resp_score + o2_score + o2_therapy_score + bp_score +
            pulse_score + consciousness_score + temp_score
        )
        
        return {
            "respiratory_rate": resp_score,
            "oxygen_saturation": o2_score,
            "supplemental_oxygen": o2_therapy_score,
            "systolic_bp": bp_score,
            "pulse": pulse_score,
            "consciousness": consciousness_score,
            "temperature": temp_score,
            "total": total_score
        }
    
    def _determine_risk_category(
        self,
        crisp_score: int,
        fuzzy_score: float,
        any_param_score_3: bool
    ) -> str:
        """
        Determine the risk category based on the NEWS-2 score.
        
        Args:
            crisp_score: Traditional NEWS-2 score
            fuzzy_score: Fuzzy NEWS-2 score
            any_param_score_3: Whether any parameter has a score of 3
        
        Returns:
            Risk category as a string
        """
        if any_param_score_3 or crisp_score >= 7 or fuzzy_score >= 7:
            return self.HIGH_RISK
        elif crisp_score >= 5 or fuzzy_score >= 5:
            return self.MEDIUM_RISK
        elif crisp_score >= 1 or fuzzy_score >= 1:
            return self.LOW_MEDIUM_RISK
        else:
            return self.LOW_RISK
    
    def _determine_recommended_response(self, risk_category: str) -> str:
        """
        Determine the recommended clinical response based on risk category.
        
        Args:
            risk_category: Risk category as determined by _determine_risk_category
        
        Returns:
            Recommended clinical response as a string
        """
        if risk_category == self.HIGH_RISK:
            return ("Urgent assessment by a clinical team / team with critical care "
                   "competencies, which may include a critical care outreach team")
        elif risk_category == self.MEDIUM_RISK:
            return ("Urgent review by ward-based clinician, which may include a critical "
                   "care outreach team")
        elif risk_category == self.LOW_MEDIUM_RISK:
            return "Clinician review within 12 hours"
        else:  # LOW_RISK
            return "Continue routine monitoring"
    
    def calculate(
        self,
        respiratory_rate: int,
        oxygen_saturation: int,
        systolic_bp: int,
        pulse: int,
        consciousness: str,
        temperature: float,
        supplemental_oxygen: bool = False
    ) -> NEWS2Result:
        """
        Calculate the NEWS-2 score using both crisp and fuzzy logic.
        
        Args:
            respiratory_rate: Breaths per minute
            oxygen_saturation: O2 saturation (%)
            systolic_bp: Systolic blood pressure (mmHg)
            pulse: Pulse rate (beats per minute)
            consciousness: Level of consciousness (A, V, P, or U)
            temperature: Body temperature (°C)
            supplemental_oxygen: Whether supplemental oxygen is being used
        
        Returns:
            NEWS2Result object containing scores, risk category, and recommended response
        """
        # Calculate crisp NEWS-2 score
        crisp_scores = self._calculate_crisp_score(
            respiratory_rate=respiratory_rate,
            oxygen_saturation=oxygen_saturation,
            systolic_bp=systolic_bp,
            pulse=pulse,
            consciousness=consciousness,
            temperature=temperature,
            supplemental_oxygen=supplemental_oxygen
        )
        
        # Calculate fuzzy NEWS-2 score
        fuzzy_inputs = {
            "respiratory_rate": respiratory_rate,
            "oxygen_saturation": oxygen_saturation,
            "systolic_bp": systolic_bp,
            "pulse": pulse,
            "temperature": temperature
        }
        
        # Add supplemental oxygen and consciousness level to crisp scores
        fuzzy_scores = dict(crisp_scores)
        
        try:
            # Compute fuzzy result
            fuzzy_result = self.fuzzy_system.compute(fuzzy_inputs)
            fuzzy_score = fuzzy_result["score"]
            
            # Add supplemental oxygen score to fuzzy score (handled as crisp)
            if supplemental_oxygen:
                fuzzy_score += 2
            
            # Add consciousness score to fuzzy score (handled as crisp)
            if consciousness in self.CONSCIOUSNESS_LEVELS:
                fuzzy_score += self.CONSCIOUSNESS_LEVELS[consciousness]
        except Exception as e:
            # Fallback to crisp score if fuzzy computation fails
            fuzzy_score = crisp_scores["total"]
        
        # Get fuzzy membership values for each input
        fuzzy_memberships = {}
        
        # Determine risk category
        any_param_score_3 = any(
            v == 3 for k, v in crisp_scores.items() if k != "total"
        )
        
        risk_category = self._determine_risk_category(
            crisp_scores["total"], fuzzy_score, any_param_score_3
        )
        
        # Determine recommended response
        recommended_response = self._determine_recommended_response(risk_category)
        
        return NEWS2Result(
            crisp_score=crisp_scores["total"],
            fuzzy_score=fuzzy_score,
            risk_category=risk_category,
            recommended_response=recommended_response,
            parameter_scores=crisp_scores,
            fuzzy_memberships=fuzzy_memberships
        )
