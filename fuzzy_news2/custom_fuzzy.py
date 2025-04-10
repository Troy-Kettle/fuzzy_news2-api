"""
Custom fuzzy logic implementation that works with Python 3.12.
This module replaces the scikit-fuzzy dependency.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Union, Optional


class FuzzyVariable:
    """Represents a fuzzy variable with membership functions."""
    
    def __init__(self, name: str, universe: np.ndarray):
        """Initialize a fuzzy variable.
        
        Args:
            name: Name of the variable
            universe: Universe of discourse (array of possible values)
        """
        self.name = name
        self.universe = universe
        self.terms = {}
    
    def add_term(self, name: str, mf_func: Callable[[np.ndarray], np.ndarray]):
        """Add a linguistic term with its membership function.
        
        Args:
            name: Name of the linguistic term
            mf_func: Membership function
        """
        self.terms[name] = mf_func
        return self


class FuzzyRule:
    """Represents a fuzzy rule with antecedents and consequent."""
    
    def __init__(self, antecedent: Dict[str, Tuple[FuzzyVariable, str]], 
                 consequent: Tuple[FuzzyVariable, str]):
        """Initialize a fuzzy rule.
        
        Args:
            antecedent: Dictionary mapping variable names to (variable, term) tuples
            consequent: Tuple of (variable, term)
        """
        self.antecedent = antecedent
        self.consequent = consequent


class CustomFuzzyLogic:
    """Custom fuzzy logic implementation."""
    
    def __init__(self):
        """Initialize the fuzzy logic system."""
        self.variables = {}
        self.rules = []
    
    def create_variable(self, name: str, universe: np.ndarray) -> FuzzyVariable:
        """Create a fuzzy variable.
        
        Args:
            name: Name of the variable
            universe: Universe of discourse
        
        Returns:
            The created FuzzyVariable
        """
        variable = FuzzyVariable(name, universe)
        self.variables[name] = variable
        return variable
    
    def add_rule(self, if_part: Dict[str, str], then_part: Tuple[str, str]) -> FuzzyRule:
        """Add a rule to the fuzzy system.
        
        Args:
            if_part: Dictionary mapping variable names to term names
            then_part: Tuple of (variable_name, term_name)
        
        Returns:
            The created FuzzyRule
        """
        # Convert names to actual variables and terms
        antecedent = {}
        for var_name, term_name in if_part.items():
            if var_name not in self.variables:
                raise ValueError(f"Unknown variable: {var_name}")
            var = self.variables[var_name]
            if term_name not in var.terms:
                raise ValueError(f"Unknown term {term_name} for variable {var_name}")
            antecedent[var_name] = (var, term_name)
        
        consequent_var_name, consequent_term_name = then_part
        if consequent_var_name not in self.variables:
            raise ValueError(f"Unknown variable: {consequent_var_name}")
        consequent_var = self.variables[consequent_var_name]
        if consequent_term_name not in consequent_var.terms:
            raise ValueError(f"Unknown term {consequent_term_name} for variable {consequent_var_name}")
        
        rule = FuzzyRule(antecedent, (consequent_var, consequent_term_name))
        self.rules.append(rule)
        return rule
    
    def compute(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Compute the fuzzy inference.
        
        Args:
            inputs: Dictionary mapping variable names to input values
        
        Returns:
            Dictionary mapping output variable names to defuzzified values
        """
        # Fuzzify inputs
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name not in self.variables:
                raise ValueError(f"Unknown input variable: {var_name}")
            
            var = self.variables[var_name]
            fuzzified[var_name] = {}
            
            for term_name, mf_func in var.terms.items():
                # Evaluate membership function at the input value
                membership = self._evaluate_mf(mf_func, value)
                fuzzified[var_name][term_name] = membership
        
        # Initialize output aggregation
        output_aggregation = {}
        for var_name, var in self.variables.items():
            # Skip input variables
            if var_name in inputs:
                continue
            
            # Initialize output aggregation for this variable
            output_aggregation[var_name] = {}
            for term_name in var.terms:
                output_aggregation[var_name][term_name] = 0.0
        
        # Evaluate rules
        for rule in self.rules:
            # Calculate rule activation (minimum of all antecedent memberships)
            antecedent_degrees = []
            for var_name, (_, term_name) in rule.antecedent.items():
                antecedent_degrees.append(fuzzified[var_name][term_name])
            
            rule_activation = min(antecedent_degrees) if antecedent_degrees else 0.0
            
            # Apply rule activation to consequent
            consequent_var, consequent_term = rule.consequent
            var_name = consequent_var.name
            
            # Use maximum aggregation method
            output_aggregation[var_name][consequent_term] = max(
                output_aggregation[var_name][consequent_term],
                rule_activation
            )
        
        # Defuzzify outputs
        defuzzified = {}
        for var_name, term_activations in output_aggregation.items():
            var = self.variables[var_name]
            
            # Skip if no rules activated this output
            if not any(term_activations.values()):
                defuzzified[var_name] = 0.0
                continue
            
            # Defuzzify using centroid method
            defuzzified[var_name] = self._defuzzify_centroid(
                var, term_activations
            )
        
        return defuzzified
    
    def _evaluate_mf(self, mf_func: Callable, x: float) -> float:
        """Evaluate a membership function at a specific point.
        
        Args:
            mf_func: Membership function
            x: Input value
        
        Returns:
            Membership degree
        """
        return float(mf_func(x))
    
    def _defuzzify_centroid(self, var: FuzzyVariable, 
                           term_activations: Dict[str, float]) -> float:
        """Defuzzify using the centroid method.
        
        Args:
            var: Fuzzy variable
            term_activations: Dictionary mapping term names to activation levels
        
        Returns:
            Defuzzified value
        """
        universe = var.universe
        
        # Aggregate membership functions
        aggregated = np.zeros_like(universe, dtype=float)
        
        for term_name, activation in term_activations.items():
            if activation > 0:
                # Apply activation level to term membership function
                term_mf = var.terms[term_name]
                term_result = np.minimum(activation, term_mf(universe))
                
                # Aggregate using maximum
                aggregated = np.maximum(aggregated, term_result)
        
        # Calculate centroid
        if np.sum(aggregated) > 0:
            return float(np.sum(universe * aggregated) / np.sum(aggregated))
        else:
            # Default to middle of universe if no rules activated
            return float(np.mean(universe))


# Helper functions for creating membership functions

def trimf(x: Union[float, np.ndarray], abc: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """Triangular membership function.
    
    Args:
        x: Input value or array
        abc: Parameters (a, b, c) where:
            a: left foot
            b: peak
            c: right foot
    
    Returns:
        Membership degree
    """
    a, b, c = abc
    
    if isinstance(x, (int, float)):
        # Handle scalar input
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c > b else 1.0
    else:
        # Handle array input
        y = np.zeros_like(x, dtype=float)
        
        # Left side
        idx = np.logical_and(a < x, x <= b)
        if b > a:
            y[idx] = (x[idx] - a) / (b - a)
        
        # Right side
        idx = np.logical_and(b < x, x < c)
        if c > b:
            y[idx] = (c - x[idx]) / (c - b)
        
        # Peak
        y[x == b] = 1.0
        
        return y


def trapmf(x: Union[float, np.ndarray], abcd: Tuple[float, float, float, float]) -> Union[float, np.ndarray]:
    """Trapezoidal membership function.
    
    Args:
        x: Input value or array
        abcd: Parameters (a, b, c, d) where:
            a: left foot
            b: left shoulder
            c: right shoulder
            d: right foot
    
    Returns:
        Membership degree
    """
    a, b, c, d = abcd
    
    if isinstance(x, (int, float)):
        # Handle scalar input
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        elif b < x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d > c else 1.0
    else:
        # Handle array input
        y = np.zeros_like(x, dtype=float)
        
        # Left slope
        idx = np.logical_and(a < x, x <= b)
        if b > a:
            y[idx] = (x[idx] - a) / (b - a)
        
        # Flat top
        idx = np.logical_and(b < x, x <= c)
        y[idx] = 1.0
        
        # Right slope
        idx = np.logical_and(c < x, x < d)
        if d > c:
            y[idx] = (d - x[idx]) / (d - c)
        
        # Shoulders
        y[x == b] = 1.0
        y[x == c] = 1.0
        
        return y


def gaussmf(x: Union[float, np.ndarray], mean: float, sigma: float) -> Union[float, np.ndarray]:
    """Gaussian membership function.
    
    Args:
        x: Input value or array
        mean: Mean of the Gaussian function
        sigma: Standard deviation of the Gaussian function
    
    Returns:
        Membership degree
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    if isinstance(x, (int, float)):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    else:
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


# Factory functions for creating membership functions

def create_trimf(abc: Tuple[float, float, float]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """Create a triangular membership function.
    
    Args:
        abc: Parameters (a, b, c)
    
    Returns:
        Membership function
    """
    return lambda x: trimf(x, abc)


def create_trapmf(abcd: Tuple[float, float, float, float]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """Create a trapezoidal membership function.
    
    Args:
        abcd: Parameters (a, b, c, d)
    
    Returns:
        Membership function
    """
    return lambda x: trapmf(x, abcd)


def create_gaussmf(mean: float, sigma: float) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """Create a Gaussian membership function.
    
    Args:
        mean: Mean of the Gaussian function
        sigma: Standard deviation of the Gaussian function
    
    Returns:
        Membership function
    """
    return lambda x: gaussmf(x, mean, sigma)
