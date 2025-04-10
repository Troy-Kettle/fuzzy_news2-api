"""
Fuzzy logic implementation for the NEWS-2 score calculation.
This module uses our custom fuzzy logic implementation.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .custom_fuzzy import (
    CustomFuzzyLogic,
    create_trimf,
    create_trapmf,
    create_gaussmf
)


class FuzzyLogic:
    """Base class for fuzzy logic operations."""
    
    def __init__(self):
        """Initialize the fuzzy logic system."""
        self.system = CustomFuzzyLogic()
        self.antecedents = {}
        self.consequents = {}
        self.rules = []
    
    def _create_antecedent(self, name, universe, labels_and_mfs):
        """
        Create an antecedent variable.
        
        Args:
            name (str): Name of the antecedent
            universe (numpy.ndarray): Universe of discourse
            labels_and_mfs (dict): Dictionary mapping labels to membership functions
        
        Returns:
            FuzzyVariable: The created antecedent
        """
        variable = self.system.create_variable(name, universe)
        
        for label, mf_params in labels_and_mfs.items():
            mf_type = mf_params["type"]
            params = mf_params["params"]
            
            if mf_type == "trimf":
                variable.add_term(label, create_trimf(params))
            elif mf_type == "trapmf":
                variable.add_term(label, create_trapmf(params))
            elif mf_type == "gaussmf":
                variable.add_term(label, create_gaussmf(params[0], params[1]))
            else:
                raise ValueError(f"Unsupported membership function type: {mf_type}")
        
        self.antecedents[name] = variable
        return variable
    
    def _create_consequent(self, name, universe, labels_and_mfs):
        """
        Create a consequent variable.
        
        Args:
            name (str): Name of the consequent
            universe (numpy.ndarray): Universe of discourse
            labels_and_mfs (dict): Dictionary mapping labels to membership functions
        
        Returns:
            FuzzyVariable: The created consequent
        """
        variable = self.system.create_variable(name, universe)
        
        for label, mf_params in labels_and_mfs.items():
            mf_type = mf_params["type"]
            params = mf_params["params"]
            
            if mf_type == "trimf":
                variable.add_term(label, create_trimf(params))
            elif mf_type == "trapmf":
                variable.add_term(label, create_trapmf(params))
            elif mf_type == "gaussmf":
                variable.add_term(label, create_gaussmf(params[0], params[1]))
            else:
                raise ValueError(f"Unsupported membership function type: {mf_type}")
        
        self.consequents[name] = variable
        return variable
    
    def _add_rule(self, antecedent_combination, consequent_result):
        """
        Add a rule to the fuzzy control system.
        
        Args:
            antecedent_combination: Combination of antecedents
            consequent_result: Consequent result
        
        Returns:
            Rule: The created rule
        """
        # Extract the antecedent and consequent parts
        antecedent_dict = {}
        consequent_tuple = None
        
        # Parse the rule expressions
        # This is a simplified version that assumes antecedent_combination is a simple AND expression
        # For a full implementation, we would need to parse more complex expressions
        if hasattr(antecedent_combination, 'term') and hasattr(antecedent_combination, 'parent'):
            # Single antecedent case
            antecedent_dict[antecedent_combination.parent.label] = antecedent_combination.term
        elif hasattr(antecedent_combination, 'left') and hasattr(antecedent_combination, 'right'):
            # AND operation between two antecedents
            if hasattr(antecedent_combination.left, 'term') and hasattr(antecedent_combination.left, 'parent'):
                antecedent_dict[antecedent_combination.left.parent.label] = antecedent_combination.left.term
            
            if hasattr(antecedent_combination.right, 'term') and hasattr(antecedent_combination.right, 'parent'):
                antecedent_dict[antecedent_combination.right.parent.label] = antecedent_combination.right.term
        else:
            # Try to handle strings or other simple types
            parts = str(antecedent_combination).split('&')
            for part in parts:
                var_term = part.strip().split('_')
                if len(var_term) == 2:
                    antecedent_dict[var_term[0]] = var_term[1]
        
        # Extract consequent
        if hasattr(consequent_result, 'term') and hasattr(consequent_result, 'parent'):
            consequent_tuple = (consequent_result.parent.label, consequent_result.term)
        else:
            # Try to handle strings
            parts = str(consequent_result).split('_')
            if len(parts) == 2:
                consequent_tuple = (parts[0], parts[1])
        
        if not antecedent_dict or not consequent_tuple:
            raise ValueError("Could not parse rule expressions")
        
        # Add the rule
        rule = self.system.add_rule(antecedent_dict, consequent_tuple)
        self.rules.append(rule)
        return rule
    
    def build_control_system(self, rules):
        """
        Build the fuzzy control system from rules.
        
        Args:
            rules (list): List of fuzzy rules
        """
        # The rules are already added during _add_rule
        # So we don't need to do anything here
        pass
    
    def compute(self, inputs):
        """
        Compute the fuzzy result based on inputs.
        
        Args:
            inputs (dict): Dictionary mapping input names to values
        
        Returns:
            dict: Dictionary mapping output names to fuzzy values
        """
        try:
            return self.system.compute(inputs)
        except Exception as e:
            raise RuntimeError(f"Error computing fuzzy result: {str(e)}")


# Create mock classes to mimic skfuzzy's API for backward compatibility

class Term:
    def __init__(self, parent, term):
        self.parent = parent
        self.term = term
    
    def __and__(self, other):
        return AndTerm(self, other)


class AndTerm:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class MockAntecedent:
    def __init__(self, universe, label):
        self.universe = universe
        self.label = label
        self.terms = {}
    
    def __getitem__(self, key):
        self.terms[key] = key
        return Term(self, key)


class MockConsequent(MockAntecedent):
    pass


# Mock control module to provide backward compatibility
class control:
    @staticmethod
    def Antecedent(universe, label):
        return MockAntecedent(universe, label)
    
    @staticmethod
    def Consequent(universe, label):
        return MockConsequent(universe, label)
    
    class Rule:
        def __init__(self, antecedent, consequent):
            self.antecedent = antecedent
            self.consequent = consequent
    
    class ControlSystem:
        def __init__(self, rules):
            self.rules = rules
            self.consequents = []
            
            # Extract consequents from rules
            for rule in rules:
                if hasattr(rule, 'consequent') and hasattr(rule.consequent, 'parent'):
                    self.consequents.append(rule.consequent.parent)
    
    class ControlSystemSimulation:
        def __init__(self, control_system):
            self.control_system = control_system
            self.input = {}
            self.output = {}
        
        def compute(self):
            # This is a mock implementation
            # In a real implementation, this would compute the fuzzy inference
            for consequent in self.control_system.consequents:
                self.output[consequent.label] = 5.0  # Dummy value

