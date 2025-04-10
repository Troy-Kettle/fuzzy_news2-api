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
    create_gaussmf,
    FuzzyTerm,
    AndTerms
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
        if isinstance(antecedent_combination, FuzzyTerm):
            # Single FuzzyTerm
            var = antecedent_combination.variable
            term = antecedent_combination.term_name
            antecedent_dict[var.name] = term
        elif isinstance(antecedent_combination, AndTerms):
            # AND operation between FuzzyTerms
            self._extract_terms_from_and_terms(antecedent_combination, antecedent_dict)
        elif isinstance(antecedent_combination, Term):
            # Single antecedent case from scikit-fuzzy
            antecedent_dict[antecedent_combination.parent.label] = antecedent_combination.term
        else:
            # Try to handle strings or other simple types
            parts = str(antecedent_combination).split('&')
            for part in parts:
                var_term = part.strip().split('_')
                if len(var_term) == 2:
                    antecedent_dict[var_term[0]] = var_term[1]
        
        # Extract consequent
        if isinstance(consequent_result, FuzzyTerm):
            var = consequent_result.variable
            term = consequent_result.term_name
            consequent_tuple = (var.name, term)
        elif isinstance(consequent_result, Term):
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
    
    def _extract_terms_from_and_terms(self, and_terms, antecedent_dict):
        """
        Extract terms from an AndTerms object.
        
        Args:
            and_terms: AndTerms to extract from
            antecedent_dict: Dictionary to store extracted terms
        """
        if isinstance(and_terms.left, FuzzyTerm):
            var = and_terms.left.variable
            term = and_terms.left.term_name
            antecedent_dict[var.name] = term
        elif isinstance(and_terms.left, AndTerms):
            self._extract_terms_from_and_terms(and_terms.left, antecedent_dict)
        
        if isinstance(and_terms.right, FuzzyTerm):
            var = and_terms.right.variable
            term = and_terms.right.term_name
            antecedent_dict[var.name] = term
        elif isinstance(and_terms.right, AndTerms):
            self._extract_terms_from_and_terms(and_terms.right, antecedent_dict)
    
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


# Create classes to mimic skfuzzy's API for backward compatibility

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
        
        def input(self, input_dict):
            """Set input values."""
            self.input = input_dict
        
        def compute(self):
            """Compute the fuzzy inference."""
            # This is a mock implementation
            # In a real implementation, this would compute the fuzzy inference
            for consequent in self.control_system.consequents:
                self.output[consequent.label] = 5.0  # Dummy value
        
        def output(self):
            """Get output values."""
            return self.output
