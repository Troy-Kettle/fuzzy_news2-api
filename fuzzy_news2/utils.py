"""
Utility functions for the fuzzy-news2 package.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
import json
import os


def validate_range(value: Union[int, float], min_value: Union[int, float], 
                  max_value: Union[int, float], param_name: str) -> Union[int, float]:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        param_name: Name of the parameter for error message
    
    Returns:
        The validated value
    
    Raises:
        ValueError: If the value is not within the specified range
    """
    if value < min_value or value > max_value:
        raise ValueError(
            f"{param_name} must be between {min_value} and {max_value}, got {value}"
        )
    return value


def validate_consciousness(value: str) -> str:
    """
    Validate the consciousness level value.
    
    Args:
        value: Consciousness level (A, V, P, or U)
    
    Returns:
        The validated consciousness level
    
    Raises:
        ValueError: If the value is not one of the valid consciousness levels
    """
    valid_values = ["A", "V", "P", "U"]
    if value not in valid_values:
        raise ValueError(
            f"Consciousness level must be one of {valid_values}, got {value}"
        )
    return value


def format_result(result_dict: Dict, include_timestamp: bool = True) -> Dict:
    """
    Format a result dictionary for output.
    
    Args:
        result_dict: Dictionary containing the result
        include_timestamp: Whether to include a timestamp in the result
    
    Returns:
        Formatted result dictionary
    """
    formatted = result_dict.copy()
    
    if include_timestamp:
        formatted["timestamp"] = datetime.now().isoformat()
    
    return formatted


def save_result(result: Dict, patient_id: str, file_path: Optional[str] = None) -> str:
    """
    Save a result to a JSON file.
    
    Args:
        result: Result dictionary to save
        patient_id: ID of the patient
        file_path: Path to the file to save to (optional)
    
    Returns:
        Path to the saved file
    """
    if file_path is None:
        # Create a data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate a filename based on patient ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(data_dir, f"{patient_id}_{timestamp}.json")
    
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return file_path


def load_result(file_path: str) -> Dict:
    """
    Load a result from a JSON file.
    
    Args:
        file_path: Path to the file to load from
    
    Returns:
        Loaded result dictionary
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
    """
    with open(file_path, "r") as f:
        result = json.load(f)
    
    return result


def get_patient_history(patient_id: str, data_dir: Optional[str] = None) -> List[Dict]:
    """
    Get the history of assessments for a patient.
    
    Args:
        patient_id: ID of the patient
        data_dir: Directory to search for assessment files
    
    Returns:
        List of assessment dictionaries
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")
    
    if not os.path.exists(data_dir):
        return []
    
    results = []
    
    for filename in os.listdir(data_dir):
        if filename.startswith(f"{patient_id}_") and filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            try:
                result = load_result(file_path)
                results.append(result)
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue
    
    # Sort by timestamp if available
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return results
