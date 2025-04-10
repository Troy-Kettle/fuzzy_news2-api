"""
Tests for the utility functions.
"""

import os
import json
import tempfile
from datetime import datetime
import pytest

from fuzzy_news2.utils import (
    validate_range,
    validate_consciousness,
    format_result,
    save_result,
    load_result,
    get_patient_history
)


def test_validate_range():
    """Test range validation."""
    # Valid values
    assert validate_range(5, 0, 10, "test") == 5
    assert validate_range(0, 0, 10, "test") == 0
    assert validate_range(10, 0, 10, "test") == 10
    
    # Invalid values
    with pytest.raises(ValueError):
        validate_range(-1, 0, 10, "test")
    
    with pytest.raises(ValueError):
        validate_range(11, 0, 10, "test")


def test_validate_consciousness():
    """Test consciousness level validation."""
    # Valid values
    assert validate_consciousness("A") == "A"
    assert validate_consciousness("V") == "V"
    assert validate_consciousness("P") == "P"
    assert validate_consciousness("U") == "U"
    
    # Invalid values
    with pytest.raises(ValueError):
        validate_consciousness("X")


def test_format_result():
    """Test result formatting."""
    # Test with timestamp
    result = {"key": "value"}
    formatted = format_result(result)
    
    assert "key" in formatted
    assert formatted["key"] == "value"
    assert "timestamp" in formatted
    
    # Test without timestamp
    formatted = format_result(result, include_timestamp=False)
    assert "timestamp" not in formatted


def test_save_and_load_result():
    """Test saving and loading results."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Prepare test data
        result = {
            "patient_id": "TEST-001",
            "crisp_score": 5,
            "fuzzy_score": 5.7,
            "risk_category": "Medium",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save the result
        saved_path = save_result(result, "TEST-001", temp_path)
        assert saved_path == temp_path
        assert os.path.exists(saved_path)
        
        # Load the result
        loaded_result = load_result(saved_path)
        
        # Verify the loaded result
        assert loaded_result["patient_id"] == result["patient_id"]
        assert loaded_result["crisp_score"] == result["crisp_score"]
        assert loaded_result["fuzzy_score"] == result["fuzzy_score"]
        assert loaded_result["risk_category"] == result["risk_category"]
        assert loaded_result["timestamp"] == result["timestamp"]
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_save_result_default_path():
    """Test saving a result with default path."""
    # Prepare test data
    result = {
        "patient_id": "TEST-002",
        "crisp_score": 3,
        "fuzzy_score": 3.2,
        "risk_category": "Low-Medium",
        "timestamp": datetime.now().isoformat()
    }
    
    # Save the result with default path
    saved_path = save_result(result, "TEST-002")
    
    try:
        # Verify the file exists
        assert os.path.exists(saved_path)
        
        # Verify the file is in the data directory
        assert os.path.dirname(saved_path).endswith("data")
        
        # Verify the filename contains the patient ID
        assert "TEST-002" in os.path.basename(saved_path)
        
        # Load the result
        loaded_result = load_result(saved_path)
        
        # Verify the loaded result
        assert loaded_result["patient_id"] == result["patient_id"]
    
    finally:
        # Clean up
        if os.path.exists(saved_path):
            os.remove(saved_path)


def test_get_patient_history():
    """Test retrieving patient history."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        for i in range(5):
            result = {
                "patient_id": "TEST-003",
                "crisp_score": i,
                "fuzzy_score": float(i) + 0.5,
                "risk_category": "Low" if i < 3 else "Medium",
                "timestamp": (datetime.now().isoformat()[:-6] + 
                             f"{i:03}000")  # Make timestamps predictably ordered
            }
            
            file_path = os.path.join(temp_dir, f"TEST-003_{i}.json")
            with open(file_path, "w") as f:
                json.dump(result, f)
        
        # Also create a file for a different patient
        other_result = {
            "patient_id": "OTHER",
            "crisp_score": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        other_path = os.path.join(temp_dir, "OTHER_0.json")
        with open(other_path, "w") as f:
            json.dump(other_result, f)
        
        # Get history for TEST-003
        history = get_patient_history("TEST-003", temp_dir)
        
        # Verify the history
        assert len(history) == 5
        assert all(item["patient_id"] == "TEST-003" for item in history)
        
        # Verify the history is sorted by timestamp (newest first)
        timestamps = [item["timestamp"] for item in history]
        assert timestamps == sorted(timestamps, reverse=True)
        
        # Get history for OTHER
        other_history = get_patient_history("OTHER", temp_dir)
        assert len(other_history) == 1
        assert other_history[0]["patient_id"] == "OTHER"
        
        # Get history for non-existent patient
        nonexistent_history = get_patient_history("NONEXISTENT", temp_dir)
        assert len(nonexistent_history) == 0


def test_get_patient_history_nonexistent_directory():
    """Test retrieving history from a non-existent directory."""
    # Get history from a non-existent directory
    history = get_patient_history("TEST", "/nonexistent/directory")
    assert len(history) == 0


def test_load_result_nonexistent_file():
    """Test loading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_result("/nonexistent/file.json")


def test_load_result_invalid_json():
    """Test loading an invalid JSON file."""
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(b"invalid json")
        temp_path = temp_file.name
    
    try:
        with pytest.raises(json.JSONDecodeError):
            load_result(temp_path)
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
