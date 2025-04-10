"""
Shared test fixtures for the fuzzy-news2 package.
"""

import os
import tempfile
import json
import pytest
from datetime import datetime, timedelta

from fuzzy_news2 import FuzzyNEWS2


@pytest.fixture
def fuzzy_news():
    """Create a FuzzyNEWS2 instance for testing."""
    return FuzzyNEWS2()


@pytest.fixture
def normal_vitals():
    """Normal vital signs for testing."""
    return {
        "respiratory_rate": 14,
        "oxygen_saturation": 97,
        "systolic_bp": 120,
        "pulse": 72,
        "consciousness": "A",
        "temperature": 37.0,
        "supplemental_oxygen": False
    }


@pytest.fixture
def abnormal_vitals():
    """Abnormal vital signs for testing."""
    return {
        "respiratory_rate": 28,
        "oxygen_saturation": 89,
        "systolic_bp": 90,
        "pulse": 135,
        "consciousness": "V",
        "temperature": 39.3,
        "supplemental_oxygen": True
    }


@pytest.fixture
def borderline_vitals():
    """Borderline vital signs for testing."""
    return {
        "respiratory_rate": 21,
        "oxygen_saturation": 94,
        "systolic_bp": 108,
        "pulse": 92,
        "consciousness": "A",
        "temperature": 38.2,
        "supplemental_oxygen": False
    }


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up environment to use this as the data directory
        original_cwd = os.getcwd()
        try:
            # Create data subdirectory
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Change working directory to temp directory
            os.chdir(temp_dir)
            
            yield data_dir
        
        finally:
            # Reset working directory
            os.chdir(original_cwd)


@pytest.fixture
def sample_patient_history(temp_data_dir):
    """Create sample patient history for testing."""
    patient_id = "TEST-PATIENT"
    base_date = datetime.now()
    
    history = []
    
    # Create 10 records over the past 30 days
    for i in range(10):
        timestamp = (base_date - timedelta(days=i*3)).isoformat()
        
        record = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "crisp_score": 10 - i if i <= 5 else 5,  # Improving trend after day 15
            "fuzzy_score": float(10 - i) + 0.5 if i <= 5 else float(5) + 0.5,
            "risk_category": "High" if i <= 2 else ("Medium" if i <= 5 else "Low-Medium"),
            "recommended_response": "Urgent review" if i <= 2 else "Routine monitoring",
            "parameter_scores": {
                "respiratory_rate": 2 if i <= 5 else 1,
                "oxygen_saturation": 2 if i <= 3 else 1,
                "supplemental_oxygen": 2 if i <= 5 else 0,
                "systolic_bp": 1,
                "pulse": 2 if i <= 4 else 1,
                "consciousness": 0,
                "temperature": 1,
                "total": 10 - i if i <= 5 else 5
            }
        }
        
        # Save record to file
        filename = f"{patient_id}_{timestamp.replace(':', '-')}.json"
        file_path = os.path.join(temp_data_dir, filename)
        
        with open(file_path, "w") as f:
            json.dump(record, f)
        
        history.append(record)
    
    return {
        "patient_id": patient_id,
        "records": history,
        "data_dir": temp_data_dir
    }
