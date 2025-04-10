"""
Tests for the API endpoints.
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from fuzzy_news2.api import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def sample_vitals():
    """Create sample vital signs for testing."""
    return {
        "patient_id": "TEST-001",
        "respiratory_rate": 18,
        "oxygen_saturation": 96,
        "systolic_bp": 130,
        "pulse": 72,
        "consciousness": "A",
        "temperature": 37.1,
        "supplemental_oxygen": False
    }


@pytest.fixture
def abnormal_vitals():
    """Create abnormal vital signs for testing."""
    return {
        "patient_id": "TEST-001",
        "respiratory_rate": 26,
        "oxygen_saturation": 91,
        "systolic_bp": 95,
        "pulse": 115,
        "consciousness": "V",
        "temperature": 39.2,
        "supplemental_oxygen": True
    }


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_calculate_normal(client, sample_vitals):
    """Test calculation with normal vital signs."""
    response = client.post("/api/calculate", json=sample_vitals)
    assert response.status_code == 200
    data = response.json()
    
    assert data["patient_id"] == sample_vitals["patient_id"]
    assert "timestamp" in data
    assert "crisp_score" in data
    assert "fuzzy_score" in data
    assert "risk_category" in data
    assert "recommended_response" in data
    assert "parameter_scores" in data
    
    # Normal vitals should result in a low risk
    assert data["risk_category"] in ["Low", "Low-Medium"]


def test_calculate_abnormal(client, abnormal_vitals):
    """Test calculation with abnormal vital signs."""
    response = client.post("/api/calculate", json=abnormal_vitals)
    assert response.status_code == 200
    data = response.json()
    
    # Abnormal vitals should result in a higher risk
    assert data["risk_category"] in ["Medium", "High"]
    assert data["crisp_score"] > 0


def test_validation(client, sample_vitals):
    """Test input validation."""
    # Test invalid respiratory rate
    invalid_vitals = sample_vitals.copy()
    invalid_vitals["respiratory_rate"] = 60  # Out of range
    
    response = client.post("/api/calculate", json=invalid_vitals)
    assert response.status_code == 400
    
    # Test invalid consciousness
    invalid_vitals = sample_vitals.copy()
    invalid_vitals["consciousness"] = "X"  # Invalid value
    
    response = client.post("/api/calculate", json=invalid_vitals)
    assert response.status_code == 400


def test_get_history(client, sample_vitals, abnormal_vitals):
    """Test retrieving patient history."""
    # Create some history by making multiple calculations
    client.post("/api/calculate", json=sample_vitals)
    client.post("/api/calculate", json=abnormal_vitals)
    
    # Get history
    response = client.get(f"/api/history/{sample_vitals['patient_id']}")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # Should have at least the two we just created
    
    # Verify data is sorted by timestamp (most recent first)
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in data]
    assert all(timestamps[i] >= timestamps[i+1] for i in range(len(timestamps)-1))


def test_get_statistics(client, sample_vitals, abnormal_vitals):
    """Test retrieving patient statistics."""
    # Create some history
    client.post("/api/calculate", json=sample_vitals)
    client.post("/api/calculate", json=abnormal_vitals)
    
    # Get statistics
    response = client.get(f"/api/statistics/{sample_vitals['patient_id']}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == sample_vitals["patient_id"]
    assert "assessments_count" in data
    assert "average_crisp_score" in data
    assert "average_fuzzy_score" in data
    assert "max_crisp_score" in data
    assert "max_fuzzy_score" in data
    assert "trend" in data
    
    # Should have at least 2 assessments
    assert data["assessments_count"] >= 2


def test_limit_history(client, sample_vitals):
    """Test limiting the number of history items returned."""
    # Create multiple calculations
    for _ in range(5):
        client.post("/api/calculate", json=sample_vitals)
    
    # Get history with limit
    response = client.get(f"/api/history/{sample_vitals['patient_id']}?limit=3")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 3  # Should respect the limit


def test_days_filter_statistics(client, sample_vitals):
    """Test filtering statistics by days."""
    # Get statistics with custom days filter
    response = client.get(f"/api/statistics/{sample_vitals['patient_id']}?days=14")
    assert response.status_code == 200
    
    data = response.json()
    assert data["days"] == 14


def test_invalid_patient_id(client):
    """Test retrieving history for non-existent patient."""
    response = client.get("/api/history/NONEXISTENT")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0  # Should be empty
