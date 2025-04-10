"""
API endpoints for the fuzzy-news2 package.
This module provides a REST API for integration with an Electron front-end.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .news2 import FuzzyNEWS2, NEWS2Result
from .utils import (
    validate_range, 
    validate_consciousness,
    format_result,
    save_result,
    load_result,
    get_patient_history
)


# Create the FastAPI app
app = FastAPI(
    title="Fuzzy NEWS-2 API",
    description="API for fuzzy logic implementation of NEWS-2 score",
    version="0.1.0",
)

# Add CORS middleware for Electron integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the FuzzyNEWS2 instance
fuzzy_news = FuzzyNEWS2()


class PatientVitals(BaseModel):
    """Model for patient vital signs."""
    patient_id: str = Field(..., description="Patient ID")
    respiratory_rate: int = Field(..., description="Respiratory rate (breaths per minute)")
    oxygen_saturation: int = Field(..., description="Oxygen saturation (%)")
    systolic_bp: int = Field(..., description="Systolic blood pressure (mmHg)")
    pulse: int = Field(..., description="Pulse rate (beats per minute)")
    consciousness: str = Field(..., description="Level of consciousness (A, V, P, or U)")
    temperature: float = Field(..., description="Body temperature (°C)")
    supplemental_oxygen: bool = Field(False, description="Whether supplemental oxygen is being used")
    
    @validator("respiratory_rate")
    def validate_respiratory_rate(cls, v):
        return validate_range(v, 0, 50, "Respiratory rate")
    
    @validator("oxygen_saturation")
    def validate_oxygen_saturation(cls, v):
        return validate_range(v, 70, 100, "Oxygen saturation")
    
    @validator("systolic_bp")
    def validate_systolic_bp(cls, v):
        return validate_range(v, 50, 250, "Systolic blood pressure")
    
    @validator("pulse")
    def validate_pulse(cls, v):
        return validate_range(v, 20, 180, "Pulse rate")
    
    @validator("consciousness")
    def validate_consciousness_level(cls, v):
        return validate_consciousness(v)
    
    @validator("temperature")
    def validate_temperature(cls, v):
        return validate_range(v, 33.0, 43.0, "Temperature")


class NEWS2Response(BaseModel):
    """Model for NEWS-2 response."""
    patient_id: str = Field(..., description="Patient ID")
    timestamp: str = Field(..., description="Timestamp of assessment")
    crisp_score: int = Field(..., description="Traditional NEWS-2 score")
    fuzzy_score: float = Field(..., description="Fuzzy NEWS-2 score")
    risk_category: str = Field(..., description="Risk category")
    recommended_response: str = Field(..., description="Recommended clinical response")
    parameter_scores: Dict[str, Union[int, float]] = Field(..., description="Individual parameter scores")


@app.post("/api/calculate", response_model=NEWS2Response)
async def calculate_news2(vitals: PatientVitals):
    """
    Calculate the NEWS-2 score using fuzzy logic.
    
    Args:
        vitals: Patient vital signs
    
    Returns:
        NEWS2Response: NEWS-2 score and recommendations
    """
    try:
        result = fuzzy_news.calculate(
            respiratory_rate=vitals.respiratory_rate,
            oxygen_saturation=vitals.oxygen_saturation,
            systolic_bp=vitals.systolic_bp,
            pulse=vitals.pulse,
            consciousness=vitals.consciousness,
            temperature=vitals.temperature,
            supplemental_oxygen=vitals.supplemental_oxygen
        )
        
        # Format the result
        response = {
            "patient_id": vitals.patient_id,
            "timestamp": datetime.now().isoformat(),
            "crisp_score": result.crisp_score,
            "fuzzy_score": result.fuzzy_score,
            "risk_category": result.risk_category,
            "recommended_response": result.recommended_response,
            "parameter_scores": result.parameter_scores
        }
        
        # Save the result
        save_result(response, vitals.patient_id)
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating NEWS-2 score: {str(e)}")


@app.get("/api/history/{patient_id}", response_model=List[NEWS2Response])
async def get_history(patient_id: str, limit: int = Query(10, ge=1, le=100)):
    """
    Get the history of assessments for a patient.
    
    Args:
        patient_id: ID of the patient
        limit: Maximum number of assessments to return
    
    Returns:
        List of assessment results
    """
    try:
        history = get_patient_history(patient_id)
        return history[:limit]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/api/statistics/{patient_id}")
async def get_statistics(patient_id: str, days: int = Query(7, ge=1, le=365)):
    """
    Get statistics for a patient over a period of time.
    
    Args:
        patient_id: ID of the patient
        days: Number of days to include in statistics
    
    Returns:
        Statistics for the patient
    """
    try:
        history = get_patient_history(patient_id)
        
        # Filter to only include the last 'days' days
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_history = [
            assessment for assessment in history
            if datetime.fromisoformat(assessment["timestamp"]).timestamp() > cutoff
        ]
        
        # Calculate statistics
        if not recent_history:
            return {
                "patient_id": patient_id,
                "days": days,
                "assessments_count": 0,
                "average_crisp_score": None,
                "average_fuzzy_score": None,
                "max_crisp_score": None,
                "max_fuzzy_score": None,
                "trend": "No data available"
            }
        
        crisp_scores = [assessment["crisp_score"] for assessment in recent_history]
        fuzzy_scores = [assessment["fuzzy_score"] for assessment in recent_history]
        
        # Determine trend
        if len(recent_history) >= 2:
            first_score = recent_history[-1]["crisp_score"]
            last_score = recent_history[0]["crisp_score"]
            
            if last_score < first_score:
                trend = "Improving"
            elif last_score > first_score:
                trend = "Worsening"
            else:
                trend = "Stable"
        else:
            trend = "Not enough data"
        
        return {
            "patient_id": patient_id,
            "days": days,
            "assessments_count": len(recent_history),
            "average_crisp_score": sum(crisp_scores) / len(crisp_scores),
            "average_fuzzy_score": sum(fuzzy_scores) / len(fuzzy_scores),
            "max_crisp_score": max(crisp_scores),
            "max_fuzzy_score": max(fuzzy_scores),
            "trend": trend
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    }


def start():
    """Start the API server."""
    uvicorn.run("fuzzy_news2.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()

