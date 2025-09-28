from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import re
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will update after frontend deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (will be loaded on startup)
rf = None
le_crop = None
le_season = None
le_state = None
soil_df = None

def load_models():
    """Load models on startup - handle errors gracefully"""
    global rf, le_crop, le_season, le_state, soil_df
    
    try:
        # Try to load models if they exist
        if os.path.exists('le_crop.joblib'):
            le_crop = joblib.load('le_crop.joblib')
        if os.path.exists('le_season.joblib'):
            le_season = joblib.load('le_season.joblib')
        if os.path.exists('le_state.joblib'):
            le_state = joblib.load('le_state.joblib')
        if os.path.exists('state_soil_data.csv'):
            soil_df = pd.read_csv('state_soil_data.csv')
            soil_df['state'] = soil_df['state'].str.strip()
        
        # Try to load main model
        if os.path.exists('random_forest_model.joblib'):
            rf = joblib.load('random_forest_model.joblib')
        
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

# Load models on startup
load_models()

# Typical crop rainfall (mm)
crop_rainfall_needs = {
    "Rice": 1200,
    "Wheat": 500,
    "Maize": 600,
    "Sugarcane": 1500,
    "Cotton(lint)": 700,
    "Jowar": 600,
    "Groundnut": 500,
    "Potato": 500,
    "Soyabean": 700,
    "Pulses": 400,
    "Gram": 450,
    "Barley": 500,
    "Onion": 600,
    "default": 700
}

def parse_float(s, default=0.0):
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return default
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    return float(m.group()) if m else default

class RecommendRequest(BaseModel):
    crops: List[str]
    season: str
    location: str
    year: Optional[Union[str, int]] = None
    area: Union[str, float, int]
    temp: Union[str, float, int]
    rainfall: Union[str, float, int]
    humidity: Union[str, float, int]
    fertilizer: Optional[Union[str, float, int]] = None
    pesticides: Optional[Union[str, float, int]] = None

class PredictRequest(BaseModel):
    crop: str
    season: str
    location: str
    year: Optional[Union[str, int]] = None
    area: Union[str, float, int]
    fertilizer: Union[str, float, int]
    pesticides: Union[str, float, int]
    temp: Union[str, float, int]
    rainfall: Union[str, float, int]
    humidity: Union[str, float, int]
    soil: Optional[str] = None

def get_crop_water_need(crop_name):
    key = crop_name.strip().lower()
    for k in crop_rainfall_needs.keys():
        if k.lower() == key:
            return crop_rainfall_needs[k]
    return crop_rainfall_needs["default"]

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Crop Prediction API", 
        "status": "running",
        "models_loaded": rf is not None
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if rf is not None else "not_loaded"
    return {
        "status": "healthy", 
        "model_status": model_status,
        "message": "API is running"
    }

@app.post("/predict")
def predict_yield(data: PredictRequest):
    """Predict crop yield"""
    try:
        # Check if models are loaded
        if rf is None or le_crop is None or le_season is None or le_state is None:
            return {
                "error": "Models not loaded",
                "status": "error",
                "message": "ML models are not available. Please contact administrator."
            }
        
        if soil_df is None:
            return {
                "error": "Soil data not available",
                "status": "error"
            }

        # Parse input values
        area = parse_float(data.area)
        fertilizer = parse_float(data.fertilizer)
        pesticides = parse_float(data.pesticides)
        temp = parse_float(data.temp)
        rainfall = parse_float(data.rainfall)
        humidity = parse_float(data.humidity)

        # Get soil data
        state_soil = soil_df[soil_df['state'] == data.location.strip()]
        if state_soil.empty:
            return {
                "error": f"Soil data for location '{data.location}' not found",
                "available_locations": soil_df['state'].tolist()[:10]  # Show first 10
            }
        
        N = state_soil['N'].values[0]
        P = state_soil['P'].values[0] 
        K = state_soil['K'].values[0]
        pH = state_soil['pH'].values[0]

        # Encode categorical variables
        try:
            crop_enc = le_crop.transform([data.crop])[0]
            season_enc = le_season.transform([data.season])[0]
            state_enc = le_state.transform([data.location])[0]
        except ValueError as e:
            return {
                "error": f"Encoding error: {str(e)}",
                "available_crops": list(le_crop.classes_)[:10],
                "available_seasons": list(le_season.classes_),
                "available_states": list(le_state.classes_)[:10]
            }

        # Create feature array
        features = np.array([[
            crop_enc, season_enc, state_enc, area, fertilizer, pesticides,
            N, P, K, pH, temp, rainfall, humidity
        ]])

        # Make prediction
        prediction = rf.predict(features)[0]

        # Generate recommendations
        recommendations = [
            f"Estimated yield: {round(float(prediction), 2)} tonnes/hectare",
            f"Based on {area} hectares in {data.location}",
        ]

        if temp < 15:
            recommendations.append("Temperature is low - consider cold-resistant varieties")
        elif temp > 35:
            recommendations.append("Temperature is high - ensure adequate irrigation")
            
        if rainfall < 300:
            recommendations.append("Low rainfall - irrigation essential")

        return {
            "yield": round(float(prediction), 2),
            "predicted_yield": round(float(prediction), 2),
            "recommendations": recommendations,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "status": "error"
        }

@app.post("/recommend_multi_crop/")
def recommend_multiple_crops(data: RecommendRequest):
    """Multi-crop recommendation endpoint"""
    try:
        if rf is None:
            return {"error": "ML model not loaded"}
        
        # Simplified response for now
        return {
            "message": "Multi-crop recommendation endpoint", 
            "status": "available",
            "note": "Full implementation requires model files"
        }
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

# Vercel serverless function handler
def handler(request):
    """Handler for Vercel serverless functions"""
    return app

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
