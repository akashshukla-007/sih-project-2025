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
import uvicorn  # Add this import

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive link for the model file (replace with your actual link)
MODEL_FILE_URL = 'https://drive.google.com/file/d/10RJCY5Mb6185Xjwlj8j_LYFiBpKlBnGg/view?usp=drive_link'

def download_model():
    """Download model file from Google Drive if not exists"""
    model_path = 'random_forest_model.joblib'
    
    if os.path.exists(model_path):
        print("Model already exists, skipping download")
        return model_path
    
    try:
        print("Downloading ML model from Google Drive...")
        response = requests.get(MODEL_FILE_URL, timeout=300)  # 5 min timeout
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully: {len(response.content)} bytes")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=503, detail="Could not load ML model")

# Download and load model on startup
try:
    model_path = download_model()
    rf = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    rf = None

# Load other files (these are small, can be in GitHub)
le_crop = joblib.load('le_crop.joblib')
le_season = joblib.load('le_season.joblib')
le_state = joblib.load('le_state.joblib')

# Load soil data
soil_df = pd.read_csv('state_soil_data.csv')
soil_df['state'] = soil_df['state'].str.strip()

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

# KEEP ONLY THIS PredictRequest CLASS (with soil field)
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

def compare_input(user_val, rec_val, nutrient_name):
    if user_val is None:
        return f"No {nutrient_name} input provided."
    try:
        user_val = float(user_val)
        rec_val = float(rec_val)
    except:
        return f"No {nutrient_name} input provided."
    diff = rec_val - user_val
    if abs(diff) < 0.1 * rec_val:
        return f"Your {nutrient_name} input is near optimal."
    elif diff > 0:
        return f"Consider increasing {nutrient_name} by {round(diff, 2)} units for better yield."
    else:
        return f"Your {nutrient_name} input is higher than recommended by {round(-diff, 2)} units; consider reducing it to save cost."

@app.get("/health")
def health_check():
    """Check if model is loaded"""
    if rf is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy", "message": "Model loaded successfully"}

@app.post("/recommend_multi_crop/")
def recommend_multiple_crops(data: RecommendRequest):
    if rf is None:
        raise HTTPException(status_code=503, detail="ML model not loaded. Please wait and try again.")
        
    area = parse_float(data.area)
    temp = parse_float(data.temp)
    rainfall = parse_float(data.rainfall)
    humidity = parse_float(data.humidity)
    year = int(parse_float(data.year, default=2025)) if data.year else 2025
    user_fert = parse_float(data.fertilizer) if data.fertilizer is not None else None
    user_pest = parse_float(data.pesticides) if data.pesticides is not None else None

    state_soil = soil_df[soil_df['state'] == data.location.strip()]
    if state_soil.empty:
        return {"error": f"Soil data for location '{data.location}' not found."}
    N = state_soil['N'].values[0]
    P = state_soil['P'].values[0]
    K = state_soil['K'].values[0]
    pH = state_soil['pH'].values[0]

    try:
        season_enc = le_season.transform([data.season])[0]
        state_enc = le_state.transform([data.location])[0]
    except ValueError as e:
        return {"error": f"Encoding error: {str(e)}. Check input values."}

    fertilizer_range = np.linspace(5000, 100000, 10)
    pesticide_range = np.linspace(500, 12000, 10)

    recommendations = []
    best_crop = None
    best_crop_yield = -np.inf

    for crop in data.crops:
        try:
            crop_enc = le_crop.transform([crop])[0]
        except ValueError:
            continue

        if user_fert is not None and user_pest is not None:
            features = np.array([[
                crop_enc, season_enc, state_enc, area, user_fert, user_pest,
                N, P, K, pH, temp, rainfall, humidity
            ]])
            best_yield = rf.predict(features)[0]
            best_inputs = {
                "fertilizer": user_fert,
                "pesticide": user_pest,
                "predicted_yield": round(float(best_yield), 3)
            }
        else:
            best_yield = -np.inf
            best_inputs = {}
            for fert in fertilizer_range:
                for pest in pesticide_range:
                    features = np.array([[
                        crop_enc, season_enc, state_enc, area, fert, pest,
                        N, P, K, pH, temp, rainfall, humidity
                    ]])
                    pred_yield = rf.predict(features)[0]
                    if pred_yield > best_yield:
                        best_yield = pred_yield
                        best_inputs = {
                            "fertilizer": round(float(fert), 2),
                            "pesticide": round(float(pest), 2),
                            "predicted_yield": round(float(pred_yield), 3)
                        }

        crop_need = get_crop_water_need(crop)
        if rainfall >= crop_need:
            irrigation_advice = "Rainfall sufficient: No irrigation needed."
        elif rainfall >= crop_need * 0.7:
            irrigation_advice = "Supplemental irrigation recommended during dry spells."
        else:
            irrigation_advice = "Irrigation essential: Rainfall below optimal for this crop."

        fert_msg = compare_input(user_fert, best_inputs.get("fertilizer"), "fertilizer")
        pest_msg = compare_input(user_pest, best_inputs.get("pesticide"), "pesticide")

        recommendations.append({
            "crop": crop,
            "fertilizer": best_inputs.get("fertilizer"),
            "fertilizer_message": fert_msg,
            "pesticide": best_inputs.get("pesticide"),
            "pesticide_message": pest_msg,
            "predicted_yield": best_inputs.get("predicted_yield"),
            "irrigation_advice": irrigation_advice
        })

        if best_inputs.get("predicted_yield") > best_crop_yield:
            best_crop_yield = best_inputs.get("predicted_yield")
            best_crop = {
                "crop": crop,
                "predicted_yield": best_crop_yield,
                "fertilizer": best_inputs.get("fertilizer"),
                "fertilizer_message": fert_msg,
                "pesticide": best_inputs.get("pesticide"),
                "pesticide_message": pest_msg,
                "irrigation_advice": irrigation_advice
            }

    recommendations.sort(key=lambda x: x["predicted_yield"], reverse=True)

    return{
        "best_crop_recommendation": best_crop,
        "all_recommendations": recommendations
    }

@app.post("/predict")
def predict_yield(data: PredictRequest):
    if rf is None:
        raise HTTPException(status_code=503, detail="ML model not loaded. Please wait and try again.")
        
    print(f"Received request data: {data}")
    
    try:
        area = parse_float(data.area)
        fertilizer = parse_float(data.fertilizer)
        pesticides = parse_float(data.pesticides)
        temp = parse_float(data.temp)
        rainfall = parse_float(data.rainfall)
        humidity = parse_float(data.humidity)
        year = int(parse_float(data.year, default=2025)) if data.year else 2025

        print(f"Parsed values - Area: {area}, Temp: {temp}, Rainfall: {rainfall}, Fertilizer: {fertilizer}, Pesticides: {pesticides}")

        state_soil = soil_df[soil_df['state'] == data.location.strip()]
        if state_soil.empty:
            print(f"Available states in soil_df: {soil_df['state'].tolist()}")
            raise HTTPException(status_code=404, detail=f"Soil data for location '{data.location}' not found.")
        
        N = state_soil['N'].values[0]
        P = state_soil['P'].values[0]
        K = state_soil['K'].values[0]
        pH = state_soil['pH'].values[0]

        print(f"Soil data - N: {N}, P: {P}, K: {K}, pH: {pH}")

        try:
            crop_enc = le_crop.transform([data.crop])[0]
            season_enc = le_season.transform([data.season])[0]
            state_enc = le_state.transform([data.location])[0]
            
            print(f"Encoded values - Crop: {crop_enc}, Season: {season_enc}, State: {state_enc}")
            
        except ValueError as e:
            print(f"Encoding error details: {str(e)}")
            print(f"Available crops: {list(le_crop.classes_)}")
            print(f"Available seasons: {list(le_season.classes_)}")
            print(f"Available states: {list(le_state.classes_)}")
            raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}. Check input values.")

        features = np.array([[
            crop_enc, season_enc, state_enc, area, fertilizer, pesticides,
            N, P, K, pH, temp, rainfall, humidity
        ]])

        print(f"Feature array shape: {features.shape}")
        print(f"Feature values: {features[0]}")

        prediction = rf.predict(features)[0]
        print(f"Raw prediction: {prediction}")

        recommendations = [
            f"Estimated yield for {data.crop}: {round(float(prediction), 2)} tonnes/hectare",
            f"Based on {area} hectares in {data.location}",
            f"Fertilizer usage: {fertilizer} g/ha, Pesticides: {pesticides} g/ha"
        ]

        if temp < 15:
            recommendations.append("Temperature is low - consider using cold-resistant varieties")
        elif temp > 35:
            recommendations.append("Temperature is high - ensure adequate irrigation")
            
        if rainfall < 300:
            recommendations.append("Low rainfall detected - irrigation is essential")
        elif rainfall > 1500:
            recommendations.append("High rainfall - ensure proper drainage")

        response = {
            "yield": round(float(prediction), 2),
            "predicted_yield": round(float(prediction), 2),
            "recommendations": recommendations,
            "status": "success"
        }
        
        print("Response to frontend:", response)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add this main section for Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port: {port}")
    uvicorn.run(
        "bend:app",  # Updated to match your filename
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )
