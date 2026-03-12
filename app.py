from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import json
from datetime import datetime
import os
from typing import Optional
import logging
from pathlib import Path
import traceback  # Add this for better error tracking

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger(__name__)

app = FastAPI(title="GLD Price Predictor API", 
              description="ML Model for Gold Price Prediction",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates directory if it doesn't exist
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create static directory for additional assets
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Load model with error handling
try:
    model_path = "gld_price_model.pkl"
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} not found. Creating a dummy model for testing.")
        from sklearn.linear_model import LinearRegression
        dummy_model = LinearRegression()
        X_dummy = np.random.rand(100, 4)
        # More realistic coefficients for GLD price (typically around $150-200)
        true_coef = np.array([0.02, 0.5, 1.2, 30.0])  # Adjusted coefficients
        y_dummy = X_dummy @ true_coef + 150 + np.random.randn(100) * 5
        dummy_model.fit(X_dummy, y_dummy)
        model = dummy_model
        logger.info(f"Dummy model created with coefficients: {model.coef_}")
    else:
        model = pickle.load(open(model_path, "rb"))
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    # Set coefficients manually
    model.coef_ = np.array([0.02, 0.5, 1.2, 30.0])
    model.intercept_ = 150.0
    logger.info("Fallback model created with hardcoded coefficients")

DB_FILE = "predictions.json"

def save_prediction(entry):
    """Save prediction to JSON file"""
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(entry)
        
        if len(data) > 100:
            data = data[-100:]
        
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"Prediction saved successfully: {entry['id']}")
        return True
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return False

def get_prediction_history(limit: int = 10):
    """Get recent prediction history"""
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r") as f:
                data = json.load(f)
            return data[-limit:] if data else []
        return []
    except Exception as e:
        logger.error(f"Error reading prediction history: {e}")
        return []

# ==================== PAGE ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    history = get_prediction_history(100)
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "app_name": "GLD Price Predictor",
            "current_year": datetime.now().year,
            "total_predictions": len(history),
            "active_page": "home"
        }
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard page"""
    history = get_prediction_history(100)
    predictions = history[-10:] if history else []
    
    if predictions:
        pred_values = [p['prediction'] for p in predictions]
        stats = {
            "total": len(history),
            "avg_price": f"{sum(pred_values)/len(pred_values):.2f}",
            "max_price": f"{max(pred_values):.2f}",
            "min_price": f"{min(pred_values):.2f}"
        }
    else:
        stats = {
            "total": 0,
            "avg_price": "0.00",
            "max_price": "0.00",
            "min_price": "0.00"
        }
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "predictions": predictions,
            "stats": stats,
            "active_page": "dashboard"
        }
    )

@app.get("/input", response_class=HTMLResponse)
async def input_page(request: Request):
    """Serve the input prediction page"""
    return templates.TemplateResponse(
        "input.html",
        {
            "request": request,
            "active_page": "input"
        }
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """Serve the about page"""
    history = get_prediction_history(100)
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "total_predictions": len(history),
            "active_page": "about"
        }
    )

# ==================== API ROUTES ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_type": str(type(model))
    }

# Test endpoint to verify model is working
@app.get("/api/test")
async def test():
    """Test endpoint to verify API is working"""
    return {
        "success": True,
        "message": "API is working",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/predict")
async def predict(
    spx: float, 
    uso: float, 
    slv: float, 
    eur_usd: float,
    request: Request
):
    """Predict GLD price based on input features"""
    try:
        logger.info(f"Prediction request received: SPX={spx}, USO={uso}, SLV={slv}, EUR/USD={eur_usd}")
        
        # Input validation
        if spx <= 0 or uso <= 0 or slv <= 0 or eur_usd <= 0:
            logger.warning(f"Invalid input values: all must be positive")
            raise HTTPException(status_code=400, detail="All values must be positive")
        
        # Prepare input data
        input_data = np.array([[spx, uso, slv, eur_usd]])
        logger.info(f"Input array shape: {input_data.shape}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Ensure prediction is reasonable (GLD typically trades between $100-$200)
        if prediction < 50 or prediction > 500:
            logger.warning(f"Unusual prediction value: {prediction}")
        
        # Calculate confidence (simplified approach without using score())
        # You can implement your own confidence logic here
        # For now, we'll use a simple confidence score based on input ranges
        confidence = 85  # Default confidence
        
        # Optional: Adjust confidence based on how "normal" the inputs are
        if 4000 <= spx <= 5000 and 60 <= uso <= 90 and 18 <= slv <= 28 and 1.00 <= eur_usd <= 1.20:
            confidence = 92  # Higher confidence for typical market ranges
        elif spx < 3000 or spx > 6000 or uso < 40 or uso > 120 or slv < 10 or slv > 40 or eur_usd < 0.8 or eur_usd > 1.4:
            confidence = 70  # Lower confidence for extreme values
        
        # Create entry for history
        entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "spx": spx,
            "uso": uso,
            "slv": slv,
            "eur_usd": eur_usd,
            "prediction": float(prediction),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "client_host": request.client.host if request.client else "unknown"
        }
        
        # Save to history
        save_prediction(entry)
        
        logger.info(f"Prediction successful: ${prediction:.2f} with {confidence}% confidence")
        
        return {
            "success": True,
            "prediction": float(prediction),
            "confidence": confidence,
            "timestamp": entry["timestamp"],
            "id": entry["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/predict/batch")
async def predict_batch(predictions: list):
    """Batch prediction endpoint"""
    try:
        results = []
        for pred in predictions:
            input_data = np.array([[pred['spx'], pred['uso'], pred['slv'], pred['eur_usd']]])
            prediction = model.predict(input_data)[0]
            results.append({
                "input": pred,
                "prediction": float(prediction)
            })
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(limit: int = 10):
    """Get prediction history"""
    try:
        history = get_prediction_history(limit)
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{prediction_id}")
async def get_prediction_by_id(prediction_id: str):
    """Get specific prediction by ID"""
    try:
        history = get_prediction_history(100)
        prediction = next((p for p in history if p['id'] == prediction_id), None)
        
        if prediction:
            return {
                "success": True,
                "prediction": prediction
            }
        else:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get prediction statistics"""
    try:
        history = get_prediction_history(100)
        
        if not history:
            return {
                "success": True,
                "total_predictions": 0,
                "average_prediction": 0,
                "min_prediction": 0,
                "max_prediction": 0
            }
        
        predictions = [p['prediction'] for p in history]
        
        stats = {
            "success": True,
            "total_predictions": len(history),
            "average_prediction": float(np.mean(predictions)),
            "median_prediction": float(np.median(predictions)),
            "min_prediction": float(np.min(predictions)),
            "max_prediction": float(np.max(predictions)),
            "std_deviation": float(np.std(predictions))
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    """Get model information"""
    try:
        model_info = {
            "success": True,
            "model_type": type(model).__name__,
            "features": ["SPX", "USO", "SLV", "EUR/USD"],
            "target": "GLD Price"
        }
        
        if hasattr(model, 'coef_'):
            model_info["coefficients"] = {
                "spx": float(model.coef_[0]) if len(model.coef_) > 0 else 0,
                "uso": float(model.coef_[1]) if len(model.coef_) > 1 else 0,
                "slv": float(model.coef_[2]) if len(model.coef_) > 2 else 0,
                "eur_usd": float(model.coef_[3]) if len(model.coef_) > 3 else 0
            }
        
        if hasattr(model, 'intercept_'):
            model_info["intercept"] = float(model.intercept_)
        
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/clear")
async def clear_history():
    """Clear all prediction history"""
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info("Prediction history cleared")
            return {"success": True, "message": "History cleared successfully"}
        else:
            return {"success": True, "message": "No history to clear"}
            
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up GLD Price Predictor API")
    logger.info(f"Model loaded: {type(model).__name__}")
    
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump([], f)
        logger.info(f"Created {DB_FILE}")
    
    # Test the model with a sample prediction
    try:
        test_input = np.array([[4500, 75, 22.5, 1.08]])
        test_pred = model.predict(test_input)[0]
        logger.info(f"Test prediction successful: ${test_pred:.2f}")
    except Exception as e:
        logger.error(f"Test prediction failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down GLD Price Predictor API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )