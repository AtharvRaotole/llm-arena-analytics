"""
FastAPI application for LLM Arena Analytics backend.

This module provides REST API endpoints for accessing analytics data,
model predictions, and cost optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.db_manager import DatabaseManager
from models.performance_predictor import PerformancePredictor
from models.cost_optimizer import CostOptimizer
from models.trend_forecaster import TrendForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Arena Analytics API",
    description="API for LLM performance analytics, cost optimization, and ML predictions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
db_manager: Optional[DatabaseManager] = None
performance_predictor: Optional[PerformancePredictor] = None
cost_optimizer: Optional[CostOptimizer] = None
trend_forecaster: Optional[TrendForecaster] = None


# Pydantic models for request/response
class BestModelRequest(BaseModel):
    """Request schema for best model prediction."""
    task_type: str = Field(..., description="Task type: coding, creative, reasoning, or general")
    prompt_length: int = Field(..., ge=1, le=100000, description="Estimated prompt length in tokens")
    complexity: int = Field(..., ge=1, le=10, description="Complexity score from 1-10")
    max_cost_per_1m: Optional[float] = Field(None, ge=0, description="Maximum cost per 1M tokens (optional)")


class ModelAlternative(BaseModel):
    """Alternative model recommendation."""
    model: str
    confidence: float
    score: float


class BestModelResponse(BaseModel):
    """Response schema for best model prediction."""
    recommended_model: str
    confidence: float
    predicted_score: float
    alternatives: List[ModelAlternative]


class ModelHistoryResponse(BaseModel):
    """Response schema for model history."""
    model_name: str
    history: List[Dict[str, Any]]


class LeaderboardEntry(BaseModel):
    """Leaderboard entry schema."""
    rank: int
    model: str
    provider: str
    score: float
    win_rate: Optional[float] = None
    total_battles: Optional[int] = None


def load_ml_model() -> Optional[PerformancePredictor]:
    """Load the trained ML model from disk."""
    model_path = backend_path.parent / "data" / "models" / "performance_predictor.pkl"
    
    if not model_path.exists():
        logger.warning(f"ML model not found at {model_path}. Predictions will use heuristics.")
        return None
    
    try:
        predictor = PerformancePredictor(db_manager=db_manager)
        predictor.load_model(str(model_path))
        logger.info("ML model loaded successfully")
        return predictor
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        return None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize components on startup."""
    global db_manager, performance_predictor, cost_optimizer, trend_forecaster
    
    logger.info("Starting up LLM Arena Analytics API...")
    
    # Initialize database
    try:
        db_manager = DatabaseManager()
        db_manager.connect()
        logger.info("Database connected")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        db_manager = None
    
    # Initialize cost optimizer
    if db_manager:
        cost_optimizer = CostOptimizer(db_manager=db_manager)
        logger.info("Cost optimizer initialized")
    
    # Initialize trend forecaster
    if db_manager:
        trend_forecaster = TrendForecaster(db_manager=db_manager)
        logger.info("Trend forecaster initialized")
    
    # Load ML model
    if db_manager:
        performance_predictor = load_ml_model()
        if not performance_predictor:
            # Fallback: create untrained predictor
            performance_predictor = PerformancePredictor(db_manager=db_manager)
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    global db_manager
    if db_manager:
        db_manager.close()
        logger.info("Database connections closed")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "LLM Arena Analytics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db_manager else "disconnected",
        "ml_model": "loaded" if (performance_predictor and performance_predictor.is_trained) else "not_loaded"
    }
    
    return health_status


@app.post("/predict/best-model", response_model=BestModelResponse)
async def predict_best_model(request: BestModelRequest) -> BestModelResponse:
    """
    Predict the best model for a given task.

    Args:
        request: Task characteristics

    Returns:
        Recommended model with confidence and alternatives
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        logger.info(f"Predicting best model for task: {request.task_type}, complexity: {request.complexity}")
        
        # Load historical data for feature engineering
        historical_df = performance_predictor.load_historical_data(days=90)
        
        if historical_df.empty:
            raise HTTPException(status_code=404, detail="No historical data available")
        
        # Engineer features for all models
        features_df = performance_predictor.engineer_features(
            historical_df,
            task_prompt_length=request.prompt_length,
            task_complexity=request.complexity,
            task_domain=request.task_type
        )
        
        if features_df.empty:
            raise HTTPException(status_code=404, detail="No models available for prediction")
        
        # Calculate cost per 1M tokens for filtering
        # total_price = input_price + output_price (both per 1K tokens)
        # For 1M tokens with 50/50 split: 500K input + 500K output
        # Cost = 500 * input_price + 500 * output_price = 500 * total_price
        features_df['cost_per_1m_tokens'] = features_df['total_price'] * 500
        
        # Filter by cost if specified
        if request.max_cost_per_1m is not None:
            features_df = features_df[features_df['cost_per_1m_tokens'] <= request.max_cost_per_1m]
        
        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No models found within cost constraint of ${request.max_cost_per_1m}/1M tokens"
            )
        
        # Prepare features for prediction
        feature_cols = [
            'current_score', 'trend_7d', 'trend_30d', 'volatility',
            'win_rate', 'total_battles', 'cost_tier', 'total_price',
            'provider_encoded', 'task_prompt_length', 'task_complexity', 'task_domain'
        ]
        
        X = features_df[feature_cols].copy()
        
        # Predict scores for all models
        if performance_predictor.is_trained:
            try:
                predictions = performance_predictor.predict(X)
                features_df['predicted_score'] = predictions
            except Exception as e:
                logger.warning(f"ML prediction failed, using current scores: {e}")
                features_df['predicted_score'] = features_df['current_score']
        else:
            # Fallback: use current scores
            features_df['predicted_score'] = features_df['current_score']
        
        # Calculate confidence based on model performance and data quality
        # Higher confidence for models with more battles and stable scores
        features_df['confidence'] = (
            features_df['win_rate'] * 0.4 +
            (features_df['total_battles'] / features_df['total_battles'].max()) * 0.3 +
            (1 - features_df['volatility'] / features_df['volatility'].max()) * 0.3
        ).clip(0, 1)
        
        # Sort by predicted score and confidence
        features_df = features_df.sort_values(
            ['predicted_score', 'confidence'],
            ascending=[False, False]
        )
        
        # Get top recommendation
        top_model = features_df.iloc[0]
        recommended_model = top_model['model_name']
        predicted_score = float(top_model['predicted_score'])
        confidence = float(top_model['confidence'])
        
        # Get alternatives (top 3 excluding the recommended one)
        alternatives_df = features_df.iloc[1:4]
        alternatives = [
            ModelAlternative(
                model=row['model_name'],
                confidence=float(row['confidence']),
                score=float(row['predicted_score'])
            )
            for _, row in alternatives_df.iterrows()
        ]
        
        logger.info(f"Recommended model: {recommended_model} (score: {predicted_score:.1f}, confidence: {confidence:.2f})")
        
        return BestModelResponse(
            recommended_model=recommended_model,
            confidence=confidence,
            predicted_score=predicted_score,
            alternatives=alternatives
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting best model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    limit: int = Query(20, ge=1, le=100, description="Number of models to return")
) -> List[LeaderboardEntry]:
    """
    Get current leaderboard of top models.

    Args:
        limit: Maximum number of models to return

    Returns:
        List of leaderboard entries
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = """
            SELECT DISTINCT ON (m.id)
                ar.rank_position as rank,
                m.name as model,
                COALESCE(m.provider, 'Unknown') as provider,
                ar.elo_rating as score,
                ar.win_rate,
                ar.total_battles
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            ORDER BY m.id, ar.recorded_at DESC
            LIMIT %s
        """
        
        results = db_manager.execute_query(query, (limit,))
        
        if not results:
            # Fallback: get any recent rankings
            query = """
                SELECT 
                    ar.rank_position as rank,
                    m.name as model,
                    COALESCE(m.provider, 'Unknown') as provider,
                    ar.elo_rating as score,
                    ar.win_rate,
                    ar.total_battles
                FROM arena_rankings ar
                JOIN models m ON ar.model_id = m.id
                ORDER BY ar.rank_position ASC
                LIMIT %s
            """
            results = db_manager.execute_query(query, (limit,))
        
        # Sort by rank
        results = sorted(results, key=lambda x: x.get('rank', 999))
        
        leaderboard = [
            LeaderboardEntry(
                rank=entry.get('rank', 0),
                model=entry.get('model', 'Unknown'),
                provider=entry.get('provider', 'Unknown'),
                score=float(entry.get('score', 0)),
                win_rate=float(entry.get('win_rate', 0)) if entry.get('win_rate') else None,
                total_battles=int(entry.get('total_battles', 0)) if entry.get('total_battles') else None
            )
            for entry in results
        ]
        
        return leaderboard
        
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/history", response_model=ModelHistoryResponse)
async def get_model_history(
    model_name: str = PathParam(..., description="Name of the model"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back")
) -> ModelHistoryResponse:
    """
    Get score history for a specific model.

    Args:
        model_name: Name of the model
        days: Number of days to look back

    Returns:
        Model history with scores over time
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get model
        model = db_manager.get_model_by_name(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Get history
        history = db_manager.get_arena_history(model['id'], days=days)
        
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No history found for model '{model_name}' in the last {days} days"
            )
        
        # Format history
        history_list = [
            {
                "date": str(record.get('recorded_at', '')),
                "score": float(record.get('elo_rating', 0)),
                "rank": int(record.get('rank_position', 0)),
                "win_rate": float(record.get('win_rate', 0)) if record.get('win_rate') else None,
                "total_battles": int(record.get('total_battles', 0)) if record.get('total_battles') else None
            }
            for record in history
        ]
        
        return ModelHistoryResponse(
            model_name=model_name,
            history=history_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_models(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    limit: int = Query(100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """
    Get list of models.

    Args:
        provider: Optional provider filter
        limit: Maximum number of results

    Returns:
        List of model dictionaries
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        models = db_manager.get_models()
        
        if provider:
            models = [m for m in models if m.get('provider') == provider]
        
        return models[:limit]
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/rank")
async def get_rank_forecast(
    days_ahead: int = Query(30, ge=1, le=90, description="Days to forecast ahead")
) -> List[Dict[str, Any]]:
    """
    Get forecasted rankings for all models.

    Args:
        days_ahead: Number of days to forecast ahead

    Returns:
        List of forecasted rankings
    """
    if not db_manager or not trend_forecaster:
        raise HTTPException(status_code=503, detail="Forecaster not available")
    
    try:
        forecast_df = trend_forecaster.rank_forecast(days_ahead=days_ahead)
        
        if forecast_df.empty:
            return []
        
        return forecast_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error forecasting rankings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/model/{model_name}")
async def get_model_forecast(
    model_name: str = PathParam(..., description="Name of the model"),
    days_ahead: int = Query(30, ge=1, le=90, description="Days to forecast ahead")
) -> Dict[str, Any]:
    """
    Get forecast for a specific model.

    Args:
        model_name: Name of the model
        days_ahead: Number of days to forecast ahead

    Returns:
        Forecast data with predictions and confidence intervals
    """
    if not db_manager or not trend_forecaster:
        raise HTTPException(status_code=503, detail="Forecaster not available")
    
    try:
        forecast_df = trend_forecaster.forecast_score(model_name, days_ahead=days_ahead)
        trend = trend_forecaster.detect_trend(model_name, window=30)
        
        if forecast_df.empty:
            raise HTTPException(status_code=404, detail=f"No forecast available for {model_name}")
        
        return {
            'model_name': model_name,
            'forecast': forecast_df.to_dict('records'),
            'trend': trend,
            'days_ahead': days_ahead
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forecasting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/anomalies/{model_name}")
async def get_model_anomalies(
    model_name: str = PathParam(..., description="Name of the model")
) -> List[Dict[str, Any]]:
    """
    Get detected anomalies for a model.

    Args:
        model_name: Name of the model

    Returns:
        List of detected anomalies
    """
    if not db_manager or not trend_forecaster:
        raise HTTPException(status_code=503, detail="Forecaster not available")
    
    try:
        anomalies = trend_forecaster.anomaly_detection(model_name)
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
