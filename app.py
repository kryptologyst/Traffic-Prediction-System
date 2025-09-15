"""
Traffic Prediction System - Flask Web Application
A web interface for the traffic prediction system with REST API endpoints.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from traffic_predictor import TrafficPredictor
from data_generator import TrafficDataGenerator

app = Flask(__name__)
CORS(app)

# Initialize components
data_generator = TrafficDataGenerator()
predictor = TrafficPredictor(model_type='random_forest')

# Global variables for caching
trained_models = {}
locations_cache = None

def ensure_data_exists():
    """Ensure that the database exists and has data."""
    if not os.path.exists('data/traffic_data.db'):
        os.makedirs('data', exist_ok=True)
        data_generator.populate_database(days_back=90)

def get_or_train_model(location_id: str):
    """Get trained model for location or train if not exists."""
    if location_id not in trained_models:
        # Get training data
        df = data_generator.get_location_data(location_id, days_back=60)
        
        if df.empty:
            raise ValueError(f"No data available for location: {location_id}")
        
        # Train model
        model = TrafficPredictor(model_type='random_forest')
        metrics = model.train(df)
        
        trained_models[location_id] = {
            'model': model,
            'metrics': metrics,
            'last_trained': datetime.now()
        }
    
    return trained_models[location_id]

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/locations')
def get_locations():
    """Get all available locations."""
    global locations_cache
    
    if locations_cache is None:
        ensure_data_exists()
        locations_cache = data_generator.get_all_locations()
    
    return jsonify(locations_cache)

@app.route('/api/historical/<location_id>')
def get_historical_data(location_id):
    """Get historical traffic data for a location."""
    days = request.args.get('days', 7, type=int)
    
    try:
        df = data_generator.get_location_data(location_id, days_back=days)
        
        if df.empty:
            return jsonify({'error': 'No data found for location'}), 404
        
        # Convert to JSON format
        data = []
        for timestamp, row in df.iterrows():
            data.append({
                'timestamp': timestamp.isoformat(),
                'traffic_volume': int(row['traffic_volume']),
                'weather_condition': row['weather_condition'],
                'is_weekend': bool(row['is_weekend'])
            })
        
        return jsonify({
            'location_id': location_id,
            'data': data,
            'total_records': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<location_id>')
def predict_traffic(location_id):
    """Predict future traffic for a location."""
    hours_ahead = request.args.get('hours', 24, type=int)
    
    try:
        # Get or train model
        model_info = get_or_train_model(location_id)
        model = model_info['model']
        
        # Get recent data for prediction
        df = data_generator.get_location_data(location_id, days_back=7)
        
        if df.empty:
            return jsonify({'error': 'No data available for prediction'}), 404
        
        # Make predictions
        predictions_df = model.predict_next_hours(df, hours_ahead=hours_ahead)
        
        # Convert to JSON format
        predictions = []
        for timestamp, row in predictions_df.iterrows():
            predictions.append({
                'timestamp': timestamp.isoformat(),
                'predicted_traffic': round(row['predicted_traffic'])
            })
        
        return jsonify({
            'location_id': location_id,
            'predictions': predictions,
            'model_metrics': model_info['metrics'],
            'hours_predicted': len(predictions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/<location_id>', methods=['POST'])
def train_model(location_id):
    """Train or retrain model for a location."""
    try:
        # Get training data
        df = data_generator.get_location_data(location_id, days_back=60)
        
        if df.empty:
            return jsonify({'error': 'No data available for training'}), 404
        
        # Train new model
        model = TrafficPredictor(model_type='random_forest')
        metrics = model.train(df)
        
        # Cache the trained model
        trained_models[location_id] = {
            'model': model,
            'metrics': metrics,
            'last_trained': datetime.now()
        }
        
        return jsonify({
            'message': 'Model trained successfully',
            'location_id': location_id,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/<location_id>')
def get_analytics(location_id):
    """Get traffic analytics for a location."""
    days = request.args.get('days', 30, type=int)
    
    try:
        df = data_generator.get_location_data(location_id, days_back=days)
        
        if df.empty:
            return jsonify({'error': 'No data found for location'}), 404
        
        # Calculate analytics
        analytics = {
            'average_daily_traffic': df.groupby(df.index.date)['traffic_volume'].sum().mean(),
            'peak_hour': df.groupby('hour')['traffic_volume'].mean().idxmax(),
            'busiest_day': df.groupby('day_of_week')['traffic_volume'].mean().idxmax(),
            'weekend_vs_weekday': {
                'weekend_avg': df[df['is_weekend'] == True]['traffic_volume'].mean(),
                'weekday_avg': df[df['is_weekend'] == False]['traffic_volume'].mean()
            },
            'weather_impact': df.groupby('weather_condition')['traffic_volume'].mean().to_dict(),
            'hourly_pattern': df.groupby('hour')['traffic_volume'].mean().to_dict(),
            'daily_pattern': df.groupby('day_of_week')['traffic_volume'].mean().to_dict()
        }
        
        return jsonify({
            'location_id': location_id,
            'analytics': analytics,
            'data_period_days': days
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_system_status():
    """Get system status and health check."""
    try:
        ensure_data_exists()
        
        locations = data_generator.get_all_locations()
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'locations_available': len(locations),
            'trained_models': len(trained_models),
            'database_path': data_generator.db_path,
            'database_exists': os.path.exists(data_generator.db_path)
        }
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Ensure data exists on startup
    ensure_data_exists()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
