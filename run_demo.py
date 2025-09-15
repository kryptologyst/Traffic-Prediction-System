#!/usr/bin/env python3
"""
Traffic Prediction System - Demo Script
Quick demonstration of the traffic prediction capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from traffic_predictor import TrafficPredictor
from data_generator import TrafficDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("🚗 Traffic Prediction System Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing data generator...")
    generator = TrafficDataGenerator()
    
    # Generate sample data if needed
    if not os.path.exists('data/traffic_data.db'):
        print("2. Generating sample traffic data...")
        generator.populate_database(days_back=30)
    else:
        print("2. Using existing traffic data...")
    
    # Get available locations
    locations = generator.get_all_locations()
    print(f"\n3. Available locations: {len(locations)}")
    for loc in locations:
        print(f"   - {loc['location_name']} ({loc['location_id']})")
    
    # Demo with first location
    location_id = locations[0]['location_id']
    location_name = locations[0]['location_name']
    
    print(f"\n4. Demo with: {location_name}")
    
    # Load training data
    print("5. Loading training data...")
    df = generator.get_location_data(location_id, days_back=21)
    print(f"   - Loaded {len(df)} data points")
    print(f"   - Date range: {df.index.min()} to {df.index.max()}")
    
    # Train model
    print("6. Training Random Forest model...")
    predictor = TrafficPredictor(model_type='random_forest')
    metrics = predictor.train(df)
    
    print(f"   - Training R² Score: {metrics['train_r2']:.3f}")
    print(f"   - Test R² Score: {metrics['test_r2']:.3f}")
    print(f"   - Test MAE: {metrics['test_mae']:.1f} vehicles")
    
    # Make predictions
    print("7. Making 24-hour predictions...")
    predictions = predictor.predict_next_hours(df, hours_ahead=24)
    
    print(f"   - Generated {len(predictions)} hourly predictions")
    print(f"   - Average predicted traffic: {predictions['predicted_traffic'].mean():.0f} vehicles/hour")
    print(f"   - Peak predicted traffic: {predictions['predicted_traffic'].max():.0f} vehicles/hour")
    
    # Show sample predictions
    print("\n8. Sample predictions:")
    for i in range(min(5, len(predictions))):
        timestamp = predictions.index[i]
        traffic = predictions.iloc[i]['predicted_traffic']
        print(f"   - {timestamp.strftime('%Y-%m-%d %H:%M')}: {traffic:.0f} vehicles")
    
    # Create visualization
    print("\n9. Creating visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot recent historical data
    recent_data = df.tail(48)  # Last 48 hours
    plt.plot(recent_data.index, recent_data['traffic_volume'], 
             label='Historical Traffic', color='blue', linewidth=2)
    
    # Plot predictions
    plt.plot(predictions.index, predictions['predicted_traffic'], 
             label='Predicted Traffic', color='red', linewidth=2, linestyle='--')
    
    plt.title(f'Traffic Prediction for {location_name}')
    plt.xlabel('Time')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('traffic_prediction_demo.png', dpi=300, bbox_inches='tight')
    print("   - Saved visualization as 'traffic_prediction_demo.png'")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the web application:")
    print("   python app.py")
    print("\nThen visit: http://localhost:5000")

if __name__ == "__main__":
    main()
