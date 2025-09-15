"""
Traffic Prediction System - Core Module
A machine learning system for predicting traffic volume based on historical data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficPredictor:
    """
    A comprehensive traffic prediction system using multiple ML algorithms.
    """
    
    def __init__(self, model_type='linear'):
        """
        Initialize the traffic predictor.
        
        Args:
            model_type (str): Type of model to use ('linear', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'linear' or 'random_forest'")
    
    def create_features(self, df: pd.DataFrame, lag_hours: int = 6) -> pd.DataFrame:
        """
        Create lag features and time-based features for traffic prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with traffic data
            lag_hours (int): Number of lag hours to create
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Create lag features
        for i in range(1, lag_hours + 1):
            df_features[f'lag_{i}'] = df_features['traffic_volume'].shift(i)
        
        # Create time-based features
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
        
        # Create rolling statistics
        df_features['rolling_mean_3h'] = df_features['traffic_volume'].rolling(window=3).mean()
        df_features['rolling_std_3h'] = df_features['traffic_volume'].rolling(window=3).std()
        df_features['rolling_mean_24h'] = df_features['traffic_volume'].rolling(window=24).mean()
        
        # Drop rows with NaN values
        df_features.dropna(inplace=True)
        
        return df_features
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training or prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        # Define feature columns (exclude target and non-numeric columns)
        exclude_cols = ['traffic_volume', 'weather_condition']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df['traffic_volume'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the traffic prediction model.
        
        Args:
            df (pd.DataFrame): Training data
            test_size (float): Proportion of data for testing
            
        Returns:
            Dict[str, float]: Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Create features
        df_features = self.create_features(df)
        
        # Prepare data
        X, y = self.prepare_data(df_features)
        
        # Split data (preserve time order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        logger.info(f"Training completed. Test R² Score: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make traffic volume predictions.
        
        Args:
            df (pd.DataFrame): Data for prediction
            
        Returns:
            np.ndarray: Predicted traffic volumes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        df_features = self.create_features(df)
        
        # Prepare data
        X, _ = self.prepare_data(df_features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_next_hours(self, df: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
        """
        Predict traffic for the next N hours.
        
        Args:
            df (pd.DataFrame): Historical data
            hours_ahead (int): Number of hours to predict
            
        Returns:
            pd.DataFrame: Predictions with timestamps
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get the last timestamp
        last_timestamp = df.index[-1]
        
        # Create future timestamps
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        predictions = []
        current_df = df.copy()
        
        for timestamp in future_timestamps:
            # Create features for current state
            df_features = self.create_features(current_df)
            
            if len(df_features) == 0:
                break
                
            # Get the last row for prediction
            last_row = df_features.iloc[[-1]]
            X, _ = self.prepare_data(last_row)
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            
            predictions.append(pred)
            
            # Add prediction to current_df for next iteration
            new_row = pd.DataFrame(
                {'traffic_volume': [pred]},
                index=[timestamp]
            )
            current_df = pd.concat([current_df, new_row])
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'timestamp': future_timestamps[:len(predictions)],
            'predicted_traffic': predictions
        })
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
