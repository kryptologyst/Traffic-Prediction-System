"""
Traffic Data Generator - Mock Database Module
Generates realistic traffic data for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TrafficDataGenerator:
    """
    Generates realistic traffic data based on various patterns and factors.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the traffic data generator.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'traffic_data.db')
        
        self.db_path = db_path
        self.locations = {
            'downtown_main': {
                'name': 'Downtown Main Street',
                'base_traffic': 300,
                'peak_multiplier': 2.5,
                'weekend_factor': 0.7
            },
            'highway_101': {
                'name': 'Highway 101 North',
                'base_traffic': 800,
                'peak_multiplier': 1.8,
                'weekend_factor': 0.9
            },
            'residential_oak': {
                'name': 'Oak Avenue Residential',
                'base_traffic': 150,
                'peak_multiplier': 1.5,
                'weekend_factor': 1.2
            },
            'shopping_center': {
                'name': 'Westfield Shopping Center',
                'base_traffic': 400,
                'peak_multiplier': 2.0,
                'weekend_factor': 1.8
            },
            'university_campus': {
                'name': 'University Campus Drive',
                'base_traffic': 250,
                'peak_multiplier': 3.0,
                'weekend_factor': 0.3
            }
        }
    
    def generate_hourly_pattern(self, hour: int, is_weekend: bool = False) -> float:
        """
        Generate traffic multiplier based on hour of day.
        
        Args:
            hour (int): Hour of day (0-23)
            is_weekend (bool): Whether it's weekend
            
        Returns:
            float: Traffic multiplier
        """
        if is_weekend:
            # Weekend pattern - later start, more evening activity
            if 6 <= hour <= 10:
                return 0.8 + 0.4 * np.sin((hour - 6) * np.pi / 4)
            elif 11 <= hour <= 14:
                return 1.2 + 0.3 * np.sin((hour - 11) * np.pi / 3)
            elif 18 <= hour <= 22:
                return 1.0 + 0.5 * np.sin((hour - 18) * np.pi / 4)
            else:
                return 0.3 + 0.2 * np.random.random()
        else:
            # Weekday pattern - morning and evening rush hours
            if 7 <= hour <= 9:  # Morning rush
                return 1.5 + 0.8 * np.sin((hour - 7) * np.pi / 2)
            elif 17 <= hour <= 19:  # Evening rush
                return 1.3 + 0.9 * np.sin((hour - 17) * np.pi / 2)
            elif 10 <= hour <= 16:  # Daytime
                return 0.8 + 0.3 * np.sin((hour - 10) * np.pi / 6)
            else:  # Night/early morning
                return 0.2 + 0.3 * np.exp(-(hour if hour <= 12 else 24 - hour) / 3)
    
    def add_weather_effect(self, base_traffic: float, weather_condition: str) -> float:
        """
        Modify traffic based on weather conditions.
        
        Args:
            base_traffic (float): Base traffic volume
            weather_condition (str): Weather condition
            
        Returns:
            float: Modified traffic volume
        """
        weather_effects = {
            'clear': 1.0,
            'cloudy': 0.95,
            'light_rain': 0.85,
            'heavy_rain': 0.65,
            'snow': 0.45,
            'fog': 0.75,
            'storm': 0.35
        }
        
        return base_traffic * weather_effects.get(weather_condition, 1.0)
    
    def add_special_events(self, base_traffic: float, date: datetime) -> float:
        """
        Add traffic variations for special events and holidays.
        
        Args:
            base_traffic (float): Base traffic volume
            date (datetime): Date to check for events
            
        Returns:
            float: Modified traffic volume
        """
        # Holiday effects
        holidays = {
            (1, 1): 0.3,    # New Year's Day
            (7, 4): 0.6,    # Independence Day
            (12, 25): 0.2,  # Christmas
            (11, 24): 0.4,  # Thanksgiving (approximate)
        }
        
        month_day = (date.month, date.day)
        if month_day in holidays:
            return base_traffic * holidays[month_day]
        
        # Random special events (5% chance)
        if np.random.random() < 0.05:
            event_multiplier = np.random.choice([0.5, 1.5, 2.0], p=[0.3, 0.5, 0.2])
            return base_traffic * event_multiplier
        
        return base_traffic
    
    def generate_traffic_data(self, 
                            location_id: str,
                            start_date: datetime,
                            end_date: datetime,
                            noise_level: float = 0.15) -> pd.DataFrame:
        """
        Generate realistic traffic data for a specific location and time period.
        
        Args:
            location_id (str): Location identifier
            start_date (datetime): Start date for data generation
            end_date (datetime): End date for data generation
            noise_level (float): Amount of random noise to add
            
        Returns:
            pd.DataFrame: Generated traffic data
        """
        if location_id not in self.locations:
            raise ValueError(f"Unknown location: {location_id}")
        
        location_config = self.locations[location_id]
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        traffic_data = []
        weather_conditions = ['clear', 'cloudy', 'light_rain', 'heavy_rain', 'fog']
        weather_probs = [0.4, 0.3, 0.15, 0.08, 0.07]
        
        for timestamp in timestamps:
            # Base traffic for location
            base_traffic = location_config['base_traffic']
            
            # Apply hourly pattern
            is_weekend = timestamp.weekday() >= 5
            hourly_multiplier = self.generate_hourly_pattern(timestamp.hour, is_weekend)
            
            # Apply weekend factor
            if is_weekend:
                base_traffic *= location_config['weekend_factor']
            
            # Apply peak multiplier
            traffic_volume = base_traffic * hourly_multiplier * location_config['peak_multiplier']
            
            # Add weather effects
            weather = np.random.choice(weather_conditions, p=weather_probs)
            traffic_volume = self.add_weather_effect(traffic_volume, weather)
            
            # Add special events
            traffic_volume = self.add_special_events(traffic_volume, timestamp)
            
            # Add random noise
            noise = np.random.normal(0, noise_level * traffic_volume)
            traffic_volume = max(0, traffic_volume + noise)
            
            traffic_data.append({
                'timestamp': timestamp,
                'location_id': location_id,
                'location_name': location_config['name'],
                'traffic_volume': round(traffic_volume),
                'weather_condition': weather,
                'is_weekend': is_weekend,
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'month': timestamp.month
            })
        
        return pd.DataFrame(traffic_data)
    
    def create_database(self):
        """Create SQLite database and tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create traffic_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                location_name TEXT,
                traffic_volume INTEGER,
                weather_condition TEXT,
                is_weekend BOOLEAN,
                hour INTEGER,
                day_of_week INTEGER,
                month INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                location_id TEXT PRIMARY KEY,
                location_name TEXT,
                base_traffic INTEGER,
                peak_multiplier REAL,
                weekend_factor REAL,
                latitude REAL,
                longitude REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database created at {self.db_path}")
    
    def populate_database(self, days_back: int = 90):
        """
        Populate database with generated traffic data.
        
        Args:
            days_back (int): Number of days of historical data to generate
        """
        self.create_database()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        conn = sqlite3.connect(self.db_path)
        
        # Insert location data
        for location_id, config in self.locations.items():
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO locations 
                (location_id, location_name, base_traffic, peak_multiplier, weekend_factor, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                location_id,
                config['name'],
                config['base_traffic'],
                config['peak_multiplier'],
                config['weekend_factor'],
                np.random.uniform(37.7, 37.8),  # Mock coordinates (SF area)
                np.random.uniform(-122.5, -122.4)
            ))
        
        # Generate and insert traffic data for each location
        for location_id in self.locations.keys():
            logger.info(f"Generating data for {location_id}...")
            
            df = self.generate_traffic_data(location_id, start_date, end_date)
            df.to_sql('traffic_data', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database populated with {days_back} days of data for {len(self.locations)} locations")
    
    def get_location_data(self, location_id: str, days_back: int = 30) -> pd.DataFrame:
        """
        Retrieve traffic data for a specific location.
        
        Args:
            location_id (str): Location identifier
            days_back (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Traffic data for the location
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, traffic_volume, weather_condition, is_weekend, hour, day_of_week, month
            FROM traffic_data 
            WHERE location_id = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days_back)
        
        df = pd.read_sql_query(query, conn, params=(location_id,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_all_locations(self) -> List[Dict]:
        """
        Get all available locations.
        
        Returns:
            List[Dict]: List of location information
        """
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM locations'
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df.to_dict('records')


def main():
    """Generate sample traffic data."""
    import os
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Initialize generator
    generator = TrafficDataGenerator()
    
    # Populate database with 90 days of data
    generator.populate_database(days_back=90)
    
    print("Sample traffic data generated successfully!")
    
    # Show sample data
    sample_data = generator.get_location_data('downtown_main', days_back=7)
    print(f"\nSample data for Downtown Main Street (last 7 days):")
    print(sample_data.head(10))


if __name__ == "__main__":
    main()
