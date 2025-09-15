# Traffic Prediction System

A comprehensive machine learning system for predicting traffic volume based on historical data, featuring a modern web interface and REST API.

## Features

- **Advanced ML Models**: Linear Regression and Random Forest algorithms for traffic prediction
- **Real-time Predictions**: Predict traffic for the next 1-168 hours
- **Interactive Dashboard**: Modern web UI with charts and analytics
- **REST API**: Complete API for integration with other systems
- **Mock Database**: Realistic traffic data generator with weather and event effects
- **Multiple Locations**: Support for different traffic monitoring locations
- **Analytics**: Comprehensive traffic pattern analysis

## Project Structure

```
traffic-prediction-system/
├── src/
│   ├── traffic_predictor.py    # Core ML prediction engine
│   └── data_generator.py       # Mock database and data generation
├── templates/
│   └── index.html             # Web dashboard interface
├── static/
│   └── js/
│       └── app.js             # Frontend JavaScript
├── data/                      # SQLite database (auto-generated)
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd traffic-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the web server
python app.py
```

The application will be available at `http://localhost:5000`

### 3. Generate Sample Data

The application automatically generates sample traffic data on first run. The mock database includes:

- **5 Different Locations**: Downtown, Highway, Residential, Shopping Center, University
- **90 Days of Data**: Realistic traffic patterns with hourly granularity
- **Weather Effects**: Traffic variations based on weather conditions
- **Special Events**: Holiday and random event impacts
- **Rush Hour Patterns**: Morning and evening traffic peaks

## Usage

### Web Dashboard

1. **Select Location**: Choose from available traffic monitoring locations
2. **Load Data**: View historical traffic patterns and statistics
3. **Make Predictions**: Generate traffic forecasts for upcoming hours
4. **Train Models**: Retrain ML models with latest data
5. **View Analytics**: Analyze traffic patterns by hour, day, and weather

### API Endpoints

#### Get Locations
```http
GET /api/locations
```

#### Historical Data
```http
GET /api/historical/{location_id}?days=7
```

#### Traffic Predictions
```http
GET /api/predict/{location_id}?hours=24
```

#### Train Model
```http
POST /api/train/{location_id}
```

#### Analytics
```http
GET /api/analytics/{location_id}?days=30
```

#### System Status
```http
GET /api/status
```

## Machine Learning Models

### TrafficPredictor Class

The core prediction engine supports multiple algorithms:

- **Linear Regression**: Fast, interpretable baseline model
- **Random Forest**: Advanced ensemble method with better accuracy

### Features Used

- **Lag Features**: Previous 1-6 hours of traffic data
- **Time Features**: Hour of day, day of week, month
- **Calendar Features**: Weekend/weekday indicators
- **Rolling Statistics**: Moving averages and standard deviations

### Model Performance

Models are evaluated using:
- **Mean Squared Error (MSE)**
- **R² Score**: Coefficient of determination
- **Mean Absolute Error (MAE)**

## Data Generation

The `TrafficDataGenerator` creates realistic traffic data with:

### Traffic Patterns
- **Rush Hours**: Morning (7-9 AM) and evening (5-7 PM) peaks
- **Weekend Patterns**: Different traffic flows on weekends
- **Location Types**: Each location has unique characteristics

### External Factors
- **Weather Impact**: Rain, snow, fog affect traffic volume
- **Special Events**: Holidays and random events modify patterns
- **Seasonal Variations**: Monthly traffic variations

### Database Schema

```sql
-- Traffic data table
CREATE TABLE traffic_data (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    location_id TEXT,
    location_name TEXT,
    traffic_volume INTEGER,
    weather_condition TEXT,
    is_weekend BOOLEAN,
    hour INTEGER,
    day_of_week INTEGER,
    month INTEGER
);

-- Locations table
CREATE TABLE locations (
    location_id TEXT PRIMARY KEY,
    location_name TEXT,
    base_traffic INTEGER,
    peak_multiplier REAL,
    weekend_factor REAL,
    latitude REAL,
    longitude REAL
);
```

## Development

### Adding New Locations

1. Update `locations` dictionary in `data_generator.py`
2. Regenerate database: `python src/data_generator.py`

### Custom Models

Extend the `TrafficPredictor` class:

```python
from src.traffic_predictor import TrafficPredictor

class CustomPredictor(TrafficPredictor):
    def __init__(self):
        # Initialize with custom model
        super().__init__(model_type='custom')
        self.model = YourCustomModel()
```

### API Extensions

Add new endpoints in `app.py`:

```python
@app.route('/api/custom/<location_id>')
def custom_endpoint(location_id):
    # Your custom logic
    return jsonify(result)
```

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

1. **Docker** (recommended):
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

2. **Cloud Platforms**: Deploy to AWS, GCP, or Azure
3. **Environment Variables**: Configure database paths and API keys

## Performance Optimization

- **Model Caching**: Trained models are cached in memory
- **Database Indexing**: Optimize queries with proper indexes
- **Async Processing**: Use Celery for long-running tasks
- **CDN**: Serve static assets via CDN

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test API endpoints
curl http://localhost:5000/api/status
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Scikit-learn**: Machine learning algorithms
- **Flask**: Web framework
- **Chart.js**: Interactive charts
- **Font Awesome**: Icons

## Support

For questions or issues:
1. Check the documentation
2. Search existing issues
3. Create new issue with detailed description
 
# Traffic-Prediction-System
