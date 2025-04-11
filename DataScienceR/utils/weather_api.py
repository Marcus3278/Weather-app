import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
import numpy as np
from utils.db_utils import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherAPI:
    def __init__(self):
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2"
        self.station_id = "GHCND:USC00116446"  # Olney 2S, IL US station ID
        
        # Initialize database connection
        try:
            self.db = DatabaseManager()
            # Test database connection
            if not self.db.test_connection():
                logger.warning("Database connection failed, falling back to synthetic data only")
                self.use_db = False
            else:
                self.use_db = True
                logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.use_db = False

    def get_historical_weather(self, start_date=None, end_date=None, use_db=True):
        """Fetch historical weather data from database or generate synthetic data"""
        if not start_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
        # Try to retrieve data from database first if enabled
        if self.use_db and use_db:
            try:
                db_data = self.db.get_weather_data(start_date, end_date)
                if not db_data.empty:
                    logger.info(f"Retrieved {len(db_data)} records from database")
                    return db_data
                logger.info("No data found in database, generating synthetic data")
            except Exception as e:
                logger.error(f"Error retrieving data from database: {str(e)}")
                
        # Generate synthetic data if database retrieval fails or is disabled
        logger.info("Generating synthetic weather data")
        return self.generate_synthetic_weather(start_date, end_date)
    
    def generate_synthetic_weather(self, start_date, end_date):
        """Generate synthetic weather data for demonstration"""
        # Create sample data for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='h')

        # Generate synthetic weather data
        np.random.seed(42)
        temp_base = 75  # Base temperature
        temp_variation = 15  # Temperature variation

        # Temperature calculations (as numpy arrays for mutability)
        temperature = temp_base + temp_variation * np.sin(np.pi * dates.hour.values / 12) + np.random.normal(0, 3, len(dates))
        
        # Humidity calculation
        humidity = np.random.uniform(40, 90, len(dates))
        
        # Calculate "feels like" temperature based on temperature and humidity
        # Simple heat index approximation - all using numpy arrays for mutability
        feels_like = temperature.copy()
        
        # Hot and humid conditions mask - feels hotter than it is
        hot_humid_mask = (temperature > 70) & (humidity > 60)
        feels_like[hot_humid_mask] += 0.1 * (humidity[hot_humid_mask] - 60)
        
        # Cold and dry conditions mask - feels colder than it is
        cold_dry_mask = (temperature < 60) & (humidity < 40)
        feels_like[cold_dry_mask] -= 0.1 * (40 - humidity[cold_dry_mask])

        # Now create the dataframe with all processed data
        weather_data = pd.DataFrame({
            'timestamp': dates,
            'temperature': np.round(temperature, 1),
            'feels_like': np.round(feels_like, 1),
            'humidity': np.round(humidity, 1),
            'wind_speed': np.round(np.abs(np.random.normal(8, 4, len(dates))), 1),
            'wind_deg': np.round(np.random.uniform(0, 360, len(dates)), 1),
            'precipitation': np.round(np.abs(np.random.exponential(0.1, len(dates))), 2),
            'pressure': np.round(np.random.uniform(1008, 1020, len(dates)), 1)
        })
        
        # Save generated data to database if connection is available
        if self.use_db:
            try:
                self.db.save_weather_data(weather_data)
                logger.info(f"Saved {len(weather_data)} records to database")
            except Exception as e:
                logger.error(f"Error saving data to database: {str(e)}")

        return weather_data

    def get_current_weather(self):
        """Get current weather based on most recent historical data"""
        current = self.get_historical_weather(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now()
        ).iloc[-1]

        return {
            'temp': current['temperature'],
            'temp_change': round(np.random.uniform(-2, 2), 1),
            'humidity': current['humidity'],
            'wind_speed': current['wind_speed'],
            'pressure': current['pressure']
        }
        
    def save_weather_data(self, data):
        """Save weather data to database"""
        if not self.use_db:
            logger.warning("Database connection not available, cannot save data")
            return False
            
        try:
            success = self.db.save_weather_data(data)
            return success
        except Exception as e:
            logger.error(f"Error saving weather data: {str(e)}")
            return False