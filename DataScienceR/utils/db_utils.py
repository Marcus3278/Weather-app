import os
import json
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, Float, String, MetaData, DateTime, Boolean, JSON
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize the database connection"""
        try:
            # Get database URL from environment variable
            self.database_url = os.environ.get("DATABASE_URL")
            if not self.database_url:
                logger.error("DATABASE_URL environment variable not set")
                raise ValueError("DATABASE_URL environment variable not set")
            
            # Create engine
            self.engine = create_engine(self.database_url)
            
            # Define metadata
            self.metadata = MetaData()
            
            # Define tables
            self.weather_data = Table(
                'weather_data', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, nullable=False),
                Column('temperature', Float, nullable=False),
                Column('feels_like', Float),
                Column('humidity', Float),
                Column('wind_speed', Float),
                Column('wind_deg', Float),
                Column('precipitation', Float),
                Column('pressure', Float),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            self.saved_analyses = Table(
                'saved_analyses', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String(255), nullable=False),
                Column('description', String),
                Column('analysis_type', String(50), nullable=False),
                Column('parameters', JSON),
                Column('results', JSON),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            self.user_uploads = Table(
                'user_uploads', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('filename', String(255), nullable=False),
                Column('original_filename', String(255), nullable=False),
                Column('file_size', Integer, nullable=False),
                Column('content_type', String(100), nullable=False),
                Column('metadata', JSON),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            self.locations = Table(
                'locations', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String(255), nullable=False),
                Column('zip_code', String(10), nullable=False),
                Column('latitude', Float),
                Column('longitude', Float),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def test_connection(self):
        """Test the database connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            return False

    def save_weather_data(self, df):
        """Save weather dataframe to database"""
        try:
            # Ensure necessary columns exist
            required_columns = ['timestamp', 'temperature']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"DataFrame is missing required columns: {required_columns}")
                return False
            
            # Convert DataFrame to records
            records = df.to_dict(orient='records')
            
            # Insert records into the database
            with self.engine.connect() as connection:
                for record in records:
                    # Ensure timestamp is a datetime object
                    if isinstance(record['timestamp'], str):
                        record['timestamp'] = pd.to_datetime(record['timestamp'])
                    
                    # Insert record
                    connection.execute(
                        self.weather_data.insert().values(**record)
                    )
                connection.commit()
            
            logger.info(f"Successfully saved {len(records)} weather records to database")
            return True
        except Exception as e:
            logger.error(f"Error saving weather data: {str(e)}")
            return False

    def get_weather_data(self, start_date=None, end_date=None, limit=1000):
        """Retrieve weather data from database with optional date filtering"""
        try:
            query = self.weather_data.select()
            
            if start_date:
                query = query.where(self.weather_data.c.timestamp >= start_date)
            
            if end_date:
                query = query.where(self.weather_data.c.timestamp <= end_date)
            
            query = query.order_by(self.weather_data.c.timestamp.desc()).limit(limit)
            
            with self.engine.connect() as connection:
                result = connection.execute(query)
                
                # Convert each row to a dictionary with proper column names
                data = []
                for row in result:
                    row_dict = {}
                    for column, value in zip(result.keys(), row):
                        row_dict[column] = value
                    data.append(row_dict)
            
            df = pd.DataFrame(data)
            
            # If the dataframe is not empty, ensure timestamp is proper datetime format
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            logger.info(f"Retrieved {len(df)} weather records from database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving weather data: {str(e)}")
            return pd.DataFrame()

    def save_analysis(self, name, description, analysis_type, parameters, results):
        """Save analysis results to database"""
        try:
            # Convert parameters and results to JSON if they're not strings
            if not isinstance(parameters, str):
                parameters = json.dumps(parameters)
            if not isinstance(results, str):
                results = json.dumps(results)
            
            with self.engine.connect() as connection:
                connection.execute(
                    self.saved_analyses.insert().values(
                        name=name,
                        description=description,
                        analysis_type=analysis_type,
                        parameters=parameters,
                        results=results,
                        created_at=datetime.utcnow()
                    )
                )
                connection.commit()
            
            logger.info(f"Successfully saved analysis '{name}' to database")
            return True
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return False

    def get_saved_analyses(self, analysis_type=None):
        """Retrieve saved analyses from database with optional type filtering"""
        try:
            query = self.saved_analyses.select()
            
            if analysis_type:
                query = query.where(self.saved_analyses.c.analysis_type == analysis_type)
            
            query = query.order_by(self.saved_analyses.c.created_at.desc())
            
            with self.engine.connect() as connection:
                result = connection.execute(query)
                
                # Convert each row to a dictionary with proper column names
                data = []
                for row in result:
                    row_dict = {}
                    for column, value in zip(result.keys(), row):
                        row_dict[column] = value
                    data.append(row_dict)
            
            df = pd.DataFrame(data)
            
            # If the dataframe is not empty, ensure created_at is proper datetime format
            if not df.empty and 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
                
            logger.info(f"Retrieved {len(df)} saved analyses from database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving saved analyses: {str(e)}")
            return pd.DataFrame()

    def log_user_upload(self, filename, original_filename, file_size, content_type, metadata=None):
        """Log a user data upload to database"""
        try:
            # Convert metadata to JSON if it's not a string
            if metadata and not isinstance(metadata, str):
                metadata = json.dumps(metadata)
            
            with self.engine.connect() as connection:
                connection.execute(
                    self.user_uploads.insert().values(
                        filename=filename,
                        original_filename=original_filename,
                        file_size=file_size,
                        content_type=content_type,
                        metadata=metadata,
                        created_at=datetime.utcnow()
                    )
                )
                connection.commit()
            
            logger.info(f"Successfully logged user upload '{original_filename}' to database")
            return True
        except Exception as e:
            logger.error(f"Error logging user upload: {str(e)}")
            return False

    def get_locations(self, active_only=True):
        """Retrieve locations from database"""
        try:
            query = self.locations.select()
            
            if active_only:
                query = query.where(self.locations.c.is_active == True)
            
            with self.engine.connect() as connection:
                result = connection.execute(query)
                data = [dict(row) for row in result]
            
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} locations from database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving locations: {str(e)}")
            return pd.DataFrame()

    def add_location(self, name, zip_code, latitude=None, longitude=None):
        """Add a new location to the database"""
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    self.locations.insert().values(
                        name=name,
                        zip_code=zip_code,
                        latitude=latitude,
                        longitude=longitude,
                        is_active=True,
                        created_at=datetime.utcnow()
                    )
                )
                connection.commit()
            
            logger.info(f"Successfully added location '{name}' to database")
            return True
        except Exception as e:
            logger.error(f"Error adding location: {str(e)}")
            return False