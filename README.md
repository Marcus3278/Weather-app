# Weather Analysis & Data Mining Dashboard

A comprehensive weather analysis toolkit consisting of two integrated applications:

1. **Weather Dashboard** - Comprehensive weather visualization and analysis for Olney, IL (62451)
2. **Data Mining Application** - Advanced data mining tools for weather data analysis

## Features

### Weather Dashboard

- **Overview**: Current weather conditions, historical weather analysis, temperature trends, precipitation analysis, wind patterns, and statistical summaries
- **Advanced Statistics**: Correlation analysis, trend analysis, time series decomposition, rolling statistics, stationarity analysis, and autocorrelation
- **Pattern Analysis**: Daily and weekly patterns, heat index analysis, rain event analysis, and wind pattern analysis
- **Extreme Events**: Temperature extremes detection, heat/cold wave identification, and diurnal temperature range analysis
- **Database Integration**: Persistent storage of weather data in PostgreSQL database

### Data Mining Application

- **Data Input**: Upload CSV data or use sample weather data
- **Data Exploration**: Distribution analysis, correlation analysis, scatter plot matrix
- **Clustering Analysis**: K-means clustering with interactive visualization
- **Anomaly Detection**: Isolation Forest algorithm for detecting anomalies
- **Dimensionality Reduction**: PCA with interactive visualization
- **Saved Analyses**: Database storage and retrieval of analysis results with visualization

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualization
- **SciPy**: Scientific and technical computing
- **Scikit-learn**: Machine learning tools
- **Statsmodels**: Statistical modeling
- **PostgreSQL**: Database for persistent storage
- **SQLAlchemy**: SQL toolkit and Object-Relational Mapping

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- PostgreSQL database (optional, but recommended for full functionality)

### Database Setup

This project uses a PostgreSQL database for storing weather data and analysis results. Follow these steps to set up your database:

1. Install PostgreSQL on your system if not already installed
2. Create a new PostgreSQL database for this project
3. Set up the following environment variables:

   - `DATABASE_URL`: Connection string for your PostgreSQL database
   - `PGHOST`: PostgreSQL host
   - `PGPORT`: PostgreSQL port
   - `PGUSER`: PostgreSQL username
   - `PGPASSWORD`: PostgreSQL password
   - `PGDATABASE`: PostgreSQL database name

4. The application will automatically create the necessary tables on first run:
   - weather_data: Stores historical weather data
   - saved_analyses: Stores saved analysis results
   - user_uploads: Logs data uploaded by users
   - locations: Stores location information

You can use a `.env` file to store these variables locally during development.

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/weather-analysis-dashboard.git
   cd weather-analysis-dashboard
   ```

2. Install dependencies:
   ```
   pip install -r dependencies.txt
   ```

3. Run the applications:
   
   **Weather Dashboard**:
   ```
   streamlit run main.py
   ```
   
   **Data Mining Application**:
   ```
   streamlit run data_mining_app.py --server.port 5001
   ```

## Project Structure

- `main.py` - Weather Dashboard entry point
- `data_mining_app.py` - Data Mining Application entry point
- `utils/` - Utility functions
  - `weather_api.py` - Weather data generation
  - `data_processor.py` - Data processing and statistical analysis
- `components/` - Visualization components
  - `visualization.py` - Charts and visualization functions
- `.streamlit/` - Streamlit configuration

## Notes

This project uses synthetic weather data to demonstrate capabilities without requiring external API keys.

## Deployment Options

### Local Deployment
The instructions in the "Installation" section will set up the applications to run locally on your machine.

### Cloud Deployment
This application can be deployed to various cloud platforms:

1. **Streamlit Cloud**:
   - Sign up for a [Streamlit Cloud](https://streamlit.io/cloud) account
   - Connect your GitHub repository
   - Deploy directly from your GitHub repository
   - Configure secrets for database connection

2. **Heroku**:
   - Create a `Procfile` with: `web: streamlit run main.py --server.port=$PORT`
   - Add a `runtime.txt` file with: `python-3.9.0` (or your Python version)
   - Create an app on Heroku and deploy using their CLI or GitHub integration
   - Set up environment variables in Heroku for database connection

3. **Docker**:
   - Use the included Dockerfile (if available) or create one
   - Build and push the Docker image to a registry
   - Deploy to any platform that supports Docker containers

Remember to set up your PostgreSQL database and configure the appropriate environment variables on your chosen deployment platform.

## Contributing

Contributions to improve the Weather Analysis & Data Mining Dashboard are welcome. Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
