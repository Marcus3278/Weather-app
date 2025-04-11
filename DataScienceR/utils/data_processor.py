import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks

class DataProcessor:
    def process_historical_data(self, raw_data):
        """Process raw weather data into pandas DataFrame"""
        # If raw_data is already a DataFrame (from the API implementation)
        if isinstance(raw_data, pd.DataFrame):
            return raw_data
            
        # Otherwise process it from the API JSON response
        hourly_data = raw_data.get("hourly", [])
        
        processed_data = []
        for entry in hourly_data:
            processed_data.append({
                "timestamp": pd.to_datetime(entry["dt"], unit="s"),
                "temperature": entry["temp"],
                "feels_like": entry["feels_like"],
                "humidity": entry["humidity"],
                "wind_speed": entry["wind_speed"],
                "wind_deg": entry["wind_deg"],
                "precipitation": entry.get("rain", {}).get("1h", 0),
                "pressure": entry["pressure"]
            })

        return pd.DataFrame(processed_data)

    def calculate_statistics(self, df):
        """Calculate comprehensive statistical analysis of weather data"""
        # Basic statistics
        basic_stats = {
            "Metric": ["Temperature (°F)", "Humidity (%)", "Wind Speed (mph)", "Precipitation (mm)", "Pressure (hPa)"],
            "Mean": [
                round(df["temperature"].mean(), 1),
                round(df["humidity"].mean(), 1),
                round(df["wind_speed"].mean(), 1),
                round(df["precipitation"].mean(), 1),
                round(df["pressure"].mean(), 1) if "pressure" in df.columns else np.nan
            ],
            "Median": [
                round(df["temperature"].median(), 1),
                round(df["humidity"].median(), 1),
                round(df["wind_speed"].median(), 1),
                round(df["precipitation"].median(), 1),
                round(df["pressure"].median(), 1) if "pressure" in df.columns else np.nan
            ],
            "Min": [
                round(df["temperature"].min(), 1),
                round(df["humidity"].min(), 1),
                round(df["wind_speed"].min(), 1),
                round(df["precipitation"].min(), 1),
                round(df["pressure"].min(), 1) if "pressure" in df.columns else np.nan
            ],
            "Max": [
                round(df["temperature"].max(), 1),
                round(df["humidity"].max(), 1),
                round(df["wind_speed"].max(), 1),
                round(df["precipitation"].max(), 1),
                round(df["pressure"].max(), 1) if "pressure" in df.columns else np.nan
            ],
            "Range": [
                round(df["temperature"].max() - df["temperature"].min(), 1),
                round(df["humidity"].max() - df["humidity"].min(), 1),
                round(df["wind_speed"].max() - df["wind_speed"].min(), 1),
                round(df["precipitation"].max() - df["precipitation"].min(), 1),
                round(df["pressure"].max() - df["pressure"].min(), 1) if "pressure" in df.columns else np.nan
            ],
            "Std Dev": [
                round(df["temperature"].std(), 2),
                round(df["humidity"].std(), 2),
                round(df["wind_speed"].std(), 2),
                round(df["precipitation"].std(), 2),
                round(df["pressure"].std(), 2) if "pressure" in df.columns else np.nan
            ],
            "Variance": [
                round(df["temperature"].var(), 2),
                round(df["humidity"].var(), 2),
                round(df["wind_speed"].var(), 2),
                round(df["precipitation"].var(), 2),
                round(df["pressure"].var(), 2) if "pressure" in df.columns else np.nan
            ],
            "Skewness": [
                round(df["temperature"].skew(), 2),
                round(df["humidity"].skew(), 2),
                round(df["wind_speed"].skew(), 2),
                round(df["precipitation"].skew(), 2),
                round(df["pressure"].skew(), 2) if "pressure" in df.columns else np.nan
            ],
            "Kurtosis": [
                round(df["temperature"].kurtosis(), 2),
                round(df["humidity"].kurtosis(), 2),
                round(df["wind_speed"].kurtosis(), 2),
                round(df["precipitation"].kurtosis(), 2),
                round(df["pressure"].kurtosis(), 2) if "pressure" in df.columns else np.nan
            ],
            "Coefficient of Variation": [
                round(df["temperature"].std() / df["temperature"].mean() * 100, 2),
                round(df["humidity"].std() / df["humidity"].mean() * 100, 2),
                round(df["wind_speed"].std() / df["wind_speed"].mean() * 100, 2),
                round(df["precipitation"].std() / df["precipitation"].mean() * 100, 2) if df["precipitation"].mean() > 0 else np.nan,
                round(df["pressure"].std() / df["pressure"].mean() * 100, 2) if "pressure" in df.columns else np.nan
            ]
        }

        return pd.DataFrame(basic_stats)

    def get_advanced_statistics(self, df):
        """Calculate advanced statistical metrics and trends"""
        # Time series analysis
        df_temp = df.set_index('timestamp')['temperature']

        # Decompose time series
        decomposition = seasonal_decompose(df_temp, period=24, extrapolate_trend='freq')

        # Calculate rolling statistics (24-hour window)
        rolling_mean = df_temp.rolling(window=24).mean()
        rolling_std = df_temp.rolling(window=24).std()
        rolling_min = df_temp.rolling(window=24).min()
        rolling_max = df_temp.rolling(window=24).max()

        # Calculate Coefficient of Variation (CV) over time
        rolling_cv = rolling_std / rolling_mean * 100

        # Correlation analysis
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        if 'pressure' in df.columns:
            numeric_cols.append('pressure')
        
        correlation_matrix = df[numeric_cols].corr().round(2)

        # Mann-Kendall trend test for temperature
        trend_test = stats.kendalltau(df_temp.values, np.arange(len(df_temp)))

        # Stationarity test (Augmented Dickey-Fuller)
        adf_result = adfuller(df_temp.dropna())
        
        # Autocorrelation and Partial Autocorrelation
        lag = min(40, len(df_temp) // 2)
        acf_values = acf(df_temp.dropna(), nlags=lag)
        pacf_values = pacf(df_temp.dropna(), nlags=lag)
        
        # LOWESS smoothing for trend analysis
        if len(df_temp) > 10:  # Need enough data for smoothing
            smoothed = lowess(df_temp.values, np.arange(len(df_temp)), frac=0.1)
            trend_smooth = pd.Series(smoothed[:, 1], index=df_temp.index)
        else:
            trend_smooth = df_temp  # Fallback if not enough data
        
        # Extreme value analysis
        threshold_high = df_temp.mean() + 2 * df_temp.std()
        threshold_low = df_temp.mean() - 2 * df_temp.std()
        extreme_high = df_temp[df_temp > threshold_high]
        extreme_low = df_temp[df_temp < threshold_low]
        
        # Peak detection
        peaks, _ = find_peaks(df_temp.values, height=threshold_high, distance=8)
        troughs, _ = find_peaks(-df_temp.values, height=-threshold_low, distance=8)
        
        # Weather pattern transition counts
        daily_data = df.set_index('timestamp').resample('D').mean()
        if len(daily_data) > 1:  # Need at least 2 days
            # Count warming and cooling days
            temp_diff = daily_data['temperature'].diff()
            warming_days = (temp_diff > 0).sum()
            cooling_days = (temp_diff < 0).sum()
            
            # Count precipitation transitions
            if 'precipitation' in daily_data.columns:
                precip = daily_data['precipitation']
                rain_to_dry = ((precip.shift(1) > 0.1) & (precip <= 0.1)).sum()
                dry_to_rain = ((precip.shift(1) <= 0.1) & (precip > 0.1)).sum()
                pattern_transitions = {
                    'warming_days': int(warming_days),
                    'cooling_days': int(cooling_days),
                    'rain_to_dry': int(rain_to_dry),
                    'dry_to_rain': int(dry_to_rain)
                }
            else:
                pattern_transitions = {
                    'warming_days': int(warming_days),
                    'cooling_days': int(cooling_days)
                }
        else:
            pattern_transitions = {'warming_days': 0, 'cooling_days': 0}
        
        # Calculate diurnal temperature range (daily max - min)
        daily_range = df.set_index('timestamp')['temperature'].resample('D').max() - \
                    df.set_index('timestamp')['temperature'].resample('D').min()
        
        # Heat wave and cold wave detection
        # A simple definition: 3+ consecutive days above/below thresholds
        daily_high = df.set_index('timestamp')['temperature'].resample('D').max()
        daily_low = df.set_index('timestamp')['temperature'].resample('D').min()
        
        heat_wave_days = 0
        cold_wave_days = 0
        
        if len(daily_high) >= 3:
            heat_threshold = daily_high.quantile(0.9)  # 90th percentile
            cold_threshold = daily_low.quantile(0.1)   # 10th percentile
            
            # Count days in heat/cold waves
            heat_days = (daily_high > heat_threshold).astype(int)
            cold_days = (daily_low < cold_threshold).astype(int)
            
            # Rolling sum with window 3 to find consecutive days
            heat_streak = heat_days.rolling(window=3).sum()
            cold_streak = cold_days.rolling(window=3).sum()
            
            # Days that are part of 3+ consecutive days above/below threshold
            heat_wave_days = (heat_streak >= 3).sum()
            cold_wave_days = (cold_streak >= 3).sum()
        
        # Compile all results
        return {
            'decomposition': decomposition,
            'rolling_stats': {
                'mean': rolling_mean,
                'std': rolling_std,
                'min': rolling_min,
                'max': rolling_max,
                'cv': rolling_cv
            },
            'correlations': correlation_matrix,
            'trend_analysis': {
                'tau': round(trend_test.statistic, 3),
                'p_value': round(trend_test.pvalue, 3)
            },
            'stationarity': {
                'adf_statistic': round(adf_result[0], 3),
                'p_value': round(adf_result[1], 3),
                'is_stationary': adf_result[1] < 0.05
            },
            'autocorrelation': {
                'acf': acf_values,
                'pacf': pacf_values
            },
            'trend_smooth': trend_smooth,
            'extremes': {
                'high_count': len(extreme_high),
                'low_count': len(extreme_low),
                'high_values': extreme_high,
                'low_values': extreme_low,
                'peaks': peaks,
                'troughs': troughs
            },
            'pattern_transitions': pattern_transitions,
            'diurnal_range': {
                'mean': daily_range.mean(),
                'max': daily_range.max(),
                'values': daily_range
            },
            'wave_analysis': {
                'heat_wave_days': heat_wave_days,
                'cold_wave_days': cold_wave_days
            }
        }
        
    def perform_advanced_pattern_analysis(self, df):
        """Perform pattern analysis looking for specific weather patterns"""
        # Ensure we have timestamp as index
        if 'timestamp' in df.columns:
            df_indexed = df.set_index('timestamp')
        else:
            df_indexed = df.copy()
            
        # Group by hour of day to analyze daily patterns
        hourly_patterns = df_indexed.groupby(df_indexed.index.hour).mean()
        
        # Group by day of week to analyze weekly patterns
        weekly_patterns = df_indexed.groupby(df_indexed.index.dayofweek).mean()
        
        # Calculate temperature-humidity relationship (heat index approximation)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] * (1 + 0.348 * df['humidity'] / 100)
            
            # Bin temperature and humidity for heat stress analysis
            temp_bins = pd.cut(df['temperature'], bins=5)
            humidity_bins = pd.cut(df['humidity'], bins=5)
            heat_stress_matrix = pd.crosstab(temp_bins, humidity_bins)
        else:
            heat_stress_matrix = None
            
        # Rain event analysis
        if 'precipitation' in df.columns:
            # Define rain event as precipitation > 0.1mm
            df['rain_event'] = df['precipitation'] > 0.1
            
            # Identify consecutive rain hours
            df['rain_event_group'] = (df['rain_event'] != df['rain_event'].shift()).cumsum()
            rain_events = df[df['rain_event']].groupby('rain_event_group')
            
            # Calculate event statistics
            rain_event_stats = {
                'total_events': len(rain_events),
                'avg_duration_hours': rain_events.size().mean() if len(rain_events) > 0 else 0,
                'max_duration_hours': rain_events.size().max() if len(rain_events) > 0 else 0,
                'avg_intensity': rain_events['precipitation'].mean().mean() if len(rain_events) > 0 else 0,
                'max_intensity': rain_events['precipitation'].max().max() if len(rain_events) > 0 else 0,
            }
        else:
            rain_event_stats = None
            
        # Wind pattern analysis
        if 'wind_speed' in df.columns and 'wind_deg' in df.columns:
            # Convert degrees to cardinal directions
            cardinal_bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
            cardinal_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
            
            df['wind_dir'] = pd.cut(df['wind_deg'], 
                                    bins=cardinal_bins, 
                                    labels=cardinal_labels,
                                    include_lowest=True,
                                    ordered=False)  # Set ordered to False to allow duplicate labels
                
            # Count frequency by direction
            wind_dir_freq = df['wind_dir'].value_counts().sort_index()
            
            # Calculate average wind speed by direction
            wind_speed_by_dir = df.groupby('wind_dir')['wind_speed'].mean()
            
            wind_analysis = {
                'direction_frequency': wind_dir_freq,
                'speed_by_direction': wind_speed_by_dir,
                'max_speed': df['wind_speed'].max(),
                'calm_periods': (df['wind_speed'] < 2).sum() / len(df) if len(df) > 0 else 0,  # percentage of calm periods
                'prevailing_direction': wind_dir_freq.idxmax() if not wind_dir_freq.empty else None
            }
        else:
            wind_analysis = None
        
        return {
            'hourly_patterns': hourly_patterns,
            'weekly_patterns': weekly_patterns,
            'heat_stress_matrix': heat_stress_matrix,
            'rain_event_stats': rain_event_stats,
            'wind_analysis': wind_analysis
        }