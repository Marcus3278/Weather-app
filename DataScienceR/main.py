import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.weather_api import WeatherAPI
from utils.data_processor import DataProcessor
from components.visualization import (
    create_temperature_chart,
    create_precipitation_chart,
    create_wind_rose,
    create_historical_comparison
)

# Page configuration
st.set_page_config(
    page_title="Weather Dashboard - Olney, IL (62451)",
    page_icon="🌤",
    layout="wide"
)

# Initialize API and data processor
weather_api = WeatherAPI()
data_processor = DataProcessor()

def main():
    # Header
    st.title("📊 Weather Dashboard - Olney, IL (62451)")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                          ["Overview", "Advanced Statistics", "Pattern Analysis", "Extreme Events"])

    # Get historical data
    historical_data = weather_api.get_historical_weather()
    processed_data = data_processor.process_historical_data(historical_data)
    
    if page == "Overview":
        # Current Weather Section
        st.header("Current Weather Conditions")
        current_weather = weather_api.get_current_weather()

        # Display current weather in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperature", f"{current_weather['temp']}°F", 
                    f"{current_weather['temp_change']}°F")
        with col2:
            st.metric("Humidity", f"{current_weather['humidity']}%")
        with col3:
            st.metric("Wind Speed", f"{current_weather['wind_speed']} mph")
        with col4:
            st.metric("Pressure", f"{current_weather['pressure']} hPa")

        # Historical Data Section
        st.header("Historical Weather Analysis")

        # Time range selector
        time_range = st.selectbox(
            "Select Time Range",
            ["Daily", "Weekly", "Monthly"]
        )

        # Temperature Trends
        st.subheader("Temperature Trends")
        temp_chart = create_temperature_chart(processed_data)
        st.plotly_chart(temp_chart, use_container_width=True)

        # Precipitation Data
        st.subheader("Precipitation Analysis")
        precip_chart = create_precipitation_chart(processed_data)
        st.plotly_chart(precip_chart, use_container_width=True)

        # Wind Patterns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Wind Patterns")
            wind_rose = create_wind_rose(processed_data)
            st.plotly_chart(wind_rose)
        
        with col2:
            st.subheader("Historical Comparison")
            hist_comparison = create_historical_comparison(processed_data)
            st.plotly_chart(hist_comparison)
            
        # Basic Statistics Table
        st.header("Statistical Summary")
        stats_df = data_processor.calculate_statistics(processed_data)
        st.dataframe(stats_df, use_container_width=True)
        
    elif page == "Advanced Statistics":
        st.header("Advanced Statistical Analysis")
        
        # Get advanced statistics
        advanced_stats = data_processor.get_advanced_statistics(processed_data)
        
        # Tabs for different statistical analyses
        stats_tab = st.tabs(["Correlation Analysis", "Trend Analysis", "Time Series Components", 
                           "Rolling Statistics", "Stationarity", "Autocorrelation"])
        
        with stats_tab[0]:
            # Correlation matrix heatmap
            st.subheader("Weather Parameter Correlations")
            fig_corr = px.imshow(
                advanced_stats['correlations'],
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            # Add text annotations manually
            for i in range(len(advanced_stats['correlations'].index)):
                for j in range(len(advanced_stats['correlations'].columns)):
                    fig_corr.add_annotation(
                        x=j, 
                        y=i,
                        text=str(advanced_stats['correlations'].values[i, j]),
                        showarrow=False
                    )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            ### Interpreting the Correlation Matrix
            
            This heatmap shows how different weather parameters are related to each other. Values close to:
            - **1.0** indicate strong positive correlation (both values increase together)
            - **-1.0** indicate strong negative correlation (one value increases as the other decreases)
            - **0.0** indicate no correlation (the variables are independent)
            """)
            
        with stats_tab[1]:
            # Trend analysis
            st.subheader("Temperature Trend Analysis")
            
            # Display Mann-Kendall trend test results
            trend = advanced_stats['trend_analysis']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trend Coefficient (τ)", f"{trend['tau']}")
                st.markdown("""
                **Kendall's Tau**: Measures the strength and direction of the monotonic relationship 
                between temperature and time. Range: -1 to 1.
                """)
            with col2:
                st.metric("P-value", f"{trend['p_value']}")
                st.markdown("""
                **P-value**: The probability that the observed trend occurred by chance. 
                Values < 0.05 indicate statistically significant trends.
                """)

            if trend['p_value'] < 0.05:
                if trend['tau'] > 0:
                    st.info("🔍 Statistically significant warming trend detected")
                else:
                    st.info("🔍 Statistically significant cooling trend detected")
            else:
                st.info("🔍 No statistically significant trend detected in temperature data")
                
            # Display LOWESS smoothed trend
            st.subheader("Smoothed Temperature Trend")
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=advanced_stats['trend_smooth'].index,
                y=processed_data.set_index('timestamp')['temperature'],
                name='Temperature',
                line=dict(color='lightblue', width=1)
            ))
            fig_trend.add_trace(go.Scatter(
                x=advanced_stats['trend_smooth'].index,
                y=advanced_stats['trend_smooth'],
                name='Smoothed Trend',
                line=dict(color='red', width=3)
            ))
            fig_trend.update_layout(
                title="Temperature Trend with LOWESS Smoothing",
                xaxis_title="Time",
                yaxis_title="Temperature (°F)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with stats_tab[2]:
            # Time series decomposition
            st.subheader("Time Series Decomposition")
            st.markdown("""
            Time series decomposition breaks down temperature data into three components:
            - **Trend**: The long-term progression of the series
            - **Seasonal**: The repeating, seasonal patterns
            - **Residual**: The random variation in the data
            """)
            
            decomp = advanced_stats['decomposition']
            
            # Create figure with subplots
            fig_decomp = go.Figure()
            
            # Original data
            fig_decomp.add_trace(go.Scatter(
                x=decomp.observed.index,
                y=decomp.observed,
                name='Original Data',
                line=dict(color='black')
            ))
            
            # Trend component
            fig_decomp.add_trace(go.Scatter(
                x=decomp.trend.index,
                y=decomp.trend,
                name='Trend Component',
                line=dict(color='red')
            ))
            
            # Seasonal component
            fig_decomp.add_trace(go.Scatter(
                x=decomp.seasonal.index,
                y=decomp.seasonal,
                name='Seasonal Component',
                line=dict(color='green')
            ))
            
            # Residual component
            fig_decomp.add_trace(go.Scatter(
                x=decomp.resid.index,
                y=decomp.resid,
                name='Residual Component',
                line=dict(color='blue')
            ))
            
            fig_decomp.update_layout(
                height=600,
                title='Time Series Decomposition of Temperature Data',
                xaxis_title='Time',
                yaxis_title='Temperature (°F)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_decomp, use_container_width=True)
            
        with stats_tab[3]:
            # Rolling statistics
            st.subheader("24-Hour Rolling Statistics")
            
            rolling_stats = advanced_stats['rolling_stats']
            
            # Display rolling statistics
            fig_rolling = go.Figure()
            
            # Add rolling mean
            fig_rolling.add_trace(go.Scatter(
                x=rolling_stats['mean'].index,
                y=rolling_stats['mean'],
                name='Rolling Mean',
                line=dict(color='blue', width=2)
            ))
            
            # Add rolling standard deviation
            fig_rolling.add_trace(go.Scatter(
                x=rolling_stats['std'].index,
                y=rolling_stats['std'],
                name='Rolling Std Dev',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add min and max as a range
            fig_rolling.add_trace(go.Scatter(
                x=rolling_stats['max'].index,
                y=rolling_stats['max'],
                name='Rolling Max',
                line=dict(color='rgba(0,100,0,0.3)', width=0)
            ))
            
            fig_rolling.add_trace(go.Scatter(
                x=rolling_stats['min'].index,
                y=rolling_stats['min'],
                name='Rolling Min',
                fill='tonexty', 
                fillcolor='rgba(0,100,0,0.1)',
                line=dict(color='rgba(0,100,0,0.3)', width=0)
            ))
            
            # Coefficient of variation on secondary y-axis
            fig_rolling.add_trace(go.Scatter(
                x=rolling_stats['cv'].index,
                y=rolling_stats['cv'],
                name='Coefficient of Variation (%)',
                line=dict(color='purple', width=2),
                yaxis='y2'
            ))
            
            fig_rolling.update_layout(
                height=500,
                title="24-Hour Rolling Temperature Statistics",
                xaxis_title="Time",
                yaxis_title="Temperature (°F)",
                yaxis2=dict(
                    title="Coefficient of Variation (%)",
                    overlaying="y",
                    side="right"
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            st.markdown("""
            ### Interpreting Rolling Statistics
            
            - **Rolling Mean**: The average temperature over the previous 24 hours
            - **Rolling Std Dev**: How variable the temperature was over the previous 24 hours
            - **Rolling Min/Max Range**: The range of temperatures over the previous 24 hours
            - **Coefficient of Variation**: Normalized measure of dispersion (std/mean as percentage)
            """)
            
        with stats_tab[4]:
            # Stationarity analysis
            st.subheader("Stationarity Analysis")
            
            # ADF test results
            stationarity = advanced_stats['stationarity']
            
            st.markdown(f"""
            **Augmented Dickey-Fuller Test Results**
            
            The ADF test checks if a time series is stationary (its statistical properties do not change over time).
            
            - **ADF Statistic**: {stationarity['adf_statistic']}
            - **P-value**: {stationarity['p_value']}
            - **Is Stationary**: {stationarity['is_stationary']}
            
            *A smaller p-value (< 0.05) indicates the temperature series is stationary (no significant trend).*
            """)
            
            # Plot the data with threshold bands
            df_temp = processed_data.set_index('timestamp')['temperature']
            
            fig_stat = go.Figure()
            
            # Original data
            fig_stat.add_trace(go.Scatter(
                x=df_temp.index,
                y=df_temp.values,
                name='Temperature',
                line=dict(color='blue')
            ))
            
            # Mean line
            fig_stat.add_trace(go.Scatter(
                x=[df_temp.index.min(), df_temp.index.max()],
                y=[df_temp.mean(), df_temp.mean()],
                name='Mean',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Upper stationary threshold
            fig_stat.add_trace(go.Scatter(
                x=[df_temp.index.min(), df_temp.index.max()],
                y=[df_temp.mean() + df_temp.std(), df_temp.mean() + df_temp.std()],
                name='Mean + 1 Std',
                line=dict(color='green', dash='dot')
            ))
            
            # Lower stationary threshold
            fig_stat.add_trace(go.Scatter(
                x=[df_temp.index.min(), df_temp.index.max()],
                y=[df_temp.mean() - df_temp.std(), df_temp.mean() - df_temp.std()],
                name='Mean - 1 Std',
                line=dict(color='green', dash='dot')
            ))
            
            fig_stat.update_layout(
                height=400,
                title="Temperature Data with Stationarity Bands",
                xaxis_title="Time",
                yaxis_title="Temperature (°F)"
            )
            
            st.plotly_chart(fig_stat, use_container_width=True)
            
        with stats_tab[5]:
            # Autocorrelation analysis
            st.subheader("Autocorrelation Analysis")
            
            st.markdown("""
            Autocorrelation measures how a time series is correlated with itself at different time lags.
            
            - **ACF (Autocorrelation Function)**: Correlation between the time series and its lagged values
            - **PACF (Partial Autocorrelation Function)**: Correlation between the time series and its lagged values, after removing effects of shorter lags
            
            Strong patterns in these functions can indicate seasonality and help in forecasting models.
            """)
            
            # ACF plot
            acf_values = advanced_stats['autocorrelation']['acf']
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color='blue'
            ))
            
            # Add confidence intervals (typically +/- 1.96/sqrt(N))
            conf_int = 1.96 / np.sqrt(len(processed_data))
            
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=[conf_int] * len(acf_values),
                name='Upper CI',
                line=dict(color='red', dash='dash')
            ))
            
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=[-conf_int] * len(acf_values),
                name='Lower CI',
                line=dict(color='red', dash='dash')
            ))
            
            fig_acf.update_layout(
                height=300,
                title='Autocorrelation Function (ACF)',
                xaxis_title='Lag',
                yaxis_title='Correlation',
                showlegend=False
            )
            
            # PACF plot
            pacf_values = advanced_stats['autocorrelation']['pacf']
            
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name='PACF',
                marker_color='green'
            ))
            
            # Add confidence intervals
            fig_pacf.add_trace(go.Scatter(
                x=list(range(len(pacf_values))),
                y=[conf_int] * len(pacf_values),
                name='Upper CI',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pacf.add_trace(go.Scatter(
                x=list(range(len(pacf_values))),
                y=[-conf_int] * len(pacf_values),
                name='Lower CI',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pacf.update_layout(
                height=300,
                title='Partial Autocorrelation Function (PACF)',
                xaxis_title='Lag',
                yaxis_title='Correlation',
                showlegend=False
            )
            
            st.plotly_chart(fig_acf, use_container_width=True)
            st.plotly_chart(fig_pacf, use_container_width=True)
            
            st.markdown("""
            **Key Insights from ACF/PACF:**
            
            - **Spikes at regular intervals in ACF**: Indicate seasonal patterns
            - **Slow decay in ACF**: Suggests non-stationarity
            - **Significant spikes in PACF**: Help identify appropriate lag orders for forecasting models
            """)
            
    elif page == "Pattern Analysis":
        st.header("Weather Pattern Analysis")
        
        # Get pattern analysis
        pattern_analysis = data_processor.perform_advanced_pattern_analysis(processed_data)
        
        # Tabs for different pattern analyses
        pattern_tab = st.tabs(["Daily Patterns", "Weekly Patterns", "Heat Index Analysis", 
                             "Rain Event Analysis", "Wind Pattern Analysis"])
        
        with pattern_tab[0]:
            # Daily patterns analysis
            st.subheader("Daily Weather Patterns (Hour of Day)")
            
            hourly_patterns = pattern_analysis['hourly_patterns']
            
            # Create multiline chart for hourly patterns
            fig_hourly = go.Figure()
            
            # Add temperature
            fig_hourly.add_trace(go.Scatter(
                x=hourly_patterns.index,
                y=hourly_patterns['temperature'],
                name='Temperature (°F)',
                line=dict(color='red', width=2)
            ))
            
            # Add humidity on secondary y-axis
            fig_hourly.add_trace(go.Scatter(
                x=hourly_patterns.index,
                y=hourly_patterns['humidity'],
                name='Humidity (%)',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            fig_hourly.update_layout(
                height=400,
                title="Average Temperature and Humidity by Hour of Day",
                xaxis=dict(
                    title="Hour of Day",
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f"{h:02d}:00" for h in range(24)]
                ),
                yaxis=dict(title="Temperature (°F)"),
                yaxis2=dict(
                    title="Humidity (%)",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Wind speed pattern by hour
            if 'wind_speed' in hourly_patterns.columns:
                fig_wind = go.Figure()
                
                fig_wind.add_trace(go.Bar(
                    x=hourly_patterns.index,
                    y=hourly_patterns['wind_speed'],
                    name='Wind Speed (mph)',
                    marker_color='lightblue'
                ))
                
                fig_wind.update_layout(
                    height=300,
                    title="Average Wind Speed by Hour of Day",
                    xaxis=dict(
                        title="Hour of Day",
                        tickmode='array',
                        tickvals=list(range(24)),
                        ticktext=[f"{h:02d}:00" for h in range(24)]
                    ),
                    yaxis=dict(title="Wind Speed (mph)")
                )
                
                st.plotly_chart(fig_wind, use_container_width=True)
            
        with pattern_tab[1]:
            # Weekly patterns analysis
            st.subheader("Weekly Weather Patterns (Day of Week)")
            
            weekly_patterns = pattern_analysis['weekly_patterns']
            
            # Map day of week numbers to names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create multiline chart for weekly patterns
            fig_weekly = go.Figure()
            
            # Add temperature
            fig_weekly.add_trace(go.Scatter(
                x=day_names,
                y=weekly_patterns['temperature'],
                name='Temperature (°F)',
                line=dict(color='red', width=2)
            ))
            
            # Add humidity on secondary y-axis
            fig_weekly.add_trace(go.Scatter(
                x=day_names,
                y=weekly_patterns['humidity'],
                name='Humidity (%)',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            fig_weekly.update_layout(
                height=400,
                title="Average Temperature and Humidity by Day of Week",
                xaxis=dict(title="Day of Week"),
                yaxis=dict(title="Temperature (°F)"),
                yaxis2=dict(
                    title="Humidity (%)",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Precipitation by day of week
            if 'precipitation' in weekly_patterns.columns:
                fig_precip = go.Figure()
                
                fig_precip.add_trace(go.Bar(
                    x=day_names,
                    y=weekly_patterns['precipitation'],
                    name='Precipitation (mm)',
                    marker_color='lightblue'
                ))
                
                fig_precip.update_layout(
                    height=300,
                    title="Average Precipitation by Day of Week",
                    xaxis=dict(title="Day of Week"),
                    yaxis=dict(title="Precipitation (mm)")
                )
                
                st.plotly_chart(fig_precip, use_container_width=True)
            
        with pattern_tab[2]:
            # Heat index analysis
            st.subheader("Heat Index Analysis")
            
            heat_stress_matrix = pattern_analysis['heat_stress_matrix']
            
            if heat_stress_matrix is not None:
                # Create heatmap
                fig_heat = px.imshow(
                    heat_stress_matrix,
                    labels=dict(
                        x="Humidity Range",
                        y="Temperature Range",
                        color="Count"
                    ),
                    x=heat_stress_matrix.columns,
                    y=heat_stress_matrix.index,
                    color_continuous_scale="Viridis"
                )
                
                fig_heat.update_layout(
                    height=500,
                    title="Temperature-Humidity Distribution (Heat Stress Matrix)"
                )
                
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.markdown("""
                ### Heat Index Analysis
                
                The heat stress matrix shows the distribution of temperature and humidity combinations. 
                Higher values in the upper right corner indicate potentially uncomfortable or hazardous conditions 
                (high temperature combined with high humidity).
                """)
                
                # Calculate basic heat index statistics if available
                if 'heat_index' in processed_data.columns:
                    heat_stats = {
                        "Mean Heat Index": f"{processed_data['heat_index'].mean():.1f}°F",
                        "Max Heat Index": f"{processed_data['heat_index'].max():.1f}°F",
                        "Hours of Discomfort (Heat Index > 80°F)": f"{(processed_data['heat_index'] > 80).sum()} hours",
                        "Hours of Heat Caution (Heat Index > 90°F)": f"{(processed_data['heat_index'] > 90).sum()} hours"
                    }
                    
                    st.table(pd.DataFrame(heat_stats.items(), columns=["Metric", "Value"]))
            else:
                st.info("Heat index analysis requires both temperature and humidity data.")
            
        with pattern_tab[3]:
            # Rain event analysis
            st.subheader("Rain Event Analysis")
            
            rain_event_stats = pattern_analysis['rain_event_stats']
            
            if rain_event_stats is not None:
                # Display rain event statistics
                rain_stats_df = pd.DataFrame({
                    "Metric": [
                        "Total Rain Events",
                        "Average Duration",
                        "Maximum Duration",
                        "Average Intensity",
                        "Maximum Intensity"
                    ],
                    "Value": [
                        f"{rain_event_stats['total_events']} events",
                        f"{rain_event_stats['avg_duration_hours']:.1f} hours",
                        f"{rain_event_stats['max_duration_hours']:.1f} hours",
                        f"{rain_event_stats['avg_intensity']:.2f} mm/hour",
                        f"{rain_event_stats['max_intensity']:.2f} mm/hour"
                    ]
                })
                
                st.table(rain_stats_df)
                
                # Precipitation distribution
                fig_precip_dist = px.histogram(
                    processed_data, 
                    x="precipitation",
                    nbins=20,
                    title="Distribution of Precipitation Amounts"
                )
                
                fig_precip_dist.update_layout(
                    height=300,
                    xaxis_title="Precipitation (mm)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig_precip_dist, use_container_width=True)
                
                # Weather pattern transitions
                if 'pattern_transitions' in advanced_stats:
                    pattern_trans = advanced_stats['pattern_transitions']
                    
                    if 'rain_to_dry' in pattern_trans and 'dry_to_rain' in pattern_trans:
                        transitions_data = {
                            "Transition Type": ["Warming Days", "Cooling Days", "Rain to Dry", "Dry to Rain"],
                            "Count": [
                                pattern_trans['warming_days'],
                                pattern_trans['cooling_days'],
                                pattern_trans['rain_to_dry'],
                                pattern_trans['dry_to_rain']
                            ]
                        }
                        
                        transitions_df = pd.DataFrame(transitions_data)
                        
                        fig_trans = px.bar(
                            transitions_df,
                            x="Transition Type",
                            y="Count",
                            title="Weather Pattern Transitions"
                        )
                        
                        st.plotly_chart(fig_trans, use_container_width=True)
            else:
                st.info("Rain event analysis requires precipitation data.")
            
        with pattern_tab[4]:
            # Wind pattern analysis
            st.subheader("Wind Pattern Analysis")
            
            wind_analysis = pattern_analysis['wind_analysis']
            
            if wind_analysis is not None:
                # Display wind summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Wind Speed", f"{wind_analysis['max_speed']:.1f} mph")
                
                with col2:
                    st.metric("Calm Periods", f"{wind_analysis['calm_periods'] * 100:.1f}%")
                
                with col3:
                    st.metric("Prevailing Direction", f"{wind_analysis['prevailing_direction']}")
                
                # Wind direction frequency
                wind_dir_freq = wind_analysis['direction_frequency']
                
                fig_wind_dir = px.bar(
                    x=wind_dir_freq.index,
                    y=wind_dir_freq.values,
                    title="Wind Direction Frequency"
                )
                
                fig_wind_dir.update_layout(
                    height=350,
                    xaxis_title="Wind Direction",
                    yaxis_title="Frequency",
                    xaxis_categoryorder='array',
                    xaxis_categoryarray=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                )
                
                st.plotly_chart(fig_wind_dir, use_container_width=True)
                
                # Average wind speed by direction
                wind_speed_by_dir = wind_analysis['speed_by_direction']
                
                fig_wind_speed = px.bar(
                    x=wind_speed_by_dir.index,
                    y=wind_speed_by_dir.values,
                    title="Average Wind Speed by Direction"
                )
                
                fig_wind_speed.update_layout(
                    height=350,
                    xaxis_title="Wind Direction",
                    yaxis_title="Avg Speed (mph)",
                    xaxis_categoryorder='array',
                    xaxis_categoryarray=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                )
                
                st.plotly_chart(fig_wind_speed, use_container_width=True)
            else:
                st.info("Wind pattern analysis requires wind speed and direction data.")
                
    elif page == "Extreme Events":
        st.header("Extreme Weather Events Analysis")
        
        # Get advanced statistics
        advanced_stats = data_processor.get_advanced_statistics(processed_data)
        
        # Extreme temperature events
        st.subheader("Temperature Extremes")
        
        extremes = advanced_stats['extremes']
        
        # Display counts of extreme events
        col1, col2 = st.columns(2)
        with col1:
            st.metric("High Temperature Events", extremes['high_count'])
        with col2:
            st.metric("Low Temperature Events", extremes['low_count'])
        
        # Plot temperature with extremes highlighted
        df_temp = processed_data.set_index('timestamp')['temperature']
        
        fig_extremes = go.Figure()
        
        # Add all temperature points
        fig_extremes.add_trace(go.Scatter(
            x=df_temp.index,
            y=df_temp.values,
            name='Temperature',
            mode='lines',
            line=dict(color='blue', width=1)
        ))
        
        # Add high temperature extremes
        if len(extremes['high_values']) > 0:
            fig_extremes.add_trace(go.Scatter(
                x=extremes['high_values'].index,
                y=extremes['high_values'].values,
                name='High Extremes',
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle')
            ))
        
        # Add low temperature extremes
        if len(extremes['low_values']) > 0:
            fig_extremes.add_trace(go.Scatter(
                x=extremes['low_values'].index,
                y=extremes['low_values'].values,
                name='Low Extremes',
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle')
            ))
        
        # Add threshold lines
        threshold_high = df_temp.mean() + 2 * df_temp.std()
        threshold_low = df_temp.mean() - 2 * df_temp.std()
        
        fig_extremes.add_trace(go.Scatter(
            x=[df_temp.index.min(), df_temp.index.max()],
            y=[threshold_high, threshold_high],
            name='High Threshold',
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        
        fig_extremes.add_trace(go.Scatter(
            x=[df_temp.index.min(), df_temp.index.max()],
            y=[threshold_low, threshold_low],
            name='Low Threshold',
            mode='lines',
            line=dict(color='blue', dash='dash')
        ))
        
        fig_extremes.update_layout(
            height=500,
            title="Temperature Extremes Detection",
            xaxis_title="Time",
            yaxis_title="Temperature (°F)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig_extremes, use_container_width=True)
        
        # Heat and cold waves
        st.subheader("Heat and Cold Waves")
        
        wave_analysis = advanced_stats['wave_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heat Wave Days", wave_analysis['heat_wave_days'])
            st.markdown("""
            *Heat waves are defined as 3+ consecutive days with maximum temperature 
            above the 90th percentile for the period.*
            """)
        with col2:
            st.metric("Cold Wave Days", wave_analysis['cold_wave_days'])
            st.markdown("""
            *Cold waves are defined as 3+ consecutive days with minimum temperature 
            below the 10th percentile for the period.*
            """)
        
        # Diurnal temperature range
        st.subheader("Diurnal Temperature Range")
        
        diurnal_range = advanced_stats['diurnal_range']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Daily Range", f"{diurnal_range['mean']:.1f}°F")
        with col2:
            st.metric("Maximum Daily Range", f"{diurnal_range['max']:.1f}°F")
        
        # Plot diurnal range over time
        if len(diurnal_range['values']) > 0:
            fig_diurnal = go.Figure()
            
            fig_diurnal.add_trace(go.Scatter(
                x=diurnal_range['values'].index,
                y=diurnal_range['values'].values,
                name='Daily Temp Range',
                mode='lines+markers',
                line=dict(color='purple')
            ))
            
            fig_diurnal.update_layout(
                height=350,
                title="Daily Temperature Range (Max - Min)",
                xaxis_title="Date",
                yaxis_title="Temperature Range (°F)"
            )
            
            st.plotly_chart(fig_diurnal, use_container_width=True)
            
            st.markdown("""
            ### Diurnal Temperature Range Significance
            
            A large diurnal temperature range (difference between daily maximum and minimum) 
            can indicate:
            
            - Clear atmospheric conditions (less cloud cover)
            - Low humidity
            - Potential stress on organisms and infrastructure
            
            Variations in this range can signal changing weather patterns or climate shifts.
            """)

if __name__ == "__main__":
    main()