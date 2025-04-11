import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import logging
import io
import os
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utility functions
from utils.weather_api import WeatherAPI
from utils.db_utils import DatabaseManager

# Initialize database connection and session state variables
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
    
if 'db_manager' not in st.session_state:
    try:
        db_manager = DatabaseManager()
        if db_manager.test_connection():
            st.session_state.db_connected = True
            st.session_state.db_manager = db_manager
            logger.info("Database connection initialized successfully")
        else:
            st.session_state.db_connected = False
            logger.warning("Database connection failed")
    except Exception as e:
        st.session_state.db_connected = False
        logger.error(f"Error initializing database: {str(e)}")

# Provide a global reference to the database manager
if st.session_state.db_connected:
    db_manager = st.session_state.db_manager
else:
    # Create a dummy db_manager to avoid unbound variable errors
    db_manager = None

try:
    logger.info("Starting Data Mining Application...")

    st.set_page_config(
        page_title="Data Mining Application",
        page_icon="📊",
        layout="wide"
    )

    # Title and description
    st.title("📊 Data Mining Application")
    st.markdown("""
    This application helps you analyze data using various data mining techniques:
    - Data preprocessing and cleaning
    - Exploratory data analysis
    - Pattern discovery and anomaly detection
    - Clustering analysis
    - Dimensionality reduction
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Add Saved Analyses page if database is connected
    pages = ["Data Input", "Data Exploration", "Clustering Analysis", 
             "Anomaly Detection", "Dimensionality Reduction"]
    
    if st.session_state.db_connected:
        pages.append("Saved Analyses")
        
    page = st.sidebar.radio("Select Page", pages)
    
    # Initialize session state for data storage
    if 'data' not in st.session_state:
        st.session_state.data = None
        
    # Data input page
    if page == "Data Input":
        st.header("Data Input")
        
        input_method = st.radio("Select Input Method", 
                              ["Upload CSV File", "Use Sample Weather Data"])
        
        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.success(f"✅ Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
                    st.subheader("Data Preview")
                    st.dataframe(data.head())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        else:  # Use sample weather data
            days = st.slider("Number of days of weather data", 7, 60, 30)
            
            if st.button("Generate Sample Data"):
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    weather_api = WeatherAPI()
                    data = weather_api.get_historical_weather(start_date, end_date)
                    
                    # Add some calculated columns to make the data more interesting
                    data['heat_index'] = data['temperature'] * (1 + 0.348 * data['humidity'] / 100)
                    data['day_of_week'] = data['timestamp'].dt.dayofweek
                    data['hour_of_day'] = data['timestamp'].dt.hour
                    
                    st.session_state.data = data
                    st.success(f"✅ Successfully generated {len(data)} rows of sample weather data.")
                    st.subheader("Data Preview")
                    st.dataframe(data.head())
                except Exception as e:
                    st.error(f"Error generating sample data: {str(e)}")
        
        # Show data summary if data is loaded
        if st.session_state.data is not None:
            st.subheader("Data Information")
            buffer = io.StringIO()
            st.session_state.data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.subheader("Statistical Summary")
            st.dataframe(st.session_state.data.describe())
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please load or generate data on the Data Input page first.")
        else:
            data = st.session_state.data
            
            # Column selection for visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            st.subheader("Distribution Analysis")
            
            # Histogram
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for histogram", numeric_cols)
                hist_fig = px.histogram(data, x=selected_col, nbins=20, 
                                        title=f"Distribution of {selected_col}")
                st.plotly_chart(hist_fig, use_container_width=True)
            
            # Correlation Analysis
            st.subheader("Correlation Analysis")
            
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr().round(2)
                corr_fig = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    aspect="auto",
                                    color_continuous_scale="RdBu_r")
                corr_fig.update_layout(title="Correlation Matrix")
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Scatter plot matrix for selected variables
                st.subheader("Scatter Plot Matrix")
                selected_cols = st.multiselect("Select columns for scatter plot matrix", 
                                             numeric_cols, 
                                             default=numeric_cols[:min(4, len(numeric_cols))])
                
                if len(selected_cols) > 1:
                    scatter_fig = px.scatter_matrix(data[selected_cols])
                    scatter_fig.update_layout(height=800)
                    st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
    
    # Clustering Analysis page
    elif page == "Clustering Analysis":
        st.header("Clustering Analysis")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please load or generate data on the Data Input page first.")
        else:
            data = st.session_state.data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            st.markdown("""
            K-Means clustering groups data points into a specified number of clusters based on similarity.
            """)
            
            # Feature selection
            selected_features = st.multiselect("Select features for clustering", 
                                             numeric_cols,
                                             default=numeric_cols[:min(3, len(numeric_cols))])
            
            if len(selected_features) >= 2:
                # K-means parameters
                n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    run_btn = st.button("Run K-Means Clustering")
                
                # Add save option if database is connected
                save_analysis = False
                analysis_name = ""
                analysis_desc = ""
                
                if st.session_state.db_connected:
                    with col2:
                        save_analysis = st.checkbox("Save results to database")
                    
                    if save_analysis:
                        col1, col2 = st.columns(2)
                        with col1:
                            analysis_name = st.text_input("Analysis name", "K-Means Clustering")
                        with col2:
                            analysis_desc = st.text_area("Description", "Clustering analysis of weather data", height=100)
                
                if run_btn:
                    # Prepare data
                    X = data[selected_features].copy()
                    
                    # Handle missing values
                    X.fillna(X.mean(), inplace=True)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply K-means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Add cluster labels to the data
                    clustered_data = data.copy()
                    clustered_data['cluster'] = clusters
                    
                    # Display cluster information
                    st.subheader("Cluster Information")
                    cluster_counts = clustered_data['cluster'].value_counts()
                    st.bar_chart(cluster_counts)
                    
                    # Save to database if requested
                    if save_analysis and st.session_state.db_connected:
                        try:
                            # Prepare parameters and results
                            parameters = {
                                "features": selected_features,
                                "n_clusters": n_clusters,
                                "date_range": [str(data['timestamp'].min()), str(data['timestamp'].max())] if 'timestamp' in data.columns else None
                            }
                            
                            # Prepare results (convert to serializable format)
                            results = {
                                "cluster_counts": cluster_counts.to_dict(),
                                "cluster_centers": kmeans.cluster_centers_.tolist(),
                                "inertia": float(kmeans.inertia_)
                            }
                            
                            # Save to database
                            if st.session_state.db_manager.save_analysis(analysis_name, analysis_desc, "kmeans_clustering", parameters, results):
                                st.success("✅ Analysis saved to database successfully!")
                            else:
                                st.error("❌ Failed to save analysis to database")
                        except Exception as e:
                            st.error(f"Error saving analysis: {str(e)}")
                    
                    # Visualize clusters (first two features)
                    if len(selected_features) >= 2:
                        viz_features = selected_features[:2]
                        
                        fig = px.scatter(
                            clustered_data, 
                            x=viz_features[0], 
                            y=viz_features[1],
                            color='cluster',
                            hover_data=selected_features,
                            title=f"Clusters by {viz_features[0]} and {viz_features[1]}"
                        )
                        
                        # Add cluster centers if using only 2 features for visualization
                        if len(selected_features) == 2:
                            centers = scaler.inverse_transform(kmeans.cluster_centers_)
                            for i, center in enumerate(centers):
                                fig.add_trace(
                                    go.Scatter(
                                        x=[center[0]],
                                        y=[center[1]],
                                        mode='markers',
                                        marker=dict(
                                            symbol='x',
                                            size=15,
                                            color='black'
                                        ),
                                        name=f"Center {i}"
                                    )
                                )
                                
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("Cluster Statistics")
                    for i in range(n_clusters):
                        with st.expander(f"Cluster {i} Statistics"):
                            cluster_data = clustered_data[clustered_data['cluster'] == i]
                            st.dataframe(cluster_data[selected_features].describe())
            else:
                st.warning("Please select at least 2 features for clustering.")
    
    # Anomaly Detection page
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please load or generate data on the Data Input page first.")
        else:
            data = st.session_state.data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            st.markdown("""
            Isolation Forest is an effective algorithm for detecting anomalies. It isolates observations by 
            randomly selecting a feature and then randomly selecting a split value between the maximum and 
            minimum values of the selected feature.
            """)
            
            # Feature selection
            selected_features = st.multiselect("Select features for anomaly detection", 
                                             numeric_cols,
                                             default=numeric_cols[:min(4, len(numeric_cols))])
            
            if len(selected_features) >= 1:
                # Isolation Forest parameters
                contamination = st.slider("Contamination (expected proportion of outliers)", 
                                       0.01, 0.3, 0.05, 0.01)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    run_btn = st.button("Detect Anomalies")
                
                # Add save option if database is connected
                save_analysis = False
                analysis_name = ""
                analysis_desc = ""
                
                if st.session_state.db_connected:
                    with col2:
                        save_analysis = st.checkbox("Save anomaly results")
                    
                    if save_analysis:
                        col1, col2 = st.columns(2)
                        with col1:
                            analysis_name = st.text_input("Analysis name", "Anomaly Detection")
                        with col2:
                            analysis_desc = st.text_area("Description", "Isolation Forest anomaly detection on weather data", height=100)
                
                if run_btn:
                    # Prepare data
                    X = data[selected_features].copy()
                    
                    # Handle missing values
                    X.fillna(X.mean(), inplace=True)
                    
                    # Apply Isolation Forest
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomalies = iso_forest.fit_predict(X)
                    
                    # -1 for outliers, 1 for inliers
                    anomaly_data = data.copy()
                    anomaly_data['anomaly'] = anomalies
                    anomaly_data['is_anomaly'] = anomaly_data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
                    
                    # Count anomalies
                    anomaly_count = (anomaly_data['anomaly'] == -1).sum()
                    normal_count = (anomaly_data['anomaly'] == 1).sum()
                    
                    # Save to database if requested
                    if save_analysis and st.session_state.db_connected:
                        try:
                            # Prepare parameters and results
                            parameters = {
                                "features": selected_features,
                                "contamination": contamination,
                                "date_range": [str(data['timestamp'].min()), str(data['timestamp'].max())] if 'timestamp' in data.columns else None
                            }
                            
                            # Get anomaly indices for serialization (iloc locations)
                            anomaly_indices = anomaly_data[anomaly_data['anomaly'] == -1].index.tolist()
                            # Convert to simple integer list if needed
                            if hasattr(anomaly_indices, 'tolist'):
                                anomaly_indices = anomaly_indices.tolist()
                                
                            # Prepare results 
                            results = {
                                "anomaly_count": int(anomaly_count),
                                "normal_count": int(normal_count),
                                "anomaly_percentage": float(anomaly_count/len(data) * 100),
                                "anomaly_indices": anomaly_indices
                            }
                            
                            # Save to database
                            if st.session_state.db_manager.save_analysis(analysis_name, analysis_desc, "anomaly_detection", parameters, results):
                                st.success("✅ Anomaly detection results saved to database!")
                            else:
                                st.error("❌ Failed to save analysis to database")
                        except Exception as e:
                            st.error(f"Error saving analysis: {str(e)}")
                    
                    st.subheader("Anomaly Detection Results")
                    st.write(f"Detected {anomaly_count} anomalies out of {len(data)} data points ({anomaly_count/len(data):.2%})")
                    
                    # Visualize anomalies
                    if len(selected_features) >= 2:
                        viz_features = selected_features[:2]
                        
                        fig = px.scatter(
                            anomaly_data, 
                            x=viz_features[0], 
                            y=viz_features[1],
                            color='is_anomaly',
                            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                            hover_data=selected_features,
                            title=f"Anomaly Detection using {viz_features[0]} and {viz_features[1]}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display anomalous data points
                    st.subheader("Anomalous Data Points")
                    st.dataframe(anomaly_data[anomaly_data['anomaly'] == -1][selected_features])
            else:
                st.warning("Please select at least 1 feature for anomaly detection.")
    
    # Dimensionality Reduction page
    elif page == "Dimensionality Reduction":
        st.header("Dimensionality Reduction with PCA")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please load or generate data on the Data Input page first.")
        else:
            data = st.session_state.data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            st.markdown("""
            Principal Component Analysis (PCA) is a technique to reduce the dimensionality of the data
            while preserving as much of the variance as possible.
            """)
            
            # Feature selection
            selected_features = st.multiselect("Select features for PCA", 
                                             numeric_cols,
                                             default=numeric_cols[:min(5, len(numeric_cols))])
            
            if len(selected_features) >= 2:
                # PCA parameters
                n_components = st.slider("Number of components", 2, min(len(selected_features), 10), 2)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    run_btn = st.button("Run PCA")
                
                # Add save option if database is connected
                save_analysis = False
                analysis_name = ""
                analysis_desc = ""
                
                if st.session_state.db_connected:
                    with col2:
                        save_analysis = st.checkbox("Save PCA results")
                    
                    if save_analysis:
                        col1, col2 = st.columns(2)
                        with col1:
                            analysis_name = st.text_input("Analysis name", "PCA Analysis")
                        with col2:
                            analysis_desc = st.text_area("Description", "Principal Component Analysis on weather data", height=100)
                
                if run_btn:
                    # Prepare data
                    X = data[selected_features].copy()
                    
                    # Handle missing values
                    X.fillna(X.mean(), inplace=True)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply PCA
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(X_scaled)
                    
                    # Create a DataFrame with PCA results
                    pca_df = pd.DataFrame(
                        data=pca_result,
                        columns=[f'PC{i+1}' for i in range(n_components)]
                    )
                    
                    # Save to database if requested
                    if save_analysis and st.session_state.db_connected:
                        try:
                            # Prepare parameters and results
                            parameters = {
                                "features": selected_features,
                                "n_components": n_components,
                                "date_range": [str(data['timestamp'].min()), str(data['timestamp'].max())] if 'timestamp' in data.columns else None
                            }
                            
                            # Get variance data
                            explained_variance = pca.explained_variance_ratio_.tolist()
                            cumulative_variance = np.cumsum(explained_variance).tolist()
                            
                            # Get loadings data
                            loadings = pca.components_.T.tolist()
                            
                            # Prepare results
                            results = {
                                "explained_variance": explained_variance,
                                "cumulative_variance": cumulative_variance,
                                "loadings": loadings,
                                "feature_names": selected_features
                            }
                            
                            # Save to database
                            if st.session_state.db_manager.save_analysis(analysis_name, analysis_desc, "pca_analysis", parameters, results):
                                st.success("✅ PCA results saved to database!")
                            else:
                                st.error("❌ Failed to save analysis to database")
                        except Exception as e:
                            st.error(f"Error saving analysis: {str(e)}")
                    
                    # Display explained variance
                    st.subheader("Explained Variance Ratio")
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    variance_df = pd.DataFrame({
                        'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                        'Explained Variance Ratio': explained_variance,
                        'Cumulative Explained Variance': cumulative_variance
                    })
                    
                    st.dataframe(variance_df)
                    
                    # Visualize explained variance
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(n_components)],
                        y=explained_variance,
                        name='Individual'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[f'PC{i+1}' for i in range(n_components)],
                        y=cumulative_variance,
                        name='Cumulative',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Explained Variance by Principal Component',
                        xaxis_title='Principal Component',
                        yaxis_title='Explained Variance Ratio',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualize first two principal components
                    if n_components >= 2:
                        st.subheader("PCA Visualization")
                        
                        # Get the loadings (feature importance for each PC)
                        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                        
                        # Create a biplot
                        biplot_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
                        
                        fig = px.scatter(
                            biplot_df,
                            x='PC1',
                            y='PC2',
                            title='PCA: First two principal components'
                        )
                        
                        # Add feature loadings as vectors
                        scale_factor = 5  # Adjust as needed for visibility
                        for i, feature in enumerate(selected_features):
                            fig.add_annotation(
                                x=loadings[i, 0] * scale_factor,
                                y=loadings[i, 1] * scale_factor,
                                ax=0,
                                ay=0,
                                text=feature,
                                showarrow=True,
                                arrowhead=2
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display principal component weights
                        st.subheader("Feature Contributions to Principal Components")
                        
                        loadings_df = pd.DataFrame(
                            data=pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=selected_features
                        )
                        
                        st.dataframe(loadings_df.round(3))
            else:
                st.warning("Please select at least 2 features for PCA.")
                
    # Saved Analyses page
    elif page == "Saved Analyses":
        st.header("Saved Analyses")
        
        if not st.session_state.db_connected:
            st.error("⚠️ Database connection is not available. Cannot access saved analyses.")
        else:
            try:
                # Get all saved analyses from the database
                analyses_df = st.session_state.db_manager.get_saved_analyses()
                
                if analyses_df.empty:
                    st.info("No saved analyses found in the database. Perform an analysis and save it first.")
                else:
                    # Format the dataframe for display
                    display_df = analyses_df[['name', 'analysis_type', 'description', 'created_at']].copy()
                    
                    # Format the datetime for better display
                    display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    # Rename columns for better display
                    display_df.columns = ['Name', 'Type', 'Description', 'Created At']
                    
                    # Display the analyses
                    st.subheader("Available Analyses")
                    st.dataframe(display_df)
                    
                    # Allow user to select an analysis to view details
                    analysis_types = analyses_df['analysis_type'].unique().tolist()
                    
                    # Filter by type
                    selected_type = st.selectbox("Filter by analysis type", 
                                               ["All"] + analysis_types)
                    
                    filtered_analyses = analyses_df
                    if selected_type != "All":
                        filtered_analyses = analyses_df[analyses_df['analysis_type'] == selected_type]
                    
                    # Select an analysis to view
                    analysis_names = filtered_analyses['name'].tolist()
                    if analysis_names:
                        selected_analysis = st.selectbox("Select an analysis to view details", 
                                                     analysis_names)
                        
                        # Get the selected analysis
                        analysis = filtered_analyses[filtered_analyses['name'] == selected_analysis].iloc[0]
                        
                        # Display analysis details
                        st.subheader(f"Analysis Details: {analysis['name']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Type:** {analysis['analysis_type']}")
                            st.markdown(f"**Created:** {pd.to_datetime(analysis['created_at']).strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            st.markdown(f"**Description:** {analysis['description']}")
                        
                        # Parse parameters and results
                        parameters = json.loads(analysis['parameters']) if isinstance(analysis['parameters'], str) else analysis['parameters']
                        results = json.loads(analysis['results']) if isinstance(analysis['results'], str) else analysis['results']
                        
                        # Display parameters
                        st.subheader("Parameters")
                        for param, value in parameters.items():
                            if param == "features" and isinstance(value, list):
                                st.markdown(f"**{param}:** {', '.join(value)}")
                            else:
                                st.markdown(f"**{param}:** {value}")
                        
                        # Display visualizations based on analysis type
                        st.subheader("Results")
                        
                        if analysis['analysis_type'] == "kmeans_clustering":
                            # Display cluster counts
                            if "cluster_counts" in results:
                                cluster_counts = pd.Series(results["cluster_counts"]).sort_index()
                                st.bar_chart(cluster_counts)
                                
                            # Display cluster centers if available
                            if "cluster_centers" in results and "features" in parameters:
                                st.markdown("**Cluster Centers:**")
                                centers_df = pd.DataFrame(
                                    results["cluster_centers"], 
                                    columns=parameters["features"]
                                )
                                st.dataframe(centers_df)
                                
                            if "inertia" in results:
                                st.markdown(f"**Inertia (sum of squared distances):** {results['inertia']:.2f}")
                                
                        elif analysis['analysis_type'] == "anomaly_detection":
                            # Display anomaly statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Anomalies", results.get("anomaly_count", 0))
                            
                            with col2:
                                st.metric("Normal Points", results.get("normal_count", 0))
                                
                            with col3:
                                st.metric("Anomaly Percentage", f"{results.get('anomaly_percentage', 0):.2f}%")
                                
                        elif analysis['analysis_type'] == "pca_analysis":
                            # Display explained variance
                            if "explained_variance" in results and "cumulative_variance" in results:
                                # Create dataframe for variance display
                                variance_df = pd.DataFrame({
                                    'Principal Component': [f'PC{i+1}' for i in range(len(results["explained_variance"]))],
                                    'Explained Variance Ratio': results["explained_variance"],
                                    'Cumulative Explained Variance': results["cumulative_variance"]
                                })
                                
                                st.dataframe(variance_df)
                                
                                # Plot variance chart
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=[f'PC{i+1}' for i in range(len(results["explained_variance"]))],
                                    y=results["explained_variance"],
                                    name='Individual'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=[f'PC{i+1}' for i in range(len(results["explained_variance"]))],
                                    y=results["cumulative_variance"],
                                    name='Cumulative',
                                    line=dict(color='red')
                                ))
                                
                                fig.update_layout(
                                    title='Explained Variance by Principal Component',
                                    xaxis_title='Principal Component',
                                    yaxis_title='Explained Variance Ratio',
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display loadings if available
                                if "loadings" in results and "feature_names" in results:
                                    st.subheader("Feature Contributions to Principal Components")
                                    
                                    # Create loadings dataframe
                                    loadings_df = pd.DataFrame(
                                        data=results["loadings"],
                                        columns=[f'PC{i+1}' for i in range(len(results["loadings"][0]))],
                                        index=results["feature_names"]
                                    )
                                    
                                    st.dataframe(loadings_df.round(3))
                                
                        # Generic JSON view for any other type or for detailed exploration
                        with st.expander("View Raw Results"):
                            st.json(results)
                    else:
                        st.info(f"No analyses of type '{selected_type}' found.")
            except Exception as e:
                st.error(f"Error retrieving saved analyses: {str(e)}")
                logger.error(f"Error retrieving saved analyses: {str(e)}")

    logger.info("Application running successfully")

except Exception as e:
    logger.error(f"Error running application: {str(e)}")
    st.error(f"An error occurred: {str(e)}")