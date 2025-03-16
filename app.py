import streamlit as st
st.set_page_config(
    page_title="Dynamic AI Data Visualization Agent", 
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['PYTORCH_JIT'] = '0'
os.environ['STREAMLIT_WATCH_EXCLUDE_PATTERNS'] = 'torch.*'

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, AffinityPropagation, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

def load_data(file):
    """Load CSV data from an uploaded file."""
    try:
        df = pd.read_csv(file)
        st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Perform basic data quality checks
        df_info = {
            "Total Rows": df.shape[0],
            "Total Columns": df.shape[1],
            "Missing Values": df.isnull().sum().sum(),
            "Duplicate Rows": df.duplicated().sum()
        }
        
        with st.expander("📊 Data Overview"):
            st.write(df_info)
            st.write("Sample Data:")
            st.dataframe(df.head())
            
            st.write("Column Data Types:")
            st.write(df.dtypes)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def data_summary(df):
    """Display basic info, missing values and summary statistics."""
    st.write("### Basic Information")
    st.write(df.info())
    st.write("### Missing Values per Column")
    st.write(df.isnull().sum())
    st.write("### Summary Statistics")
    st.write(df.describe())

def clean_data(df):
    """Perform basic cleaning: remove duplicate rows and rows where all entries are missing."""
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.dropna(how='all')
    return df_cleaned

# Update the impute_data function
def impute_data(df, strategy, knn_k=5):
    """Impute missing values using the selected strategy."""
    df_imputed = df.copy()
    
    if strategy == 'Most Frequent':
        # Handle both numeric and categorical columns
        for col in df.columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df_imputed[col] = df[col].fillna(mode_value)
    else:
        # Handle numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if strategy == 'Mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'Median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'KNN':
                imputer = KNNImputer(n_neighbors=knn_k)
            else:
                st.error("Invalid imputation strategy!")
                return df
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_imputed

def scale_data(df, method='Standard'):
    """Scale numeric data using Standard or MinMax scaling."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    if len(numeric_cols) > 0:
        if method == 'Standard':
            scaler = StandardScaler()
        elif method == 'MinMax':
            scaler = MinMaxScaler()
        else:
            st.error("Invalid scaling method!")
            return df
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled

def encode_categorical_data(df, method='OneHot'):
    """Encode categorical columns using OneHot or Label encoding."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = df.copy()
    
    if len(categorical_cols) == 0:
        st.warning("No categorical columns found for encoding.")
        return df_encoded
    
    if method == 'OneHot':
        for col in categorical_cols:
            # Create a one-hot encoder for the column
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cols = encoder.fit_transform(df_encoded[[col]])
            
            # Create column names for the encoded features
            encoded_col_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            
            # Add encoded columns to the dataframe
            encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            # Drop the original column
            df_encoded = df_encoded.drop(col, axis=1)
    
    elif method == 'Label':
        for col in categorical_cols:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
    
    else:
        st.error("Invalid encoding method!")
        return df
    
    return df_encoded

# Add this function before apply_model
def prepare_data_for_lightgbm(X, y):
    """Prepare data specifically for LightGBM to avoid warnings."""
    # Convert data to float32 to reduce memory usage and improve speed
    X = X.astype('float32')
    if y is not None:
        y = y.astype('float32')
    
    # Handle infinite values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, y

def apply_model(df, model_type, target_column=None, n_clusters=3, n_components=2, **kwargs):
    """Apply selected model to the data."""
    try:
        if target_column and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        if target_column:
            # Prepare data for supervised learning
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # Determine if regression or classification based on target variable
            is_regression = True
            if pd.api.types.is_numeric_dtype(y):
                unique_values = len(np.unique(y))
                is_regression = unique_values > 10
            else:
                is_regression = False
            
            # Convert categorical columns to numeric
            for col in X.select_dtypes(include=['object']):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            
            # Handle classification target variable - ensure labels start from 0
            if not is_regression:
                le = LabelEncoder()
                # Convert to string first to handle numeric classes
                y = y.astype(str)
                y = le.fit_transform(y)
                label_encoder = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model based on type
            if is_regression:
                if "Classifier" in model_type:
                    raise ValueError(f"Cannot use classifier {model_type} for regression task")
                model = initialize_regression_model(model_type, X_train)
            else:
                if "Regressor" in model_type:
                    raise ValueError(f"Cannot use regressor {model_type} for classification task")
                model = initialize_classification_model(model_type)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results = {
                'model': model,
                'metrics': {},
                'data': df,
                'feature_importance': None,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'label_encoder': label_encoder if not is_regression else None
            }
            
            if is_regression:
                results['metrics']['mse'] = mean_squared_error(y_test, y_pred)
                results['metrics']['r2'] = r2_score(y_test, y_pred)
            else:
                results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
                # Convert predictions back to original labels for classification report
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_original = label_encoder.inverse_transform(y_pred)
                results['metrics']['classification_report'] = classification_report(
                    y_test_original, 
                    y_pred_original
                )
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
            
            return results
            
    except Exception as e:
        st.error(f"Error in model application: {str(e)}")
        return None

def create_scatter_plot(df, x_col, y_col, color_by=None, size_by=None, title=None):
    """Generate a 2D scatter plot using selected numeric columns."""
    if title is None:
        title = f"Scatter Plot: {x_col} vs {y_col}"
    
    if color_by and color_by in df.columns:
        if size_by and size_by in df.columns:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_by, size=size_by,
                title=title, template="plotly_dark"
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_by,
                title=title, template="plotly_dark"
            )
    else:
        if size_by and size_by in df.columns:
            fig = px.scatter(
                df, x=x_col, y=y_col, size=size_by,
                title=title, template="plotly_dark"
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=title, template="plotly_dark"
            )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_scatter_plot_3d(df, x_col, y_col, z_col, color_by=None, title=None):
    """Generate a 3D scatter plot using selected numeric columns."""
    if title is None:
        title = f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}"
    
    if color_by and color_by in df.columns:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col, color=color_by,
            title=title, template="plotly_dark"
        )
    else:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
            zaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
        ),
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif"))
    )
    return fig

def create_histogram(df, column, bins=None, color=None, title=None):
    """Generate a histogram for a selected numeric column."""
    if title is None:
        title = f"Histogram of {column}"
    
    fig = px.histogram(
        df, x=column, nbins=bins, color=color,
        title=title, template="plotly_dark"
    )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_box_plot(df, column, group_by=None, title=None):
    """Generate a box plot for a selected numeric column."""
    if title is None:
        title = f"Box Plot of {column}"
    
    if group_by and group_by in df.columns:
        fig = px.box(
            df, y=column, x=group_by,
            title=title, template="plotly_dark"
        )
    else:
        fig = px.box(
            df, y=column,
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_bar_plot(df, x_col, y_col, color_by=None, title=None):
    """Generate a bar plot using selected columns."""
    if title is None:
        title = f"Bar Plot: {x_col} vs {y_col}"
    
    if color_by and color_by in df.columns:
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_by,
            title=title, template="plotly_dark"
        )
    else:
        fig = px.bar(
            df, x=x_col, y=y_col,
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, .2)')
    )
    return fig

def create_pie_chart(df, names_col, values_col=None, title=None):
    """Generate a pie chart using a categorical column and optional values column."""
    if title is None:
        if values_col:
            title = f"Pie Chart: {names_col} by {values_col}"
        else:
            title = f"Pie Chart: {names_col} (Counts)"
    
    if values_col:
        fig = px.pie(
            df, names=names_col, values=values_col,
            title=title, template="plotly_dark"
        )
    else:
        # If no values column provided, compute counts
        pie_data = df[names_col].value_counts().reset_index()
        pie_data.columns = [names_col, 'count']
        fig = px.pie(
            pie_data, names=names_col, values='count',
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        paper_bgcolor='rgba(20, 24, 35, 0.8)'
    )
    return fig

def create_violin_plot(df, x_col, y_col, color_by=None, title=None):
    """Generate a violin plot using selected columns."""
    if title is None:
        title = f"Violin Plot: {x_col} vs {y_col}"
    
    if color_by and color_by in df.columns:
        fig = px.violin(
            df, x=x_col, y=y_col, color=color_by, box=True,
            title=title, template="plotly_dark"
        )
    else:
        fig = px.violin(
            df, x=x_col, y=y_col, box=True,
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_line_chart(df, x_col, y_col, color_by=None, title=None):
    """Generate a line chart using selected columns."""
    if title is None:
        title = f"Line Chart: {x_col} vs {y_col}"
    
    if color_by and color_by in df.columns:
        fig = px.line(
            df, x=x_col, y=y_col, color=color_by,
            title=title, template="plotly_dark"
        )
    else:
        fig = px.line(
            df, x=x_col, y=y_col,
            title=title, template="plotly_dark"
        )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_density_plot(df, x_col, y_col, title=None):
    """Generate a density heatmap (density plot) for two selected numeric columns."""
    if title is None:
        title = f"Density Plot: {x_col} vs {y_col}"
    
    fig = px.density_heatmap(
        df, x=x_col, y=y_col,
        title=title, template="plotly_dark"
    )
    
    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(20, 24, 35, 0.8)',
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        xaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
    )
    return fig

def create_heatmap(df, title=None):
    """Generate a correlation heatmap for numeric data."""
    if title is None:
        title = "Correlation Heatmap"
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation heatmap")
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto='.2f',  # Show 2 decimal places
        aspect="auto",
        title=title,
        template="plotly_dark",
        color_continuous_scale="RdBu_r"  # Better color scale for correlations
    )
    
    # Improve layout
    fig.update_layout(
        font=dict(family="Roboto, sans-serif", color="white"),
        title=dict(font=dict(size=24, family="Poppins, sans-serif")),
        paper_bgcolor='rgba(20, 24, 35, 0.8)',
        xaxis_title="Features",
        yaxis_title="Features",
        width=800,
        height=800
    )
    
    # Update traces for better readability
    fig.update_traces(
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Correlation",
                side="right"
            ),
            thickness=20,
            len=0.8
        )
    )
    
    return fig

def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for an object (DataFrame or string)."""
    if isinstance(object_to_download, pd.DataFrame):
        # Convert DataFrame before creating CSV
        df_to_download = convert_df_for_streamlit(object_to_download)
        object_to_download = df_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def download_plotly_fig(fig, filename="plot.png"):
    """Generate a download link for a plotly figure."""
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Image</a>'
    return href

def make_prediction(model, input_data, feature_columns):
    """Make predictions using the trained model."""
    try:
        # Convert input data to correct format
        input_array = np.array([float(input_data[col]) for col in feature_columns]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Get prediction probability if available (for classifiers)
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_array)
        
        return prediction[0], prob[0] if prob is not None else None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def preprocess_input_data(input_df, handle_missing=False, missing_strategy=None,
                         scale_features=False, scaler_type=None,
                         encode_categorical=False, encoding_method=None):
    """Preprocess input data for prediction."""
    df_processed = input_df.copy()
    
    # Handle missing values
    if handle_missing and missing_strategy:
        imputer = SimpleImputer(strategy=missing_strategy.lower())
        df_processed = pd.DataFrame(
            imputer.fit_transform(df_processed),
            columns=df_processed.columns
        )
    
    # Scale features
    if scale_features and scaler_type:
        scaler = StandardScaler() if scaler_type == "Standard" else MinMaxScaler()
        df_processed = pd.DataFrame(
            scaler.fit_transform(df_processed),
            columns=df_processed.columns
        )
    
    # Encode categorical variables
    if encode_categorical and encoding_method:
        for col in df_processed.select_dtypes(include=['object']):
            if encoding_method == "Label":
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
            elif encoding_method == "One-Hot":
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df_processed[[col]])
                encoded_cols = [f"{col}_{i}" for i in range(encoded.shape[1])]
                df_processed = pd.concat([
                    df_processed.drop(col, axis=1),
                    pd.DataFrame(encoded, columns=encoded_cols, index=df_processed.index)
                ], axis=1)
    
    return df_processed

# Add this helper function at the top of the file with other helper functions
def convert_df_for_streamlit(df):
    """Convert DataFrame to make it compatible with Streamlit's display functions."""
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_converted = df.copy()
        
        # Convert object columns to string
        for col in df_converted.select_dtypes(['object']).columns:
            df_converted[col] = df_converted[col].astype(str)
        
        # Convert Int64 columns to regular int64
        for col in df_converted.select_dtypes(['Int64']).columns:
            df_converted[col] = df_converted[col].astype('int64')
        
        # Convert nullable integer columns to float
        for col in df_converted.select_dtypes(['Int64', 'Float64']).columns:
            df_converted[col] = df_converted[col].astype('float64')
        
        # Convert any problematic numeric types
        for col in df_converted.select_dtypes(['float64', 'int64']).columns:
            if df_converted[col].isnull().any():
                df_converted[col] = df_converted[col].astype('float64')
            else:
                df_converted[col] = df_converted[col].astype('float64')
        
        # Handle any remaining problematic columns
        for col in df_converted.columns:
            if df_converted[col].dtype.name not in ['float64', 'int64', 'bool', 'str']:
                df_converted[col] = df_converted[col].astype(str)
        
        return df_converted
        
    except Exception as e:
        st.error(f"Error converting DataFrame: {str(e)}")
        # Return original DataFrame if conversion fails
        return df

def safe_display_dataframe(df, container):
    """Safely display DataFrame in Streamlit with error handling."""
    try:
        display_df = convert_df_for_streamlit(df)
        container.dataframe(display_df)
    except Exception as e:
        container.error(f"Error displaying data: {str(e)}")
        container.write("Showing raw data instead:")
        container.write(df)

def remove_columns(df, columns_to_remove):
    """Remove selected columns from the DataFrame permanently."""
    try:
        return df.drop(columns=columns_to_remove, inplace=False)
    except Exception as e:
        st.error(f"Error removing columns: {str(e)}")
        return df

# --- Custom Styles ---
def set_custom_theme():
    """Set custom theme for the app"""
    custom_css = """
    <style>
        .stApp {
            background-color: #121826;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #5663F7;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4051E8;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(86, 99, 247, 0.3);
        }
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #FFFFFF;
        }
        .stSidebar {
            background-color: #1A2333;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stFileUploader"] {
            background-color: #1A2333;
            border-radius: 10px;
            padding: 10px;
        }
        [data-testid="stExpander"] {
            background-color: #1A2333;
            border-radius: 10px;
        }
        .stCheckbox label {
            color: #FFFFFF;
            font-family: 'Roboto', sans-serif;
        }
        .stSelectbox label {
            color: #FFFFFF;
            font-family: 'Roboto', sans-serif;
        }
        .stRadio label {
            color: #FFFFFF;
            font-family: 'Roboto', sans-serif;
        }
        .css-145kmo2 {
            font-family: 'Roboto', sans-serif;
        }
        .success {
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(47, 158, 68, 0.2);
            border-left: 5px solid #2F9E44;
        }
        .info {
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(49, 130, 206, 0.2);
            border-left: 5px solid #3182CE;
        }
        .warning {
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(236, 153, 53, 0.2);
            border-left: 5px solid #EC9935;
        }
        .error {
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(229, 62, 62, 0.2);
            border-left: 5px solid #E53E3E;
        }
        .prediction-container {
            background-color: #1A2333;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            color: #5663F7;
            text-align: center;
            padding: 20px;
            background-color: rgba(86, 99, 247, 0.1);
            border-radius: 10px;
            margin-top: 10px;
        }
    </style>
    """
    # Add Google fonts
    fonts = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    """
    st.markdown(fonts + custom_css, unsafe_allow_html=True)

# --- Streamlit Application ---

def main():
    # Initialize session state only once
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.trained_models = {}
        st.session_state.processed_df = None
        st.session_state.current_fig = None
    
    # Set custom theme
    set_custom_theme()
    
    # Initialize session state for data persistence
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    # App header with custom styling
    st.markdown("<h1 style='text-align: center; color: #5663F7; margin-bottom: 0;'>Dynamic AI Data Visualization Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Upload your dataset and create beautiful, interactive visualizations with just a few clicks</p>", unsafe_allow_html=True)
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 📊 Upload & Process")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    # Global variable to hold the visualization figure
    current_fig = None
    
    if uploaded_file is not None:
        # Load data only if it hasn't been processed yet or if it's a new file
        if st.session_state.processed_df is None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.processed_df = df
        df = st.session_state.processed_df
        
        if df is not None:
            with col1:
                st.markdown("### 🔧 Data Processing Options")
                
                # Add Column Removal Box
                with st.expander("❌ Remove Columns", expanded=False):
                    columns_to_remove = st.multiselect(
                        "Select columns to remove",
                        df.columns,
                        help="Selected columns will be permanently removed from the dataset"
                    )
                    if st.button("Remove Selected Columns"):
                        if columns_to_remove:
                            df = remove_columns(df, columns_to_remove)
                            st.session_state.processed_df = df
                            st.success(f"Removed {len(columns_to_remove)} columns")
                        else:
                            st.warning("Please select at least one column to remove")

                # Data Cleaning Box
                with st.expander("🧹 Data Cleaning", expanded=False):
                    if st.checkbox("Enable data cleaning"):
                        st.info("This will remove duplicates and rows with all missing values")
                        if st.button("Clean Data"):
                            with st.spinner("Cleaning data..."):
                                df = clean_data(df)
                                st.session_state.processed_df = df
                                st.success("Data cleaned successfully!")

                # Missing Values Box
                with st.expander("🔍 Handle Missing Values", expanded=False):
                    if st.checkbox("Enable missing value imputation"):
                        strategy = st.selectbox(
                            "Choose imputation method:",
                            ["Mean", "Median", "Most Frequent", "KNN"],
                            help="Select how you want to fill missing values"
                        )
                        
                        # Separate numeric and categorical columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        # Show appropriate columns based on strategy
                        if strategy in ["Mean", "Median", "KNN"]:
                            cols_to_impute = st.multiselect(
                                "Select numeric columns to impute",
                                numeric_cols,
                                default=numeric_cols,
                                help="Only numeric columns can be imputed with Mean/Median/KNN"
                            )
                        else:  # Most Frequent strategy
                            all_cols = numeric_cols + categorical_cols
                            cols_to_impute = st.multiselect(
                                "Select columns to impute",
                                all_cols,
                                default=all_cols,
                                help="Both numeric and categorical columns can be imputed with Most Frequent strategy"
                            )
                        
                        if strategy == "KNN":
                            k = st.slider("Number of neighbors (k)", 1, 10, 5)
                        
                        if st.button("Apply Imputation"):
                            if cols_to_impute:
                                with st.spinner(f"Imputing missing values using {strategy}..."):
                                    if strategy == "Most Frequent":
                                        # Handle both numeric and categorical columns
                                        for col in cols_to_impute:
                                            mode_value = df[col].mode()[0]
                                            df[col] = df[col].fillna(mode_value)
                                        st.success("✅ Missing values filled successfully!")
                                    else:
                                        # Only numeric columns for Mean/Median/KNN
                                        df_numeric = df[cols_to_impute]
                                        df = impute_data(df, strategy, knn_k=k if strategy == "KNN" else None)
                                        st.session_state.processed_df = df
                                        st.success("✅ Missing values filled successfully!")
                            else:
                                st.warning("Please select at least one column to impute")

                # Data Scaling Box
                with st.expander("📏 Scale Features", expanded=False):
                    if st.checkbox("Enable data scaling"):
                        scale_method = st.selectbox(
                            "Choose scaling method:",
                            ["Standard Scaling", "MinMax Scaling"],
                            help="StandardScaler normalizes data to mean=0, std=1. MinMaxScaler scales to a fixed range [0,1]"
                        )
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        cols_to_scale = st.multiselect(
                            "Select columns to scale",
                            numeric_cols,
                            default=numeric_cols
                        )
                        
                        if st.button("Apply Scaling"):
                            if cols_to_scale:
                                with st.spinner(f"Applying {scale_method}..."):
                                    df = scale_data(df, scale_method.split()[0])
                                    st.session_state.processed_df = df
                                    st.success("✅ Data scaled successfully!")
                            else:
                                st.warning("Please select at least one column to scale")

                # Encoding Box
                with st.expander("🏷️ Encode Categorical Data", expanded=False):
                    if st.checkbox("Enable categorical encoding"):
                        encode_method = st.selectbox(
                            "Choose encoding method:",
                            ["One-Hot Encoding", "Label Encoding"],
                            help="One-Hot creates binary columns. Label assigns numeric labels."
                        )
                        
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        cols_to_encode = st.multiselect(
                            "Select columns to encode",
                            categorical_cols,
                            default=categorical_cols
                        )
                        
                        if st.button("Apply Encoding"):
                            if cols_to_encode:
                                with st.spinner(f"Applying {encode_method}..."):
                                    df = encode_categorical_data(df, encode_method.split()[0])
                                    st.session_state.processed_df = df
                                    st.success("✅ Categorical data encoded successfully!")
                            else:
                                st.warning("Please select at least one column to encode")

                # Model Selection Box
                with st.expander("🤖 Apply Models", expanded=False):
                    if st.checkbox("Enable model application"):
                        model_category = st.selectbox(
                            "Select model type:",
                            ["Dimensionality Reduction", "Clustering", "Regression", "Classification"],  # Removed "Data Prediction"
                            help="Choose the type of model to apply"
                        )
                        
                        # Dynamic model options based on category
                        if model_category == "Dimensionality Reduction":
                            st.write("#### Dimensionality Reduction Settings")
    
                            dim_model = st.selectbox(
                                "Choose dimensionality reduction method:",
                                ["PCA", "t-SNE", "UMAP"]
                            )
                            
                            # Get numeric columns only
                            numeric_df = df.select_dtypes(include=[np.number])
                            if numeric_df.empty:
                                st.error("No numeric columns available for dimensionality reduction!")
                            else:
                                n_components = st.slider(
                                    "Number of components", 
                                    min_value=2, 
                                    max_value=min(5, numeric_df.shape[1]), 
                                    value=2
                                )
                                
                                # Model-specific parameters
                                params = {}
                                if dim_model == "t-SNE":
                                    params['perplexity'] = st.slider("Perplexity", 5, 50, 30)
                                    params['n_iter'] = st.slider("Number of iterations", 250, 1000, 500)
                                elif dim_model == "UMAP":
                                    params['n_neighbors'] = st.slider("Number of neighbors", 2, 100, 15)
                                    params['min_dist'] = st.slider("Minimum distance", 0.0, 1.0, 0.1)
                                
                                if st.button(f"Apply {dim_model}"):
                                    with st.spinner(f"Applying {dim_model}..."):
                                        try:
                                            # Apply dimensionality reduction
                                            if dim_model == "PCA":
                                                model = PCA(n_components=n_components)
                                                transformed_data = model.fit_transform(numeric_df)
                                                explained_var = model.explained_variance_ratio_
                                                
                                                # Create results dictionary
                                                results = {
                                                    'data': pd.DataFrame(
                                                        transformed_data,
                                                        columns=[f'Component_{i+1}' for i in range(n_components)]
                                                    ),
                                                    'explained_variance': explained_var,
                                                    'model': model
                                                }
                                            
                                            elif dim_model == "t-SNE":
                                                model = TSNE(
                                                    n_components=n_components,
                                                    perplexity=params['perplexity'],
                                                    n_iter=params['n_iter'],
                                                    random_state=42
                                                )
                                                transformed_data = model.fit_transform(numeric_df)
                                                
                                                results = {
                                                    'data': pd.DataFrame(
                                                        transformed_data,
                                                        columns=[f'Component_{i+1}' for i in range(n_components)]
                                                    ),
                                                    'model': model
                                                }
                                            
                                            elif dim_model == "UMAP":
                                                reducer = umap.UMAP(
                                                    n_components=n_components,
                                                    n_neighbors=params['n_neighbors'],
                                                    min_dist=params['min_dist'],
                                                    random_state=42
                                                )
                                                transformed_data = reducer.fit_transform(numeric_df)
                                                
                                                results = {
                                                    'data': pd.DataFrame(
                                                        transformed_data,
                                                        columns=[f'Component_{i+1}' for i in range(n_components)]
                                                    ),
                                                    'model': reducer
                                                }
                                            
                                            # Display results
                                            st.success(f"{dim_model} applied successfully!")
                                            show_dim_reduction_results(dim_model, results, n_components)
                                            
                                            # Add download button
                                            transformed_df = results['data'].copy()
                                            if st.button("Download Transformed Data"):
                                                tmp_download_link = download_link(
                                                    transformed_df,
                                                    f"{dim_model.lower()}_transformed_data.csv",
                                                    "Click here to download"
                                                )
                                                st.markdown(tmp_download_link, unsafe_allow_html=True)
                                        
                                        except Exception as e:
                                            st.error(f"Error applying {dim_model}: {str(e)}")

                        # Update the clustering section in the model selection box
                        elif model_category == "Clustering":
                            cluster_model = st.selectbox(
                                "Choose clustering method:",
                                ["K-means", "DBSCAN", "Gaussian Mixture", "BIRCH", 
                                 "Affinity Propagation", "Mean-Shift", "OPTICS", 
                                 "Agglomerative Clustering"]
                            )
                            
                            # Parameters for each clustering algorithm
                            if cluster_model == "K-means":
                                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                                params = {"n_clusters": n_clusters}
                            
                            elif cluster_model == "DBSCAN":
                                eps = st.slider("Epsilon (neighborhood distance)", 0.1, 2.0, 0.5)
                                min_samples = st.slider("Minimum samples in neighborhood", 2, 10, 5)
                                params = {"eps": eps, "min_samples": min_samples}
                            
                            elif cluster_model == "Gaussian Mixture":
                                n_components = st.slider("Number of components", 2, 10, 3)
                                covariance_type = st.selectbox("Covariance type", 
                                                             ["full", "tied", "diag", "spherical"])
                                params = {"n_components": n_components, 
                                         "covariance_type": covariance_type}
                            
                            elif cluster_model == "BIRCH":
                                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                                threshold = st.slider("Branching factor", 0.1, 2.0, 0.5)
                                params = {"n_clusters": n_clusters, "threshold": threshold}
                            
                            elif cluster_model == "Affinity Propagation":
                                damping = st.slider("Damping factor", 0.5, 1.0, 0.5, step=0.1)
                                max_iter = st.slider("Maximum iterations", 100, 1000, 200)
                                params = {"damping": damping, "max_iter": max_iter}
                            
                            elif cluster_model == "Mean-Shift":
                                bandwidth = st.slider("Bandwidth", 0.1, 5.0, 2.0)
                                params = {"bandwidth": bandwidth}
                            
                            elif cluster_model == "OPTICS":
                                min_samples = st.slider("Minimum samples", 2, 10, 5)
                                max_eps = st.slider("Maximum epsilon", 0.1, 5.0, 2.0)
                                params = {"min_samples": min_samples, "max_eps": max_eps}
                            
                            elif cluster_model == "Agglomerative Clustering":
                                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                                linkage = st.selectbox("Linkage criterion", 
                                                     ["ward", "complete", "average", "single"])
                                params = {"n_clusters": n_clusters, "linkage": linkage}

                            if st.button(f"Apply {cluster_model}"):
                                with st.spinner(f"Applying {cluster_model}..."):
                                    # Update the model creation based on selected algorithm
                                    if cluster_model == "K-means":
                                        model = KMeans(**params)
                                    elif cluster_model == "DBSCAN":
                                        model = DBSCAN(**params)
                                    elif cluster_model == "Gaussian Mixture":
                                        model = GaussianMixture(**params)
                                    elif cluster_model == "BIRCH":
                                        model = Birch(**params)
                                    elif cluster_model == "Affinity Propagation":
                                        model = AffinityPropagation(**params)
                                    elif cluster_model == "Mean-Shift":
                                        model = MeanShift(**params)
                                    elif cluster_model == "OPTICS":
                                        model = OPTICS(**params)
                                    elif cluster_model == "Agglomerative Clustering":
                                        model = AgglomerativeClustering(**params)

                                    # Apply clustering
                                    numeric_data = df.select_dtypes(include=[np.number])
                                    clusters = model.fit_predict(numeric_data)
                                    
                                    results = {
                                        'model': model,
                                        'data': df.assign(Cluster=clusters),
                                        'metrics': {
                                            'n_clusters': len(np.unique(clusters)),
                                            'params': params
                                        }
                                    }
                                    
                                    if hasattr(model, 'inertia_'):
                                        results['metrics']['inertia'] = model.inertia_
                                        
                                    show_clustering_results(results, cluster_model)

                        elif model_category in ["Regression", "Classification"]:
                            # Define available models
                            if model_category == "Regression":
                                models = {
                                    "Linear Regression": {
                                        "model": LinearRegression(),
                                        "params": {
                                            "fit_intercept": [True, False],
                                            "normalize": [True, False]
                                        }
                                    },
                                    "Random Forest Regressor": {
                                        "model": RandomForestRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "max_depth": (3, 30),
                                            "min_samples_split": (2, 10),
                                            "min_samples_leaf": (1, 4)
                                        }
                                    },
                                    "Decision Tree Regressor": {
                                        "model": DecisionTreeRegressor(random_state=42),
                                        "params": {
                                            "max_depth": (3, 30),
                                            "min_samples_split": (2, 10),
                                            "min_samples_leaf": (1, 4)
                                        }
                                    },
                                    "Neural Network Regressor": {
                                        "model": MLPRegressor(random_state=42),
                                        "params": {
                                            "hidden_layer_sizes": ["(50,)", "(100,)", "(100,50)", "(100,50,25)"],
                                            "activation": ["relu", "tanh"],
                                            "alpha": (0.0001, 0.01),
                                            "learning_rate": ["constant", "adaptive"],
                                            "max_iter": (500, 2000)
                                        }
                                    },
                                    "SVR": {
                                        "model": SVR(),
                                        "params": {
                                            "C": (0.1, 10.0),
                                            "kernel": ["rbf", "linear"],
                                            "epsilon": (0.01, 0.5)
                                        }
                                    },
                                    "XGBoost Regressor": {
                                        "model": xgb.XGBRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "max_depth": (3, 10),
                                            "learning_rate": (0.01, 0.3),
                                            "subsample": (0.5, 1.0),
                                            "colsample_bytree": (0.5, 1.0)
                                        }
                                    },
                                    "LightGBM Regressor": {
                                        "model": lgb.LGBMRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "num_leaves": (20, 100),
                                            "learning_rate": (0.01, 0.3),
                                            "feature_fraction": (0.5, 1.0)
                                        }
                                    },
                                    "Gradient Boosting Regressor": {
                                        "model": GradientBoostingRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "max_depth": (3, 10),
                                            "learning_rate": (0.01, 0.3),
                                            "subsample": (0.5, 1.0)
                                        }
                                    },
                                    "Extra Trees Regressor": {
                                        "model": ExtraTreesRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "max_depth": (3, 15),
                                            "min_samples_split": (2, 10),
                                            "min_samples_leaf": (1, 4)
                                        }
                                    },
                                    "AdaBoost Regressor": {
                                        "model": AdaBoostRegressor(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "learning_rate": (0.01, 1.0)
                                        }
                                    }
                                }
                                target_col = st.selectbox(
                                    "Select target variable",
                                    df.select_dtypes(include=[np.number]).columns
                                )
                            else:  # Classification
                                models = {
                                    "Logistic Regression": {
                                        "model": LogisticRegression(random_state=42),
                                        "params": {
                                            "C": (0.001, 10.0),  # Wider range for regularization
                                            "max_iter": (100, 1000),
                                            "penalty": ["l1", "l2"]
                                        }
                                    },
                                    "Random Forest Classifier": {
                                        "model": RandomForestClassifier(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "max_depth": (3, 20),
                                            "min_samples_split": (2, 10),
                                            "min_samples_leaf": (1, 4)
                                        }
                                    },
                                    "Neural Network Classifier": {
                                        "model": MLPClassifier(random_state=42),
                                        "params": {
                                            "hidden_layer_sizes": ["50,25", "100,50", "100,50,25"],
                                            "alpha": (0.0001, 0.01),
                                            "learning_rate_init": (0.001, 0.1),
                                            "max_iter": (200, 2000)
                                        }
                                    },
                                    "XGBoost Classifier": {
                                        "model": xgb.XGBClassifier(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "max_depth": (3, 10),
                                            "learning_rate": (0.01, 0.3),
                                            "subsample": (0.5, 1.0),
                                            "colsample_bytree": (0.5, 1.0)
                                        }
                                    },
                                    "LightGBM Classifier": {
                                        "model": lgb.LGBMClassifier(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "num_leaves": (20, 100),
                                            "learning_rate": (0.01, 0.3),
                                            "feature_fraction": (0.5, 1.0)
                                        }
                                    },
                                    "Gradient Boosting Classifier": {
                                        "model": GradientBoostingClassifier(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 500),
                                            "max_depth": (3, 10),
                                            "learning_rate": (0.01, 0.3),
                                            "subsample": (0.5, 1.0)
                                        }
                                    },
                                    "Extra Trees Classifier": {
                                        "model": ExtraTreesClassifier(random_state=42),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "max_depth": (3, 15),
                                            "min_samples_split": (2, 10),
                                            "min_samples_leaf": (1, 4)
                                        }
                                    },
                                    "AdaBoost Classifier": {
                                        "model": AdaBoostClassifier(
                                            estimator=DecisionTreeClassifier(max_depth=10),
                                            n_estimators=50,
                                            learning_rate=0.1,
                                            random_state=42
                                        ),
                                        "params": {
                                            "n_estimators": (50, 300),
                                            "learning_rate": (0.01, 1.0)
                                        }
                                    }
                                }
                                target_col = st.selectbox(
                                    "Select target variable",
                                    df.columns
                                )
                            
                            model_name = st.selectbox(
                                "Choose model:",
                                list(models.keys())
                            )
                            
                            # Model hyperparameters
                            st.markdown("##### Model Hyperparameters")
                            if "Random Forest" in model_name:
                                n_estimators = st.slider("Number of trees", 10, 200, 100)
                                max_depth = st.slider("Maximum depth", 1, 50, 10)
                                if "Regressor" in model_name:
                                    models[model_name] = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        random_state=42
                                    )
                                else:
                                    models[model_name] = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        random_state=42
                                    )
                            elif "Neural Network" in model_name:
                                hidden_layers = st.text_input("Hidden layer sizes (comma-separated)", "100,50")
                                max_iter = st.slider("Maximum iterations", 100, 2000, 1000)
                                hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
                                if "Regressor" in model_name:
                                    models[model_name] = MLPRegressor(
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        max_iter=max_iter,
                                        random_state=42
                                    )
                                else:
                                    models[model_name] = MLPClassifier(
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        max_iter=max_iter,
                                        random_state=42
                                    )

                            feature_cols = st.multiselect(
                                "Select feature columns",
                                [col for col in df.columns if col != target_col],
                                default=[col for col in df.columns if col != target_col]
                            )
                            
                            if st.button(f"Apply {model_name}"):
                                if feature_cols:
                                    with st.spinner(f"Applying {model_name}..."):
                                        results = apply_model(
                                            df[feature_cols + [target_col]], 
                                            model_name, 
                                            target_column=target_col
                                        )
                                        if results:
                                            if model_category == "Regression":
                                                show_regression_results(model_name, results)
                                            else:
                                                show_classification_results(model_name, results)
                                else:
                                    st.warning("Please select at least one feature column")

                # Add Reset Processing Button
                if st.button("🔄 Reset Data Processing"):
                    reset_session_state()
                    st.rerun()

                # Download processed data
                if st.button("💾 Download Processed Data", key="dl_data"):
                    tmp_download_link = download_link(df, "processed_data.csv", "Click here to download")
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
            
            with col2:
                # Create tabs
                tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🔮 Predictions", "🔢 Data Preview"])
                
                # Visualization tab
                with tab1:
                    st.markdown("### 📊 Data Visualization")
                    
                    # Visualization selection
                    vis_container = st.container()
                    with vis_container:
                        vis_option = st.selectbox("Choose Visualization Type:", 
                                                ["2D Scatter Plot", "3D Scatter Plot", "Histogram", 
                                                "Box Plot", "Bar Plot", "Pie Chart", "Violin Plot", 
                                                "Line Chart", "Density Plot", "Correlation Heatmap"])
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        all_cols = df.columns.tolist()
                        
                        with st.expander("✨ Advanced Settings", expanded=True):
                            # Common settings for all plots
                            custom_title = st.text_input("Custom title (optional):")
                            # 2D Scatter Plot settings
                            if vis_option == "2D Scatter Plot":
                                if len(numeric_cols) >= 2:
                                    x_col = st.selectbox("X-axis", numeric_cols, index=0, key="x_scatter")
                                    y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="y_scatter")
                                    color_by = st.selectbox("Color by (optional)", [None] + all_cols, key="color_scatter")
                                    size_by = st.selectbox("Size by (optional)", [None] + numeric_cols, key="size_scatter")
                                else:
                                    st.error("At least two numeric columns are required for a scatter plot!")
                            
                            # 3D Scatter Plot settings
                            elif vis_option == "3D Scatter Plot":
                                if len(numeric_cols) >= 3:
                                    x_col = st.selectbox("X-axis", numeric_cols, index=0, key="x_3d")
                                    y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="y_3d")
                                    z_col = st.selectbox("Z-axis", numeric_cols, index=2, key="z_3d")
                                    color_by = st.selectbox("Color by (optional)", [None] + all_cols, key="color_3d")
                                else:
                                    st.error("At least three numeric columns are required for a 3D scatter plot!")
                            
                            # Histogram settings
                            elif vis_option == "Histogram":
                                if numeric_cols:
                                    column = st.selectbox("Column", numeric_cols, key="hist_col")
                                    bins = st.slider("Number of bins", min_value=10, max_value=100, value=30, key="hist_bins")
                                    color = st.selectbox("Color by (optional)", [None] + all_cols, key="hist_color")
                                else:
                                    st.error("No numeric columns available for histogram!")
                            
                            # Box Plot settings
                            elif vis_option == "Box Plot":
                                if numeric_cols:
                                    column = st.selectbox("Column", numeric_cols, key="box_col")
                                    group_by = st.selectbox("Group by (optional)", [None] + all_cols, key="box_group")
                                else:
                                    st.error("No numeric columns available for box plot!")
                            
                            # Bar Plot settings
                            elif vis_option == "Bar Plot":
                                x_col = st.selectbox("X-axis", all_cols, key="bar_x")
                                if numeric_cols:
                                    y_col = st.selectbox("Y-axis", numeric_cols, key="bar_y")
                                    color_by = st.selectbox("Color by (optional)", [None] + all_cols, key="bar_color")
                                else:
                                    st.error("No numeric columns available for Y-axis!")
                            
                            # Pie Chart settings
                            elif vis_option == "Pie Chart":
                                names_col = st.selectbox("Names column", all_cols, key="pie_names")
                                values_col = st.selectbox("Values column (optional)", [None] + numeric_cols, key="pie_values")
                            
                            # Violin Plot settings
                            elif vis_option == "Violin Plot":
                                x_col = st.selectbox("X-axis", all_cols, key="violin_x")
                                if numeric_cols:
                                    y_col = st.selectbox("Y-axis", numeric_cols, key="violin_y")
                                    color_by = st.selectbox("Color by (optional)", [None] + all_cols, key="violin_color")
                                else:
                                    st.error("No numeric columns available for Y-axis!")
                            
                            # Line Chart settings
                            elif vis_option == "Line Chart":
                                x_col = st.selectbox("X-axis", all_cols, key="line_x")
                                if numeric_cols:
                                    y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
                                    color_by = st.selectbox("Color by (optional)", [None] + all_cols, key="line_color")
                                else:
                                    st.error("No numeric columns available for Y-axis!")
                            
                            # Density Plot settings
                            elif vis_option == "Density Plot":
                                if len(numeric_cols) >= 2:
                                    x_col = st.selectbox("X-axis", numeric_cols, index=0, key="density_x")
                                    y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="density_y")
                                else:
                                    st.error("At least two numeric columns are required for a density plot!")
                            
                            # Correlation Heatmap settings
                            elif vis_option == "Correlation Heatmap":
                                pass  # No specific settings for heatmap
                        
                        if st.button("Generate Visualization", key="generate_viz"):
                            try:
                                fig = None  # Initialize fig as None
                                
                                if vis_option == "2D Scatter Plot":
                                    fig = create_scatter_plot(df, x_col, y_col, color_by, size_by, custom_title)
                                elif vis_option == "3D Scatter Plot":
                                    fig = create_scatter_plot_3d(df, x_col, y_col, z_col, color_by, custom_title)
                                elif vis_option == "Histogram":
                                    fig = create_histogram(df, column, bins, color, custom_title)
                                elif vis_option == "Box Plot":
                                    fig = create_box_plot(df, column, group_by, custom_title)
                                elif vis_option == "Bar Plot":
                                    fig = create_bar_plot(df, x_col, y_col, color_by, custom_title)
                                elif vis_option == "Pie Chart":
                                    fig = create_pie_chart(df, names_col, values_col, custom_title)
                                elif vis_option == "Violin Plot":
                                    fig = create_violin_plot(df, x_col, y_col, color_by, custom_title)
                                elif vis_option == "Line Chart":
                                    fig = create_line_chart(df, x_col, y_col, color_by, custom_title)
                                elif vis_option == "Density Plot":
                                    fig = create_density_plot(df, x_col, y_col, custom_title)
                                elif vis_option == "Correlation Heatmap":
                                    numeric_df = df.select_dtypes(include=[np.number])
                                    if numeric_df.shape[1] < 2:
                                        st.error("At least two numeric columns are required for a correlation heatmap!")
                                    else:
                                        fig = create_heatmap(df, custom_title)
                                
                                # Display the figure if it was created
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True, key=f"viz_{vis_option.lower().replace(' ', '_')}")
                                    
                            except Exception as e:
                                st.error(f"An error occurred while generating the visualization: {e}")
                
                # Predictions tab
                with tab2:
                    st.markdown("### 🔮 Machine Learning Predictions")
                    
                    # Create two columns for preprocessing and model
                    pred_prep_col, pred_model_col = st.columns([1, 1])
                    
                    with pred_prep_col:
                        st.markdown("#### Data Preprocessing")
                        
                        # Target Selection
                        target_col = st.selectbox(
                            "Select Target Variable",
                            df.columns,
                            help="Choose the variable you want to predict"
                        )
                        
                        # Feature Selection
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        feature_cols = st.multiselect(
                            "Select Features",
                            [col for col in df.columns if col != target_col],
                            default=[col for col in numeric_cols if col != target_col],
                            help="Choose the features to use for prediction"
                        )
                        
                        if feature_cols:
                            with st.expander("Preprocessing Options", expanded=True):
                                # Handle Missing Values
                                handle_missing = st.checkbox("Handle Missing Values")
                                if handle_missing:
                                    missing_strategy = st.selectbox(
                                        "Missing Values Strategy",
                                        ["Mean", "Median", "most_frequent"]
                                    )
                                
                                # Feature Scaling
                                scale_features = st.checkbox("Scale Features")
                                if scale_features:
                                    scaler_type = st.selectbox(
                                        "Scaling Method",
                                        ["Standard", "MinMax"]
                                    )
                                
                                # Categorical Encoding
                                if categorical_cols:
                                    encode_categorical = st.checkbox("Encode Categorical Variables")
                                    if encode_categorical:
                                        encoding_method = st.selectbox(
                                            "Encoding Method",
                                            ["One-Hot", "Label"]
                                        )
                    
                    with pred_model_col:
                        st.markdown("#### Model Configuration")
                        
                        # Determine if regression or classification and set up models
                        if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 10:
                            task_type = "Regression"
                            models = {
                                "Linear Regression": {
                                    "model": LinearRegression(),
                                    "params": {
                                        "fit_intercept": [True, False],
                                        "normalize": [True, False]
                                    }
                                },
                                "Random Forest Regressor": {
                                    "model": RandomForestRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "max_depth": (3, 30),
                                        "min_samples_split": (2, 10),
                                        "min_samples_leaf": (1, 4)
                                    }
                                },
                                "Decision Tree Regressor": {
                                    "model": DecisionTreeRegressor(random_state=42),
                                    "params": {
                                        "max_depth": (3, 30),
                                        "min_samples_split": (2, 10),
                                        "min_samples_leaf": (1, 4)
                                    }
                                },
                                "Neural Network Regressor": {
                                    "model": MLPRegressor(random_state=42),
                                    "params": {
                                        "hidden_layer_sizes": ["(50,)", "(100,)", "(100,50)", "(100,50,25)"],
                                        "activation": ["relu", "tanh"],
                                        "alpha": (0.0001, 0.01),
                                        "learning_rate": ["constant", "adaptive"],
                                        "max_iter": (500, 2000)
                                    }
                                },
                                "SVR": {
                                    "model": SVR(),
                                    "params": {
                                        "C": (0.1, 10.0),
                                        "kernel": ["rbf", "linear"],
                                        "epsilon": (0.01, 0.5)
                                    }
                                },
                                "XGBoost Regressor": {
                                    "model": xgb.XGBRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "max_depth": (3, 10),
                                        "learning_rate": (0.01, 0.3),
                                        "subsample": (0.5, 1.0),
                                        "colsample_bytree": (0.5, 1.0)
                                    }
                                },
                                "LightGBM Regressor": {
                                    "model": lgb.LGBMRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "num_leaves": (20, 100),
                                        "learning_rate": (0.01, 0.3),
                                        "feature_fraction": (0.5, 1.0)
                                    }
                                },
                                "Gradient Boosting Regressor": {
                                    "model": GradientBoostingRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "max_depth": (3, 10),
                                        "learning_rate": (0.01, 0.3),
                                        "subsample": (0.5, 1.0)
                                    }
                                },
                                "Extra Trees Regressor": {
                                    "model": ExtraTreesRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "max_depth": (3, 15),
                                        "min_samples_split": (2, 10),
                                        "min_samples_leaf": (1, 4)
                                    }
                                },
                                "AdaBoost Regressor": {
                                    "model": AdaBoostRegressor(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "learning_rate": (0.01, 1.0)
                                    }
                                }
                            }
                        else:
                            task_type = "Classification"
                            models = {
                                "Logistic Regression": {
                                    "model": LogisticRegression(random_state=42),
                                    "params": {
                                        "C": (0.001, 10.0),  # Wider range for regularization
                                        "max_iter": (100, 1000),
                                        "penalty": ["l1", "l2"]
                                    }
                                },
                                "Random Forest Classifier": {
                                    "model": RandomForestClassifier(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "max_depth": (3, 20),
                                        "min_samples_split": (2, 10),
                                        "min_samples_leaf": (1, 4)
                                    }
                                },
                                "Neural Network Classifier": {
                                    "model": MLPClassifier(random_state=42),
                                    "params": {
                                        "hidden_layer_sizes": ["50,25", "100,50", "100,50,25"],
                                        "alpha": (0.0001, 0.01),
                                        "learning_rate_init": (0.001, 0.1),
                                        "max_iter": (200, 2000)
                                    }
                                },
                                "XGBoost Classifier": {
                                    "model": xgb.XGBClassifier(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "max_depth": (3, 10),
                                        "learning_rate": (0.01, 0.3),
                                        "subsample": (0.5, 1.0),
                                        "colsample_bytree": (0.5, 1.0)
                                    }
                                },
                                "LightGBM Classifier": {
                                    "model": lgb.LGBMClassifier(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "num_leaves": (20, 100),
                                        "learning_rate": (0.01, 0.3),
                                        "feature_fraction": (0.5, 1.0)
                                    }
                                },
                                "Gradient Boosting Classifier": {
                                    "model": GradientBoostingClassifier(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 500),
                                        "max_depth": (3, 10),
                                        "learning_rate": (0.01, 0.3),
                                        "subsample": (0.5, 1.0)
                                    }
                                },
                                "Extra Trees Classifier": {
                                    "model": ExtraTreesClassifier(random_state=42),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "max_depth": (3, 15),
                                        "min_samples_split": (2, 10),
                                        "min_samples_leaf": (1, 4)
                                    }
                                },
                                "AdaBoost Classifier": {
                                    "model": AdaBoostClassifier(
                                        estimator=DecisionTreeClassifier(max_depth=10),
                                        n_estimators=50,
                                        learning_rate=0.1,
                                        random_state=42
                                    ),
                                    "params": {
                                        "n_estimators": (50, 300),
                                        "learning_rate": (0.01, 1.0)
                                    }
                                }
                            }
                        
                        st.markdown(f"Task Type: **{task_type}**")
                        
                        # Model Selection with Apply buttons for each model
                        st.write("#### Select Models to Apply")
                        
                        selected_models = []
                        for model_name, model_info in models.items():
                            # Use expander instead of columns
                            with st.expander(f"📊 {model_name}", expanded=False):
                                if st.checkbox(f"Select {model_name}", key=f"select_{model_name}"):
                                    selected_models.append(model_name)
                                    st.markdown("##### Model Parameters")
                                    params = {}
                                    for param_name, param_range in model_info["params"].items():
                                        if isinstance(param_range, tuple):
                                            if isinstance(param_range[0], int):
                                                params[param_name] = st.slider(
                                                    f"{param_name}", 
                                                    param_range[0], 
                                                    param_range[1], 
                                                    key=f"{model_name}_{param_name}"
                                                )
                                            else:
                                                params[param_name] = st.slider(
                                                    f"{param_name}", 
                                                    float(param_range[0]), 
                                                    float(param_range[1]),
                                                    key=f"{model_name}_{param_name}"
                                                )
                                        elif isinstance(param_range, list):
                                            params[param_name] = st.selectbox(
                                                f"{param_name}",
                                                param_range,
                                                key=f"{model_name}_{param_name}"
                                            )

                        # Apply selected models
                        if selected_models and st.button("Train Selected Models"):
                            st.markdown("### 📊 Model Results Comparison")
                            
                            results_container = st.container()
                            with results_container:
                                all_results = {}
                                for model_name in selected_models:
                                    with st.spinner(f"Training {model_name}..."):
                                        try:
                                            # Get model and its parameters
                                            model_info = models[model_name]
                                            model = model_info["model"]
                                            params = model_info.get("current_params", {})
                                            
                                            # Set parameters
                                            if params:
                                                if "hidden_layer_sizes" in params:
                                                    params["hidden_layer_sizes"] = tuple(map(int, params["hidden_layer_sizes"].split(',')))
                                                model.set_params(**params)
                                            
                                            # Train and evaluate model
                                            results = apply_model(df[feature_cols + [target_col]], model_name, target_column=target_col)
                                            all_results[model_name] = results
                                            
                                            # Store trained model in session state
                                            st.session_state.trained_models[model_name] = results
                                            
                                            if results:
                                                st.markdown(f"#### {model_name} Results")
                                                if task_type == "Regression":
                                                    show_regression_results(model_name, results)
                                                else:
                                                    show_classification_results(model_name, results)
                                        except Exception as e:
                                            st.error(f"Error training {model_name}: {str(e)}")
                                    
                                show_model_comparison(all_results, task_type)
                        
                        # Add this inside tab2 (Predictions tab), after the model comparison section:
                        if selected_models:
                            st.markdown("### 🎯 Make New Predictions")
                            st.info("Enter values for features to get predictions from trained models")
                            
                            # Create a single form for all inputs
                            with st.form("prediction_form"):
                                st.markdown("#### Enter Feature Values")
                                
                                # Create input fields without nested columns
                                input_data = {}
                                
                                # Create input fields in a grid-like layout
                                for feature in feature_cols:
                                    if df[feature].dtype in ['int64', 'float64']:
                                        # For numeric features
                                        mean_val = float(df[feature].mean())
                                        std_val = float(df[feature].std())
                                        input_data[feature] = st.number_input(
                                            f"{feature}",
                                            value=mean_val,
                                            format="%.2f",
                                            help=f"Mean: {mean_val:.2f}, Std: {std_val:.2f}",
                                            key=f"input_{feature}"
                                        )
                                    else:
                                        # For categorical features
                                        unique_values = df[feature].unique()
                                        input_data[feature] = st.selectbox(
                                            f"{feature}",
                                            options=unique_values,
                                            help=f"Unique values: {len(unique_values)}",
                                            key=f"input_{feature}"
                                        )
                                
                                # Add submit button at the bottom of the form
                                submit_button = st.form_submit_button(
                                    "Make Predictions",
                                    help="Click to get predictions from all selected models"
                                )
                            
                            # Handle form submission
                            if submit_button:
                                if not st.session_state.trained_models:
                                    st.error("Please train the models first before making predictions!")
                                else:
                                    st.markdown("### 📊 Prediction Results")
                                    
                                    # Create predictions table
                                    predictions_df = pd.DataFrame(columns=['Model', 'Prediction', 'Confidence/Probability'])
                                    
                                    for model_name in selected_models:
                                        with st.spinner(f"Getting prediction from {model_name}..."):
                                            try:
                                                # Get the trained model from session state
                                                model_results = st.session_state.trained_models.get(model_name)
                                                if model_results is None:
                                                    st.warning(f"Model {model_name} has not been trained yet.")
                                                    continue
                                                    
                                                model = model_results['model']
                                                
                                                # Prepare input data
                                                input_df = pd.DataFrame([input_data])
                                                
                                                # Apply preprocessing
                                                processed_input = preprocess_input_data(
                                                    input_df,
                                                    handle_missing=handle_missing,
                                                    missing_strategy=missing_strategy if handle_missing else None,
                                                    scale_features=scale_features,
                                                    scaler_type=scaler_type if scale_features else None,
                                                    encode_categorical=encode_categorical if categorical_cols else False,
                                                    encoding_method=encoding_method if categorical_cols and encode_categorical else None
                                                )
                                                
                                                # Make prediction
                                                prediction = model.predict(processed_input)[0]
                                                
                                                # Get probability for classification models
                                                probability = None
                                                if hasattr(model, 'predict_proba'):
                                                    probability = model.predict_proba(processed_input)[0].max()
                                                
                                                # Add to predictions dataframe
                                                predictions_df.loc[len(predictions_df)] = [
                                                    model_name,
                                                    f"{prediction:.4f}" if isinstance(prediction, (float, np.float64)) else prediction,
                                                    f"{probability:.4f}" if probability is not None else "N/A"
                                                ]
                                            except Exception as e:
                                                st.error(f"Error making prediction with {model_name}: {str(e)}")
                                    
                                    # Display results if we have any predictions
                                    if not predictions_df.empty:
                                        show_prediction_results(predictions_df, task_type)
                
                # Data Preview tab
                with tab3:
                    st.markdown("### 🔢 Data Preview")
                    display_df = convert_df_for_streamlit(df)
                    st.dataframe(display_df)

def show_dim_reduction_results(model_type, results, n_components):
    """Display dimensionality reduction results."""
    st.markdown(f"### {model_type} Results")
    
    if results is None:
        st.error("No results to display")
        return
    
    # Show explained variance for PCA
    if model_type == "PCA" and 'explained_variance' in results:
        exp_var = results['explained_variance']
        cum_exp_var = np.cumsum(exp_var)
        
        # Create explained variance plot
        fig_var = px.line(
            y=cum_exp_var,
            title="Cumulative Explained Variance Ratio",
            labels={"index": "Number of Components", "y": "Cumulative Explained Variance"},
            template="plotly_dark"
        )
        fig_var.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(20, 24, 35, 0.8)',
            paper_bgcolor='rgba(20, 24, 35, 0.8)'
        )
        st.plotly_chart(fig_var)
        
        # Show variance table
        var_df = pd.DataFrame({
            'Component': [f'Component_{i+1}' for i in range(len(exp_var))],
            'Explained Variance Ratio': exp_var,
            'Cumulative Explained Variance': cum_exp_var
        })
        st.write("Explained Variance Details:")
        st.dataframe(var_df)

    # Show transformed data visualizations
    if 'data' in results:
        transformed_data = results['data']
        
        # 2D Visualization
        if n_components >= 2:
            fig_2d = px.scatter(
                transformed_data,
                x='Component_1',
                y='Component_2',
                title=f"{model_type} 2D Projection",
                template="plotly_dark"
            )
            fig_2d.update_layout(
                plot_bgcolor='rgba(20, 24, 35, 0.8)',
                paper_bgcolor='rgba(20, 24, 35, 0.8)'
            )
            st.plotly_chart(fig_2d)
        
        # 3D Visualization
        if n_components >= 3:
            fig_3d = px.scatter_3d(
                transformed_data,
                x='Component_1',
                y='Component_2',
                z='Component_3',
                title=f"{model_type} 3D Projection",
                template="plotly_dark"
            )
            fig_3d.update_layout(
                scene=dict(
                    bgcolor='rgba(20, 24, 35, 0.8)'
                ),
                paper_bgcolor='rgba(20, 24, 35, 0.8)'
            )
            st.plotly_chart(fig_3d)
        
        # Show transformed data preview
        st.write("#### Transformed Data Preview")
        st.dataframe(transformed_data.head())
        
        # Show shape information
        st.write(f"Original data shape: {results['data'].shape}")
        st.write(f"Transformed data shape: {transformed_data.shape}")

def show_clustering_results(results, algorithm_name):
    """Display clustering results with algorithm-specific visualizations."""
    st.markdown(f"#### {algorithm_name} Clustering Results")
    
    if 'data' in results and 'Cluster' in results['data'].columns:
        # Get numeric columns for visualization
        numeric_cols = results['data'].select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if len(numeric_cols) >= 2:
            # 2D Scatter plot
            fig_2d = px.scatter(
                results['data'],
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='Cluster',
                title=f"{algorithm_name}: 2D Cluster Visualization",
                template="plotly_dark"
            )
            st.plotly_chart(fig_2d)
            
            # 3D Scatter plot if we have enough dimensions
            if len(numeric_cols) >= 3:
                fig_3d = px.scatter_3d(
                    results['data'],
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    z=numeric_cols[2],
                    color='Cluster',
                    title=f"{algorithm_name}: 3D Cluster Visualization",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_3d)
        
        # Show cluster distribution
        st.write("#### Cluster Distribution")
        cluster_dist = results['data']['Cluster'].value_counts().sort_index()
        fig_dist = px.bar(
            x=cluster_dist.index,
            y=cluster_dist.values,
            title="Cluster Size Distribution",
            labels={'x': 'Cluster', 'y': 'Number of Samples'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist)
        
        # Algorithm-specific metrics
        st.write("#### Clustering Metrics")
        metrics = results.get('metrics', {})
        
        st.write(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}")
        if 'inertia' in metrics:
            st.write(f"Inertia: {metrics['inertia']:.4f}")
        
        # Show parameters used
        st.write("#### Algorithm Parameters")
        st.json(metrics.get('params', {}))
        
        # Add download button for results
        if st.button("Download Clustering Results"):
            tmp_download_link = download_link(
                results['data'],
                f"{algorithm_name.lower()}_clustering_results.csv",
                "Click here to download"
            )
            st.markdown(tmp_download_link, unsafe_allow_html=True)

def show_regression_results(model_name, results):
    """Display regression results."""
    st.markdown(f"#### {model_name} Results")
    
    metrics = results.get('metrics', {})
    st.write("Model Performance:")
    st.write(f"- R² Score: {metrics.get('r2', 'N/A'):.4f}")
    st.write(f"- Mean Squared Error: {metrics.get('mse', 'N/A'):.4f}")
    
    if 'feature_importance' in results and results['feature_importance'] is not None:
        st.write("Feature Importance:")
        fig = px.bar(
            results['feature_importance'],
            x='Feature',
            y='Importance',
            title="Feature Importance",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

def show_classification_results(model_name, results):
    """Display classification results."""
    st.markdown(f"#### {model_name} Results")
    
    metrics = results.get('metrics', {})
    st.write("Model Performance:")
    st.write(f"- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    
    if 'classification_report' in metrics:
        st.write("Classification Report:")
        st.code(metrics['classification_report'])
    
    if 'feature_importance' in results and results['feature_importance'] is not None:
        st.write("Feature Importance:")
        fig = px.bar(
            results['feature_importance'],
            x='Feature',
            y='Importance',
            title="Feature Importance",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

def show_model_comparison(all_results, task_type):
    if not isinstance(all_results, dict) or not all_results:
        st.warning("No model results available for comparison.")
        return
    
    # Initialize DataFrame for metrics
    metrics_df = pd.DataFrame()
    
    # Collect metrics from valid results
    for model_name, results in all_results.items():
        try:
            if results is None:
                st.warning(f"No results available for {model_name}")
                continue
                
            metrics = results.get('metrics', {})
            if not metrics:
                st.warning(f"No metrics available for {model_name}")
                continue
                
            if task_type == "Regression":
                metrics_df.loc[model_name, 'R² Score'] = metrics.get('r2', 0)
                metrics_df.loc[model_name, 'MSE'] = metrics.get('mse', 0)
            else:
                metrics_df.loc[model_name, 'Accuracy'] = metrics.get('accuracy', 0)
        except Exception as e:
            st.error(f"Error processing results for {model_name}: {str(e)}")
            continue
    
    # Only create visualizations if we have data
    if metrics_df.empty:
        st.warning("No valid metrics available for comparison.")
        return
        
    # Create comparison plots
    try:
        if task_type == "Regression":
            if 'R² Score' in metrics_df.columns:
                fig_r2 = px.bar(
                    metrics_df.reset_index(),
                    x='index',
                    y='R² Score',
                    title="R² Score Comparison",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_r2)
            
            if 'MSE' in metrics_df.columns:
                fig_mse = px.bar(
                    metrics_df.reset_index(),
                    x='index',
                    y='MSE',
                    title="Mean Squared Error Comparison",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_mse)
        else:
            if 'Accuracy' in metrics_df.columns:
                fig_acc = px.bar(
                    metrics_df.reset_index(),
                    x='index',
                    y='Accuracy',
                    title="Accuracy Comparison",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_acc)
        
        # Show metrics table
        st.write("#### Detailed Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0))
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def show_prediction_results(predictions_df, task_type):
    """Display prediction results with improved styling and visibility."""
    # Custom CSS for better visibility
    st.markdown("""
    <style>
    .prediction-container {
        background-color: rgba(26, 35, 51, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prediction-header {
        color: #5663F7;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    .prediction-table {
        width: 100%;
        color: white !important;
    }
    .dataframe {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display predictions in styled container
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown('<p class="prediction-header">Model Predictions</p>', unsafe_allow_html=True)
    
    # Convert numeric values to formatted strings
    formatted_df = predictions_df.copy()
    
    # Modified prediction formatting to handle both numerical and categorical values
    def format_prediction(x):
        if x == 'N/A':
            return x
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)
    
    formatted_df['Prediction'] = formatted_df['Prediction'].apply(format_prediction)
    
    # Format confidence/probability values
    def format_confidence(x):
        if x == 'N/A':
            return x
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)
    
    formatted_df['Confidence/Probability'] = formatted_df['Confidence/Probability'].apply(format_confidence)
    
    # Display the table with custom styling
    st.dataframe(
        formatted_df.style
        .set_properties(**{
            'background-color': 'rgba(70, 73, 80, 0.2)',
            'color': 'white',
            'border': '1px solid rgba(255, 255, 255, 0.1)'
        })
        .highlight_max(
            subset=['Confidence/Probability'],
            props='color: #5663F7; font-weight: bold;'
        )
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Show visualization of predictions
    if task_type == "Classification":
        fig = px.bar(
            predictions_df,
            x='Model',
            y='Confidence/Probability',
            title="Model Predictions Confidence",
            template="plotly_dark",
            color_discrete_sequence=['#5663F7']
        )
    else:
        # For regression, only create bar plot if predictions are numeric
        try:
            numeric_preds = pd.to_numeric(predictions_df['Prediction'])
            fig = px.bar(
                predictions_df,
                x='Model',
                y='Prediction',
                title="Model Predictions Comparison",
                template="plotly_dark",
                color_discrete_sequence=['#5663F7']
            )
        except (ValueError, TypeError):
            # If predictions are categorical, create a different visualization
            fig = px.bar(
                predictions_df,
                x='Model',
                y='Confidence/Probability',
                title="Model Confidence Scores",
                template="plotly_dark",
                color_discrete_sequence=['#5663F7']
            )
    
    # Update figure layout for better visibility
    fig.update_layout(
        plot_bgcolor='rgba(26, 35, 51, 0.9)',
        paper_bgcolor='rgba(26, 35, 51, 0.9)',
        font=dict(color='white'),
        title=dict(
            font=dict(size=20, color='white', family='Poppins'),
            x=0.5
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='white')
        )
    )
    
    st.plotly_chart(fig)

# Add this function for model evaluation
def evaluate_model(model, X, y, task_type, cv=5):
    """Evaluate model using cross-validation."""
    from sklearn.model_selection import cross_val_score, cross_val_predict
    
    if task_type == "Regression":
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        predictions = cross_val_predict(model, X, y, cv=cv)
        mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return {
            'cv_r2_mean': scores.mean(),
            'cv_r2_std': scores.std(),
            'cv_mse_mean': mse_scores.mean(),
            'cv_mse_std': mse_scores.std(),
            'predictions': predictions
        }
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        predictions = cross_val_predict(model, X, y, cv=cv)
        if hasattr(model, 'predict_proba'):
            proba_predictions = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        else:
            proba_predictions = None
        return {
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'predictions': predictions,
            'probabilities': proba_predictions
        }

def analyze_regression_features(X, y, feature_names):
    """Analyze feature importance and relationships for regression."""
    try:
        # Correlation analysis
        corr_matrix = pd.DataFrame(X, columns=feature_names).corrwith(pd.Series(y))
        
        fig = px.bar(
            x=feature_names,
            y=corr_matrix.values,
            title="Feature Correlation with Target",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
        
        # Feature distribution analysis
        for feature in feature_names:
            fig = px.histogram(
                x=X[:, feature_names.index(feature)],
                title=f"Distribution of {feature}",
                template="plotly_dark"
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error in feature analysis: {str(e)}")

def check_regression_assumptions(model, X, y):
    """Check basic regression assumptions."""
    try:
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Normality of residuals
        fig = px.histogram(
            x=residuals,
            title="Distribution of Residuals",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
        
        # Homoscedasticity
        fig = px.scatter(
            x=y_pred,
            y=np.abs(residuals),
            title="Residuals Magnitude vs Predicted Values",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error checking regression assumptions: {str(e)}")

def reset_session_state():
    """Reset all session state variables to their initial values."""
    st.session_state.processed_df = None
    st.session_state.trained_models = {}
    # Add any other session state variables that need to be reset

def initialize_classification_model(model_type):
    """Initialize classification model with proper parameters."""
    try:
        if model_type == "Neural Network Classifier":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),  # Two hidden layers
                max_iter=1000,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        elif model_type == "AdaBoost Classifier":
            return AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=10),
                n_estimators=50,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "Logistic Regression":
            return LogisticRegression(random_state=42)
        elif model_type == "Random Forest Classifier":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif model_type == "XGBoost Classifier":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "LightGBM Classifier":
            return lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1  # Suppress warnings
            )
        elif model_type == "Gradient Boosting Classifier":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "Extra Trees Classifier":
            return ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            st.error(f"Unknown classification model type: {model_type}")
            return None
    except Exception as e:
        st.error(f"Error initializing {model_type}: {str(e)}")
        return None

def initialize_regression_model(model_type, X_train=None):
    """Initialize regression model with proper parameters."""
    try:
        if model_type == "Linear Regression":
            return LinearRegression()
            
        elif model_type == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
        elif model_type == "Decision Tree Regressor":
            return DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
        elif model_type == "Neural Network Regressor":
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                random_state=42
            )
            
        elif model_type == "SVR":
            return SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
            
        elif model_type == "XGBoost Regressor":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == "LightGBM Regressor":
            return lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                min_child_samples=20,
                min_child_weight=1e-3,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                force_row_wise=True
            )
            
        elif model_type == "Gradient Boosting Regressor":
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
        elif model_type == "Extra Trees Regressor":
            return ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
        elif model_type == "AdaBoost Regressor":
            return AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=50,
                learning_rate=0.1,
                random_state=42
            )
            
        else:
            st.error(f"Unknown regression model type: {model_type}")
            return None
            
    except Exception as e:
        st.error(f"Error initializing {model_type}: {str(e)}")
        return None

if __name__ == '__main__':
    main()