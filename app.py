# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io
from io import StringIO
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    r2_score,
    accuracy_score,
    classification_report
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ['STREAMLIT_WATCH_EXCLUDE_PATHS'] = 'torch._classes'

# --- Helper Functions ---

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

def impute_data(df, strategy, knn_k=5):
    """Impute missing values for numeric columns using the selected strategy."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_imputed = df.copy()
    if len(numeric_cols) > 0:
        if strategy == 'Mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'Median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'Most Frequent':
            imputer = SimpleImputer(strategy='most_frequent')
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

def apply_model(df, model_type, target_column=None, n_clusters=3, n_components=2, **kwargs):
    """Apply selected model to the data."""
    if target_column:
        # Prepare data for supervised learning
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize result dictionary
        results = {
            'model': None,
            'metrics': {},
            'data': df,
            'feature_importance': None
        }
        
        if model_type == "Linear Regression":
            model = LinearRegression()
            is_classifier = False
        
        elif model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
            is_classifier = True
        
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
            is_classifier = False
        
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42)
            is_classifier = True
        
        elif model_type == "Naive Bayes":
            model = GaussianNB()
            is_classifier = True
        
        elif model_type == "SVM":
            if len(np.unique(y)) > 2:  # Regression task
                model = SVR(kernel='rbf')
                is_classifier = False
            else:  # Classification task
                model = SVC(kernel='rbf', probability=True)
                is_classifier = True
        
        elif model_type == "Neural Network":
            if len(np.unique(y)) > 10:  # Regression task
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=42
                )
                is_classifier = False
            else:  # Classification task
                model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=42
                )
                is_classifier = True
        
        # Fit model and get predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_classifier:
            results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
            results['metrics']['classification_report'] = classification_report(y_test, y_pred)
        else:
            results['metrics']['mse'] = mean_squared_error(y_test, y_pred)
            results['metrics']['r2'] = r2_score(y_test, y_pred)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        results['model'] = model
        return results
    
    else:  # Unsupervised learning
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.error("Need at least 2 numeric columns")
            return None
        
        results = {
            'model': None,
            'data': df.copy(),
            'embeddings': None
        }
        
        if model_type == "K-means Clustering":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(numeric_df)
            results['data']['Cluster'] = clusters
            results['model'] = model
        
        elif model_type == "PCA":
            model = PCA(n_components=n_components)
            embeddings = model.fit_transform(numeric_df)
            results['embeddings'] = embeddings
            results['explained_variance'] = model.explained_variance_ratio_
            results['model'] = model
            
            # Add PCA components to dataframe
            pca_cols = [f'PC{i+1}' for i in range(n_components)]
            results['data'] = pd.concat([
                df,
                pd.DataFrame(embeddings, columns=pca_cols, index=df.index)
            ], axis=1)
        
        elif model_type == "t-SNE":
            model = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                random_state=42
            )
            embeddings = model.fit_transform(numeric_df)
            results['embeddings'] = embeddings
            results['model'] = model
            
            # Add transformed columns
            transformed_cols = [f'TSNE{i+1}' for i in range(n_components)]
            results['data'] = pd.concat([
                df,
                pd.DataFrame(embeddings, columns=transformed_cols, index=df.index)
            ], axis=1)
        
        elif model_type == "UMAP":
            model = umap.UMAP(
                n_components=n_components,
                random_state=42
            )
            embeddings = model.fit_transform(numeric_df)
            results['embeddings'] = embeddings
            results['model'] = model
            
            # Add transformed columns
            transformed_cols = [f'UMAP{i+1}' for i in range(n_components)]
            results['data'] = pd.concat([
                df,
                pd.DataFrame(embeddings, columns=transformed_cols, index=df.index)
            ], axis=1)
        
        return results

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
        yaxis=dict(showgrid=True, gridcolor='rgba(211, 211, 211, 0.2)')
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
        object_to_download = object_to_download.to_csv(index=False)
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
    st.set_page_config(page_title="Dynamic AI Data Visualization Agent", layout="wide")
    set_custom_theme()
    
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
        df = load_data(uploaded_file)
        if df is not None:
            with col1:
                st.markdown("### 🔧 Data Processing Options")
                
                # Data Cleaning Box
                with st.expander("🧹 Data Cleaning", expanded=False):
                    if st.checkbox("Enable data cleaning"):
                        st.info("This will remove duplicates and rows with all missing values")
                        if st.button("Clean Data"):
                            with st.spinner("Cleaning data..."):
                                df = clean_data(df)
                                st.success("Data cleaned successfully!")

                # Missing Values Box
                with st.expander("🔍 Handle Missing Values", expanded=False):
                    if st.checkbox("Enable missing value imputation"):
                        strategy = st.selectbox(
                            "Choose imputation method:",
                            ["Mean", "Median", "Most Frequent", "KNN"],
                            help="Select how you want to fill missing values"
                        )
                        
                        if strategy == "KNN":
                            k = st.slider("Number of neighbors (k)", 1, 10, 5)
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        cols_to_impute = st.multiselect(
                            "Select columns to impute",
                            numeric_cols,
                            default=numeric_cols
                        )
                        
                        if st.button("Apply Imputation"):
                            if cols_to_impute:
                                with st.spinner(f"Imputing missing values using {strategy}..."):
                                    df = impute_data(df, strategy, knn_k=k if strategy == "KNN" else None)
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
                                    st.success("✅ Categorical data encoded successfully!")
                            else:
                                st.warning("Please select at least one column to encode")

                # Model Selection Box
                with st.expander("🤖 Apply Models", expanded=False):
                    if st.checkbox("Enable model application"):
                        model_category = st.selectbox(
                            "Select model type:",
                            ["Dimensionality Reduction", "Clustering", "Regression", "Classification"],
                            help="Choose the type of model to apply"
                        )
                        
                        # Dynamic model options based on category
                        if model_category == "Dimensionality Reduction":
                            dim_model = st.selectbox(
                                "Choose dimensionality reduction method:",
                                ["PCA", "t-SNE", "UMAP"]
                            )
                            
                            n_components = st.slider(
                                "Number of components", 
                                min_value=2, 
                                max_value=min(5, df.shape[1]), 
                                value=2
                            )
                            
                            if dim_model == "t-SNE":
                                perplexity = st.slider("Perplexity", 5, 50, 30)
                            
                            if st.button(f"Apply {dim_model}"):
                                with st.spinner(f"Applying {dim_model}..."):
                                    results = apply_model(df, dim_model, 
                                                       n_components=n_components,
                                                       perplexity=perplexity if dim_model == "t-SNE" else None)
                                    if results:
                                        show_dim_reduction_results(dim_model, results, n_components)

                        elif model_category == "Clustering":
                            cluster_model = st.selectbox(
                                "Choose clustering method:",
                                ["K-means"]
                            )
                            
                            n_clusters = st.slider("Number of clusters", 2, 10, 3)
                            
                            if st.button(f"Apply {cluster_model}"):
                                with st.spinner(f"Applying {cluster_model}..."):
                                    results = apply_model(df, "K-means Clustering", n_clusters=n_clusters)
                                    if results:
                                        show_clustering_results(results)

                        elif model_category in ["Regression", "Classification"]:
                            if model_category == "Regression":
                                target_col = st.selectbox(
                                    "Select target variable",
                                    df.select_dtypes(include=[np.number]).columns
                                )
                                models = ["Linear Regression", "Decision Tree", "Neural Network"]
                            else:
                                target_col = st.selectbox(
                                    "Select target variable",
                                    df.select_dtypes(exclude=[np.number]).columns
                                )
                                models = ["Logistic Regression", "Decision Tree", "Neural Network"]
                            
                            selected_model = st.selectbox("Choose model:", models)
                            
                            feature_cols = st.multiselect(
                                "Select feature columns",
                                [col for col in df.columns if col != target_col],
                                default=[col for col in df.columns if col != target_col]
                            )
                            
                            if st.button(f"Apply {selected_model}"):
                                if feature_cols:
                                    with st.spinner(f"Applying {selected_model}..."):
                                        results = apply_model(
                                            df[feature_cols + [target_col]], 
                                            selected_model, 
                                            target_column=target_col
                                        )
                                        if results:
                                            if model_category == "Regression":
                                                show_regression_results(selected_model, results)
                                            else:
                                                show_classification_results(selected_model, results)
                                else:
                                    st.warning("Please select at least one feature column")

                # Download processed data
                if st.button("💾 Download Processed Data", key="dl_data"):
                    tmp_download_link = download_link(df, "processed_data.csv", "Click here to download")
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
            
            with col2:
                tab1, tab2 = st.tabs(["📊 Visualization", "🔢 Data Preview"])
                
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
                
                with tab2:
                    st.markdown("### 🔢 Data Preview")
                    st.dataframe(df)

def show_dim_reduction_results(model_type, results, n_components):
    """Display dimensionality reduction results."""
    st.markdown(f"#### {model_type} Results")
    
    # Show explained variance for PCA
    if model_type == "PCA" and 'explained_variance' in results:
        exp_var = results['explained_variance']
        cum_exp_var = np.cumsum(exp_var)
        
        fig_var = px.line(
            y=cum_exp_var,
            title="Cumulative Explained Variance Ratio",
            labels={"index": "Number of Components", "y": "Cumulative Explained Variance"},
            template="plotly_dark"
        )
    if results['embeddings'] is not None:
        if n_components >= 2:
            # Define column names based on model type
            if model_type == "t-SNE":
                x_col, y_col = "TSNE1", "TSNE2"
            elif model_type == "UMAP":
                x_col, y_col = "UMAP1", "UMAP2"
            else:  # PCA
                x_col, y_col = "PC1", "PC2"
            
            fig_2d = px.scatter(
                results['data'],
                x=x_col,
                y=y_col,
                title=f"{model_type} 2D Projection",
                template="plotly_dark"
            )
            st.plotly_chart(fig_2d)
        
        if n_components >= 3:
            # Define column names for 3D plot
            if model_type == "t-SNE":
                x_col, y_col, z_col = "TSNE1", "TSNE2", "TSNE3"
            elif model_type == "UMAP":
                x_col, y_col, z_col = "UMAP1", "UMAP2", "UMAP3"
            else:  # PCA
                x_col, y_col, z_col = "PC1", "PC2", "PC3"
            
            fig_3d = px.scatter_3d(
                results['data'],
                x=x_col,
                y=y_col,
                z=z_col,
                title=f"{model_type} 3D Projection",
                template="plotly_dark"
            )
            st.plotly_chart(fig_3d)
    
    # Add download button
    if st.button(f"Download {model_type} Results"):
        tmp_download_link = download_link(
            results['data'],
            f"{model_type.lower()}_results.csv",
            "Click here to download"
        )
        st.markdown(tmp_download_link, unsafe_allow_html=True)

def show_clustering_results(results):
    """Display clustering results."""
    st.markdown("#### Clustering Results")
    
    if 'data' in results and 'Cluster' in results['data'].columns:
        fig = px.scatter(
            results['data'],
            x=results['data'].select_dtypes(include=[np.number]).columns[0],
            y=results['data'].select_dtypes(include=[np.number]).columns[1],
            color='Cluster',
            title="Clustering Results",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
        
        # Show cluster distribution
        st.write("Cluster Distribution:")
        st.write(results['data']['Cluster'].value_counts())

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

if __name__ == '__main__':
    main()