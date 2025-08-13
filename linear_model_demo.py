import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import shapiro
import io

# Page configuration
st.set_page_config(
    page_title="Simple Linear Model Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Title and description
st.title("📊 Simple Linear Model Explorer")
st.markdown("""
**Interactive Learning Tool for Linear Models**

This application helps you understand how simple linear regression works by allowing you to:
- Input your own data or use sample datasets
- Manually adjust model parameters (slope and intercept)
- Visualize how changes affect predictions and errors
- Find the optimal model using least squares regression
- Calculate and understand key performance metrics
- Perform inferential analysis on the regression results
""")


# Initialize session state
if 'df' not in st.session_state:
    # Start with empty dataframe
    st.session_state.df = pd.DataFrame({
        'x': [0.4, 0.75, 1, 1.4, 1.7],
        'y_actual': [8, 13, 28, 32, 45],
        'y_model': [pd.NA] * 5,
        'error': [pd.NA] * 5,
        'error_squared': [pd.NA] * 5,
        'abs_error': [pd.NA] * 5,
        'pct_abs_error': [pd.NA] * 5
    })

if 'x_name' not in st.session_state:
    st.session_state.x_name = 'x'
    
if 'y_name' not in st.session_state:
    st.session_state.y_name = 'y'

if 'slope' not in st.session_state:
    st.session_state.slope = 17.0
    
if 'intercept' not in st.session_state:
    st.session_state.intercept = 7.0

if 'is_optimal_fit' not in st.session_state:
    st.session_state.is_optimal_fit = False

# Helper functions
def calculate_predictions(df, slope, intercept):
    """Calculate model predictions and errors - improved version"""
    if df is None or len(df) == 0:
        return pd.DataFrame({
            'x': [],
            'y_actual': [],
            'y_model': [],
            'error': [],
            'error_squared': [],
            'abs_error': [],
            'pct_abs_error': []

        })
    
    df = df.copy()
    
    # Ensure we have the required columns
    if 'x' not in df.columns:
        df['x'] = np.nan
    if 'y_actual' not in df.columns:
        df['y_actual'] = np.nan
    
    # Calculate predictions only for rows with valid x values
    valid_mask = df['x'].notna()
    
    # Initialize calculated columns
    df['y_model'] = np.nan
    df['error'] = np.nan
    df['error_squared'] = np.nan
    df['abs_error'] = np.nan
    df['pct_abs_error'] = np.nan

    
    # Calculate for valid rows only
    if valid_mask.any():
        df.loc[valid_mask, 'y_model'] = slope * df.loc[valid_mask, 'x'] + intercept
        
        # Only calculate errors where we have both actual and predicted values
        error_mask = valid_mask & df['y_actual'].notna()
        if error_mask.any():
            df.loc[error_mask, 'error'] = df.loc[error_mask, 'y_actual'] - df.loc[error_mask, 'y_model']
            df.loc[error_mask, 'error_squared'] = df.loc[error_mask, 'error'] ** 2
            df.loc[error_mask, 'abs_error'] = np.abs(df.loc[error_mask, 'error'])
            df.loc[error_mask, 'pct_abs_error'] = df.loc[error_mask, 'abs_error'] / df.loc[error_mask, 'y_actual']
    
    # Fill NaN values with 0 for display purposes (but preserve NaN for actual missing data)
    display_cols = ['y_model', 'error', 'error_squared', 'abs_error', 'pct_abs_error']
    for col in display_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    return df

def calculate_metrics(df):
    """Calculate regression metrics"""
    if len(df) == 0:
        return None, None, None
    
    mse = np.mean(df['error_squared'])
    rmse = np.sqrt(mse)
    mae = np.mean(df['abs_error'])
    mape = np.mean(df['pct_abs_error'])
    
    # Calculate R²
    r2 = r2_score(df['y_actual'], df['y_model'])
    
    return mse, rmse, r2, mae, mape

def find_best_fit(df):
    """Calculate optimal slope and intercept using OLS"""
    # Clean the data first
    clean_df = df.dropna(subset=['x', 'y_actual'])
    
    if len(clean_df) < 2:
        return 1.0, 0.0
    
    X = clean_df['x'].values.reshape(-1, 1)
    y = clean_df['y_actual'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return float(model.coef_[0]), float(model.intercept_)
    

def calculate_regression_statistics(df):
    """Calculate detailed regression statistics for inferential analysis"""
    if len(df) < 3:
        return None
    
    n = len(df)
    x = df['x'].values
    y = df['y_actual'].values
    y_pred = df['y_model'].values
    residuals = df['error'].values
    
    # Mean values
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Sum of squares
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_res = np.sum(residuals ** 2)  # Residual sum of squares
    
    # Degrees of freedom
    df_total = n - 1
    df_regression = 1
    df_residual = n - 2
    
    # Mean squares
    ms_regression = np.sum((y_pred - y_mean) ** 2) / df_regression
    ms_residual = ss_res / df_residual
    
    # Standard errors
    se_residual = np.sqrt(ms_residual)
    se_slope = se_residual / np.sqrt(ss_xx)
    se_intercept = se_residual * np.sqrt(1/n + (x_mean**2)/ss_xx)
    
    # Current slope and intercept
    slope = st.session_state.slope
    intercept = st.session_state.intercept
    
    # t-statistics and p-values
    t_slope = slope / se_slope if se_slope > 0 else np.inf
    t_intercept = intercept / se_intercept if se_intercept > 0 else np.inf
    
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df_residual))
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df_residual))
    
    # F-statistic for overall regression
    f_statistic = ms_regression / ms_residual if ms_residual > 0 else np.inf
    p_f = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)
    
    # R-squared
    r_squared = r2_score(y, y_pred)
    
    return {
        'n': n,
        'df_residual': df_residual,
        'se_slope': se_slope,
        'se_intercept': se_intercept,
        'se_residual': se_residual,
        't_slope': t_slope,
        't_intercept': t_intercept,
        'p_slope': p_slope,
        'p_intercept': p_intercept,
        'f_statistic': f_statistic,
        'p_f': p_f,
        'r_squared': r_squared,
        'residuals': residuals,
    }

def load_sample_data(dataset_name):
    """Load predefined sample datasets - removed but keeping function for compatibility"""
    return pd.DataFrame({
        'x': [0.4, 0.75, 1, 1.4, 1.7],
        'y_actual': [8, 13, 28, 32, 45],
        'y_model': [pd.NA] * 5,
        'error': [pd.NA] * 5,
        'error_squared': [pd.NA] * 5
    })

# Sidebar controls
st.sidebar.header("🎛️ Model Controls")

# Variable naming
st.sidebar.subheader("Variable Names\n(DO NOT USE UNDERSCORES!)")
new_x_name = st.sidebar.text_input("X-axis variable name", value=st.session_state.x_name, help="Name for your independent variable")
new_y_name = st.sidebar.text_input("Y-axis variable name", value=st.session_state.y_name, help="Name for your dependent variable")

# Update names if changed
if new_x_name != st.session_state.x_name or new_y_name != st.session_state.y_name:
    st.session_state.x_name = new_x_name
    st.session_state.y_name = new_y_name
    st.rerun()

# Model parameters
st.sidebar.subheader("Linear Model Parameters")

intercept = st.sidebar.number_input("Intercept", value=st.session_state.intercept, step=0.1, format="%.3f",
                                   help="Value of Y when X equals zero")

slope = st.sidebar.number_input("Slope", value=st.session_state.slope, step=0.1, format="%.3f", 
                                help="How much Y increases for each unit increase in X")

# Update session state
if slope != st.session_state.slope or intercept != st.session_state.intercept:
    st.session_state.slope = slope
    st.session_state.intercept = intercept
    st.session_state.is_optimal_fit = False  # Manual adjustment means not optimal

# Display current equation with MathJax
st.sidebar.markdown(f"**Current Model:**")

# Format the equation with proper mathematical notation
x_var = st.session_state.x_name
y_var = st.session_state.y_name

# Handle sign for slope
if slope >= 0:
    slope_str = f" + {slope:.3f}"
else:
    slope_str = f" - {abs(slope):.3f}"

# Create the LaTeX equation with proper text formatting
latex_equation = f"\\text{{{y_var}}}_{r'{\text{model}}'} = {intercept:.3f}{slope_str} \\cdot \\text{{{x_var}}}"

# Display with LaTeX rendering
st.sidebar.latex(latex_equation)

# Also show the general form
st.sidebar.markdown("**General Form:**")
st.sidebar.latex(r"y_{\text{model}} = \beta_0 + \beta_1 x")
st.sidebar.markdown(f"Current values: β₀ = {intercept:.3f} (intercept), β₁ = {slope:.3f} (slope)")


# Action buttons
st.sidebar.subheader("Actions")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("⚙️ Calculate Predictions", help="Update predictions based on current slope and intercept"):
        # Don't reset the dataframe, just update predictions
        current_df = st.session_state.df.copy()
        st.session_state.df = calculate_predictions(current_df, st.session_state.slope, st.session_state.intercept)
        st.rerun()

with col2:
    if st.button("🎯 Find Parameters with Linear Regression", help="Use linear regression to find least square parameters"):
        # Calculate optimal parameters without resetting data
        current_df = st.session_state.df.copy()
        if len(current_df.dropna(subset=['x', 'y_actual'])) >= 2:
            optimal_slope, optimal_intercept = find_best_fit(current_df)
            st.session_state.slope = optimal_slope
            st.session_state.intercept = optimal_intercept
            st.session_state.is_optimal_fit = True
            # Update predictions with new parameters
            st.session_state.df = calculate_predictions(current_df, optimal_slope, optimal_intercept)
            st.rerun()
        else:
            st.sidebar.error("Need at least 2 data points for regression")

# Data input section - MOVED TO TOP OF MAIN CONTENT
st.subheader("📝 Data Input")

# Instructions
with st.expander("📋 How to use the data editor", expanded=False):
    st.markdown("""
    **Adding/Editing Data:**
    - Click on any cell to edit values directly
    - Use the + button to add new rows
    - Delete rows by selecting them and using the delete key
    - **For copy/paste:** Paste your data into the x and y_actual columns, then click "Calculate Predictions"
    - The app will automatically reset predictions when you change the input data
    
    **Tips:**
    - Only x and y_actual columns are editable - other columns are calculated automatically
    - After pasting or editing data, use "Calculate Predictions" to update the model
    - Try different slopes and intercepts to see how they affect the fit
    - Use 'Find Best Fit' to see the optimal linear model
    - Watch how the metrics change as you modify the model
    """)

# Create column names with proper labels
display_columns = {
    'x': st.session_state.x_name,
    'y_actual': f'{st.session_state.y_name} (Actual)',
    'y_model': f'{st.session_state.y_name} (Model)',
    'error': 'Error',
    'error_squared': 'Error²',
    'abs_error': '|Error|',
    'pct_abs_error': '|% Error|'
}

# Replace your existing data editor section with this fixed version:

# Data editor with enhanced styling
# Data editor with enhanced styling
edited_df = st.data_editor(
    st.session_state.df.reset_index(drop=True),  # Reset index to avoid index column issues
    column_config={
        'x': st.column_config.NumberColumn(
            st.session_state.x_name,
            help=f"Independent variable values",
            format="%.3f"
        ),
        'y_actual': st.column_config.NumberColumn(
            f'{st.session_state.y_name} (Actual)',
            help="Observed/actual values of the dependent variable",
            format="%.3f"
        ),
        'y_model': st.column_config.NumberColumn(
            f'{st.session_state.y_name} (Model)',
            help="Predicted values from the linear model",
            format="%.3f",
            disabled=True
        ),
        'error': st.column_config.NumberColumn(
            'Error',
            help="Difference between actual and predicted values",
            format="%.3f",
            disabled=True
        ),
        'error_squared': st.column_config.NumberColumn(
            'Error²',
            help="Squared errors used in MSE calculation",
            format="%.3f",
            disabled=True
        ),
        'abs_error': st.column_config.NumberColumn(
            '|Error|',
            help="Absolute of difference between actual and predicted values",
            format="%.3f",
            disabled=True
        ),
        'pct_abs_error': st.column_config.NumberColumn(
            '|% Error|',
            help="Absolute of difference between actual and predicted values divided by actual values",
            format="percent",
            disabled=True
        ),
    },
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor",
    hide_index=True  # Explicitly hide the index column
)

# Update session state with edited data - be more careful about data preservation
if edited_df is not None and len(edited_df) >= 0:
    # Check if the dataframe structure has changed (not just values)
    if not edited_df.equals(st.session_state.df):
        # Ensure we have the required columns
        new_df = edited_df.copy()
        
        # Remove any index columns that might have been added
        if 'index' in new_df.columns:
            new_df = new_df.drop('index', axis=1)
        
        # If user pasted data and we're missing calculated columns, add them
        required_cols = ['x', 'y_actual', 'y_model', 'error', 'error_squared']
        for col in required_cols:
            if col not in new_df.columns:
                if col in ['y_model', 'error', 'error_squared']:
                    new_df[col] = pd.NA
                else:
                    new_df[col] = pd.NA
        
        # Only reset calculated columns if x or y_actual actually changed
        old_data = st.session_state.df[['x', 'y_actual']].dropna()
        new_data = new_df[['x', 'y_actual']].dropna()
        
        if not old_data.equals(new_data):
            # Data has changed, reset calculated columns
            new_df['y_model'] = pd.NA
            new_df['error'] = pd.NA
            new_df['error_squared'] = pd.NA
            st.session_state.is_optimal_fit = False
        
        # Remove any rows with NaN in x or y_actual
        new_df = new_df.dropna(subset=['x', 'y_actual'])
        
        # Reset index to avoid issues
        new_df = new_df.reset_index(drop=True)
        
        st.session_state.df = new_df
            
# Download section
st.subheader("💾 Export Data")
col1, col2 = st.columns(2)

with col1:
    # Prepare data for download
    download_df = st.session_state.df.copy()
    download_df.columns = [display_columns.get(col, col) for col in download_df.columns]
    
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="linear_regression_data.csv",
        mime="text/csv",
        help="Download the current data with predictions and errors"
    )

with col2:
    if st.button("🔄 Reset", help="Reset to Sample Data"):
        st.session_state.df = pd.DataFrame({
        'x': [0.4, 0.75, 1, 1.4, 1.7],
        'y_actual': [8, 13, 28, 32, 45],
        'y_model': [pd.NA] * 5,
        'error': [pd.NA] * 5,
        'error_squared': [pd.NA] * 5
    })
        st.session_state.is_optimal_fit = False
        st.rerun()

# Main visualization section - NOW BELOW DATA INPUT
st.subheader("📊 Visualization")

# Create the plot
if len(st.session_state.df) > 0:
    fig = go.Figure()
    
    # Add scatter plot for actual data
    fig.add_trace(go.Scatter(
        x=st.session_state.df['x'],
        y=st.session_state.df['y_actual'],
        mode='markers',
        name=f'Actual {st.session_state.y_name}',
        marker=dict(color='blue', size=8),
        hovertemplate=f'{st.session_state.x_name}: %{{x}}<br>{st.session_state.y_name}: %{{y}}<extra></extra>'
    ))
    
    # Add model line
    if st.session_state.df['y_model'].sum() != 0:  # Only if predictions have been calculated
        fig.add_trace(go.Scatter(
            x=st.session_state.df['x'],
            y=st.session_state.df['y_model'],
            mode='lines+markers',
            name=f'Predicted {st.session_state.y_name}',
            line=dict(color='red', width=2),
            marker=dict(color='red', size=6),
            hovertemplate=f'{st.session_state.x_name}: %{{x}}<br>Predicted {st.session_state.y_name}: %{{y}}<extra></extra>'
        ))
        
        # Add error lines
        for i, row in st.session_state.df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['x'], row['x']],
                y=[row['y_actual'], row['y_model']],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=f"{st.session_state.y_name} vs {st.session_state.x_name}",
        xaxis_title=st.session_state.x_name,
        yaxis_title=st.session_state.y_name,
        height=500,
        hovermode='closest',
        xaxis=dict(
            titlefont=dict(color='black', size=18),
            tickfont=dict(color='black', size=15)
        ),
        yaxis=dict(
            titlefont=dict(color='black', size=18),
            tickfont=dict(color='black', size=15)
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=17,
            font_color="black",
            font_family="Arial"
        ),
        legend=dict(font_size=17,
                    font_color="black",
                    font_family="Arial")
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add some data points to see the visualization! Use the + button in the data table above.")

# Performance Metrics - Now in an expandable section
st.subheader("📈 Performance Metrics")
with st.expander("📊 View Model Performance Metrics", expanded=False):
    # Calculate and display metrics
    if len(st.session_state.df) > 0 and st.session_state.df['y_model'].sum() != 0:
        mse, rmse, r2, mae, mape = calculate_metrics(st.session_state.df)
        
        if mse is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{mse:.3f}", help="Average of squared errors - lower is better")
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}", help="Square root of MSE - in original units")
                st.metric("Mean Absolute Error (MAE)", f"{mae:.3f}", help="Average of absolute errors - less influenced by outliers")
                st.metric("R² (Coefficient of Determination)", f"{r2:.3f}", 
                         help="Proportion of variance explained (0-1, higher is better).\nNegative means the model is extremely poor fit.⚠️NOTE: R² is ONLY MEANINGFUL for LINEAR MODELS.⚠️")
            
            with col2:
                st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{actual},i} - y_{\text{model},i})^2")
                st.latex(r"\text{RMSE} = \sqrt{\text{MSE}}")
                st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_{\text{actual},i} - y_{\text{model},i}\right|")
                st.latex(r"R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_{\text{actual},i} - y_{\text{model},i})^2}{\sum(y_{\text{actual},i} - \bar{y}_{\text{actual}})^2}")

                        
            # Interpretation help
            st.markdown("**📖 R² Interpretation:**")
            if r2 >= 0:
                percentage = r2 * 100
                st.info(f"R² = {r2:.3f} means that **{percentage:.1f}%** of the variance in **{st.session_state.y_name}** is explained by **{st.session_state.x_name}**.")
            else:
                st.warning(f"R² = {r2:.3f} (negative) indicates that the model performs worse than a horizontal line at the mean of **{st.session_state.y_name}**.")
        else:
            st.info("Add data points and calculate predictions to see metrics")
    else:
        st.info("Add data points and calculate predictions to see metrics")

# Show inferential analytics if we have data and predictions
if (len(st.session_state.df) >= 3 and 
    st.session_state.df['y_model'].sum() != 0):
    
    # NEW SECTION: Residual Analytics
    st.subheader("Residual Analytics")
    # Residual Analysis Section (always available when we have predictions)
    with st.expander("📊 Residual Analysis", expanded=False):
        
        # Calculate residuals for any type of fit
        actual_y = st.session_state.df['y_actual'].values
        predicted_y = st.session_state.df['y_model'].values
        residuals = actual_y - predicted_y
        

        # Residual scatterplot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=predicted_y,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='red', size=8)
        ))
        
        # Add horizontal line at y=0
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="black")
        y_name = st.session_state.y_name

        fig_scatter.update_layout(
            title=f"Residuals vs Predicted {y_name}",
            xaxis_title=f"Predicted {y_name}",
            yaxis_title="Residuals",
            height=400,
            xaxis=dict(
                    titlefont=dict(color='black', size=18),
                    tickfont=dict(color='black', size=15)
                    ),
            yaxis=dict(
                        titlefont=dict(color='black', size=18),
                        tickfont=dict(color='black', size=15)
                    ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=17,
                font_color="black",
                font_family="Arial"
                ),
            legend=dict(font_size=17,
                        font_color="black",
                        font_family="Arial")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Statistical Tests Section (only for optimal fit)
    # NEW SECTION: Inferential Analytics
    st.subheader("Inferential Analytics")

    if st.session_state.is_optimal_fit:
        with st.expander("🧪 Statistical Tests", expanded=False):
            
            # Calculate regression statistics
            reg_stats = calculate_regression_statistics(st.session_state.df)
            
            if reg_stats is not None:
                residuals = reg_stats['residuals']
                n = reg_stats['n']
            
            # Create tabs for different analyses
            tab1, tab2 = st.tabs(["Normality Tests", "Parameter Tests"])
            
            with tab1:
                st.markdown("### 🧪 Normality Tests")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residual histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=max(5, min(10, n//2)),
                        name="Residuals",
                        marker_color="lightblue",
                        opacity=0.7
                    ))
                    fig_hist.update_layout(
                        title="Distribution of Residuals",
                        xaxis_title="Residuals",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=350,
                        xaxis=dict(
                                titlefont=dict(color='black', size=18),
                                tickfont=dict(color='black', size=15)
                                ),
                        yaxis=dict(
                                titlefont=dict(color='black', size=18),
                                tickfont=dict(color='black', size=15)
                                ),
                        hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=17,
                                font_color="black",
                                font_family="Arial"
                                ),
                        legend=dict(font_size=17,
                                font_color="black",
                                font_family="Arial")
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Q-Q plot
                    from scipy.stats import probplot
                    qq_data = probplot(residuals, dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode='markers',
                        name='Sample Quantiles',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Add theoretical line
                    line_x = np.array([qq_data[0][0].min(), qq_data[0][0].max()])
                    line_y = qq_data[1][1] + qq_data[1][0] * line_x
                    fig_qq.add_trace(go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name='Theoretical Normal',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title="Q-Q Plot (Normal Distribution)",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        height=350,
                        xaxis=dict(
                            titlefont=dict(color='black', size=18),
                            tickfont=dict(color='black', size=15)
                            ),
                        yaxis=dict(
                            titlefont=dict(color='black', size=18),
                            tickfont=dict(color='black', size=15)
                            ),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=17,
                            font_color="black",
                            font_family="Arial"
                            ),
                        legend=dict(font_size=17,
                                    font_color="black",
                                    font_family="Arial")
                            )
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Residual statistics
                st.markdown("### 📈 Residual Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{np.mean(residuals):.6f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(residuals, ddof=1):.4f}")
                with col3:
                    st.metric("Skewness", f"{stats.skew(residuals):.4f}")
                with col4:
                    st.metric("Kurtosis", f"{stats.kurtosis(residuals) + 3:.4f}")
                
                # Shapiro-Wilk interpretation
                st.markdown("### 🧪 Shapiro-Wilk Test Result")
                st.markdown("**H₀:** Residuals are normally distributed")
                st.markdown("**H₁:** Residuals are not normally distributed")
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = shapiro(residuals)
                
                st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.6f}")
                if shapiro_p < 0.05:
                    st.error("🔴 **Reject normality** (p < 0.05)")
                else:
                    st.success("🟢 **Cannot reject normality** (p ≥ 0.05)")
            
            with tab2:
                st.markdown("### 🧮 Parameter Tests")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Slope Test")
                    st.markdown("**H₀:** β₁ = 0 (no linear relationship)")
                    st.markdown("**H₁:** β₁ ≠ 0 (linear relationship exists)")
                    st.markdown("---")
                    st.metric("Parameter Value", f"{st.session_state.slope:.6f}")
                    st.metric("Standard Error", f"{reg_stats['se_slope']:.6f}")
                    st.metric("p-value", f"{reg_stats['p_slope']:.6f}")
                    
                    if reg_stats['p_slope'] < 0.05:
                        st.success("🟢 **Significant** (p < 0.05)")
                    else:
                        st.error("🔴 **Not significant** (p ≥ 0.05)")
                
                with col2:
                    st.markdown("#### 📊 Intercept Test")
                    st.markdown("**H₀:** β₀ = 0 (line passes through origin)")
                    st.markdown("**H₁:** β₀ ≠ 0 (line does not pass through origin)")
                    st.markdown("---")
                    st.metric("Parameter Value", f"{st.session_state.intercept:.6f}")
                    st.metric("Standard Error", f"{reg_stats['se_intercept']:.6f}")
                    st.metric("p-value", f"{reg_stats['p_intercept']:.6f}")
                    
                    if reg_stats['p_intercept'] < 0.05:
                        st.success("🟢 **Significant** (p < 0.05)")
                    else:
                        st.info("ℹ️ **Not significant** (p ≥ 0.05)")
                
                # F-test for overall model
                st.markdown("#### 🎯 Overall Model F-Test")
                st.markdown("**H₀:** All slope equals zero (β1 = β2 = ... = βn= 0)")
                st.markdown("**H₁:** At least one slope is nonzero")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("p-value", f"{reg_stats['p_f']:.6f}")
                with col2:
                    if reg_stats['p_f'] < 0.05:
                        st.success("🟢 **Significant** (p < 0.05)")
                    else:
                        st.error("🔴 **Not significant** (p ≥ 0.05)")

    else:
        with st.expander("🧪 Statistical Tests and Advanced Analysis", expanded=False):
            st.warning("⚠️ **Statistical tests are only available when using least square regression parameters.**")
            st.info("Click '🎯 Find Parameters with Linear Regression' to enable statistical tests.")

else:
    # Show empty expanders with appropriate messages
    with st.expander("📊 Residual Analysis", expanded=False):
        if len(st.session_state.df) < 3:
            st.warning("⚠️ **Need at least 3 data points for residual analysis.**")
        else:
            st.info("Calculate predictions first to enable residual analysis.")
    
    with st.expander("🧪 Statistical Tests and Advanced Analysis", expanded=False):
        if len(st.session_state.df) < 3:
            st.warning("⚠️ **Need at least 3 data points for statistical inference.**")
        else:
            st.info("Calculate predictions first, then use least square regression to enable statistical tests.")# Mathematical concepts section
st.subheader("📚 Mathematical Concepts")
with st.expander("🧮 Understanding Linear Regression Mathematics", expanded=False):
    st.markdown("**Linear Regression Model:**")
    st.latex(r"y_{\text{actual}} = y_{\text{model}} + \epsilon = \beta_0 + \beta_1 x + \epsilon")
    
    st.markdown("Where:")
    st.markdown("- $y_{\\text{model}}$ = predicted value")
    st.markdown("- $\\beta_0$ = intercept (y-value when x=0)")
    st.markdown("- $\\beta_1$ = slope (change in y per unit change in x)")  
    st.markdown("- $x$ = independent variable")
    st.markdown("- $\\epsilon$ = error term")
    
    st.markdown("**Least Squares Solution:**")
    st.latex(r"\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_{\text{actual},i} - \bar{y}_{\text{actual}})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}")
    st.latex(r"\beta_0 = \bar{y}_{\text{actual}} - \beta_1\bar{x}")
    
    st.markdown("**Error Calculation:**")
    st.latex(r"\epsilon_i = y_{\text{actual},i} - y_{\text{model},i} = y_{\text{actual},i} - (\beta_0 + \beta_1 x_i)")
    
    st.markdown("**Statistical Inference (when assumptions are met):**")
    st.latex(r"t_{\beta_1} = \frac{\hat{\beta_1} - 0}{SE(\hat{\beta_1})} \sim t_{n-2}")
    st.latex(r"F = \frac{MS_{regression}}{MS_{residual}} \sim F_{1,n-2}")

# Footer with educational content
st.markdown("---")
st.markdown("""
**🎓 Learning Objectives:**
- Understand the relationship between slope, intercept, and model predictions
- Visualize how errors contribute to model performance metrics
- Compare manual parameter tuning vs. optimal least squares solution
- Interpret regression metrics (MSE, RMSE, R²) in context
- Perform statistical inference on regression parameters
- Assess residual distribution

**💡 Try This:**
1. Load sample data and find the best fit
2. Manually adjust parameters and observe how metrics change  
3. Add outliers and see their impact on the model
4. Compare different datasets to understand when linear regression works well
5. Use inferential analytics to test parameter significance
6. Examine residual distributions and interpret normality tests
7. Understand the relationship between sample size and test validity
""")
