import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

# Page configuration
st.set_page_config(
    page_title="Simple Linear Model Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for bigger and darker data editor
st.markdown("""
    <style>
    /* Data editor cells */
    .stDataFrame div[data-testid="stDataFrameResizable"] div[role="gridcell"] {
        font-size: 16px !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Data editor header cells */
    .stDataFrame div[data-testid="stDataFrameResizable"] div[role="columnheader"] {
        font-size: 17px !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Data editor input fields when editing */
    .stDataFrame input {
        font-size: 16px !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Row index cells */
    .stDataFrame div[data-testid="stDataFrameResizable"] div[data-testid="StyledDataFrameRowIndex"] {
        font-size: 15px !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Make the entire data editor container text darker */
    div[data-testid="stDataFrame"] {
        color: #000000 !important;
    }
    
    /* Column resize handles */
    .dvn-scroller {
        color: #000000 !important;
    }
    
    /* Selected cells */
    .stDataFrame div[data-testid="stDataFrameResizable"] div[aria-selected="true"] {
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
""")


# Initialize session state
if 'df' not in st.session_state:
    # Start with empty dataframe
    st.session_state.df = pd.DataFrame({
        'x': [0.4, 0.75, 1, 1.4, 1.7],
        'y_actual': [8, 13, 28, 32, 45],
        'y_model': [pd.NA] * 5,
        'error': [pd.NA] * 5,
        'error_squared': [pd.NA] * 5
    })

if 'x_name' not in st.session_state:
    st.session_state.x_name = 'x'
    
if 'y_name' not in st.session_state:
    st.session_state.y_name = 'y'

if 'slope' not in st.session_state:
    st.session_state.slope = 17.0
    
if 'intercept' not in st.session_state:
    st.session_state.intercept = 7.0

# Helper functions
def calculate_predictions(df, slope, intercept):
    """Calculate model predictions and errors"""
    df = df.copy()
    df['y_model'] = slope * df['x'] + intercept
    df['error'] = df['y_actual'] - df['y_model']
    df['error_squared'] = df['error'] ** 2
    return df

def calculate_metrics(df):
    """Calculate regression metrics"""
    if len(df) == 0:
        return None, None, None
    
    mse = np.mean(df['error_squared'])
    rmse = np.sqrt(mse)
    
    # Calculate R²
    r2 = r2_score(df['y_actual'], df['y_model'])
    
    return mse, rmse, r2

def find_best_fit(df):
    """Calculate optimal slope and intercept using OLS"""
    if len(df) < 2:
        return 1.0, 0.0
    
    X = df['x'].values.reshape(-1, 1)
    y = df['y_actual'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return float(model.coef_[0]), float(model.intercept_)

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
st.session_state.slope = slope
st.session_state.intercept = intercept

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
        st.session_state.df = calculate_predictions(st.session_state.df, slope, intercept)
        st.rerun()

with col2:
    if st.button("🎯 Find Parameters with Linear Regression", help="Use least squares regression to find optimal parameters"):
        optimal_slope, optimal_intercept = find_best_fit(st.session_state.df)
        st.session_state.slope = optimal_slope
        st.session_state.intercept = optimal_intercept
        st.session_state.df = calculate_predictions(st.session_state.df, optimal_slope, optimal_intercept)
        st.rerun()

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
    'error_squared': 'Error²'
}

# Data editor with enhanced styling
edited_df = st.data_editor(
    st.session_state.df,
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
        )
    },
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)

# Update session state with edited data - be more careful about data preservation
if edited_df is not None and len(edited_df) >= 0:
    # When new data is pasted, preserve only x and y_actual, reset calculated columns
    if not edited_df.equals(st.session_state.df):
        # Ensure we have the required columns
        new_df = edited_df.copy()
        
        # If user pasted data and we're missing calculated columns, add them
        required_cols = ['x', 'y_actual', 'y_model', 'error', 'error_squared']
        for col in required_cols:
            if col not in new_df.columns:
                if col in ['y_model', 'error', 'error_squared']:
                    new_df[col] = 0.0
                else:
                    new_df[col] = None
        
        # Reset calculated columns when x or y_actual changes
        new_df['y_model'] = 0.0
        new_df['error'] = 0.0 
        new_df['error_squared'] = 0.0
        
        # Remove any rows with NaN in x or y_actual
        new_df = new_df.dropna(subset=['x', 'y_actual'])
        
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
        mse, rmse, r2 = calculate_metrics(st.session_state.df)
        
        if mse is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{mse:.3f}", help="Average of squared errors - lower is better")
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}", help="Square root of MSE - in original units")
                st.metric("R² (Coefficient of Determination)", f"{r2:.3f}", 
                         help="Proportion of variance explained (0-1, higher is better).\nNegative means the model is extremely poor fit.⚠️NOTE: R² is ONLY MEANINGFUL for LINEAR MODELS.⚠️")
            
            with col2:
                st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{actual},i} - y_{\text{model},i})^2")
                st.latex(r"\text{RMSE} = \sqrt{\text{MSE}}")
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

# Mathematical concepts section
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

# Footer with educational content
st.markdown("---")
st.markdown("""
**🎓 Learning Objectives:**
- Understand the relationship between slope, intercept, and model predictions
- Visualize how errors contribute to model performance metrics
- Compare manual parameter tuning vs. optimal least squares solution
- Interpret regression metrics (MSE, RMSE, R²) in context

**💡 Try This:**
1. Load sample data and find the best fit
2. Manually adjust parameters and observe how metrics change  
3. Add outliers and see their impact on the model
4. Compare different datasets to understand when linear regression works well
""")
