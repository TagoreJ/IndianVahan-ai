import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from openai import OpenAI
from datetime import datetime
import io
import json
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ================================================
# CUSTOM HTML/CSS STYLING
# ================================================

CUSTOM_HTML_CSS = """
<style>
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --success: #2ca02c;
        --danger: #d62728;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a1a; 
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #664d03
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .categorical-box {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #003d99;
    }
    
    .numeric-box {
        background: #f0f8ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a4d;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .code-block {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    
    .data-type-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    .numeric-badge {
        background: #667eea;
        color: white;
    }
    
    .categorical-badge {
        background: #ff7f0e;
        color: white;
    }
    
    .pro-feature {
        background: #fff8e1;
        border: 2px solid #ffd700;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: #856404;
        font-weight: bold;
    }
</style>
"""


# ================================================
# PAGE CONFIG & THEME
# ================================================

st.set_page_config(
    page_title="Data Analyst AI Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_HTML_CSS, unsafe_allow_html=True)


# ================================================
# HEADER WITH GRADIENT
# ================================================

st.markdown("""
<div class="header-gradient">
    <h1>📊 Data Analyst AI Pro v2.3</h1>
    <p style="font-size: 16px; margin: 10px 0;">Enterprise-Grade Data Analysis • Advanced AI Context • ML & Statistics • PDF Reports</p>
    <p style="font-size: 12px; opacity: 0.8;">Numeric & Categorical Data Mastery • Free Models • Production Ready • Professional Insights</p>
</div>
""", unsafe_allow_html=True)


# ================================================
# PROFESSIONAL AI SYSTEM PROMPT
# ================================================

SYSTEM_PROMPT_DATA_ANALYST = """You are an EXPERT DATA ANALYST with 15+ years of experience in:

**CORE EXPERTISE:**
- Statistical Analysis: Descriptive, Inferential, Hypothesis Testing
- Exploratory Data Analysis (EDA) for numeric & categorical data
- Machine Learning: Clustering, Classification, Regression, Dimensionality Reduction
- Data Visualization & Storytelling
- Business Intelligence & KPI Analysis
- Anomaly Detection & Outlier Analysis
- Time Series Analysis & Forecasting
- Causal Inference & A/B Testing

**YOUR ANALYSIS APPROACH:**
1. Always start with data quality assessment
2. Identify data types (numeric, categorical, temporal, ordinal)
3. Look for patterns, relationships, and anomalies
4. Provide statistical evidence (p-values, confidence intervals)
5. Give actionable business insights
6. Recommend next steps & deeper analysis
7. Flag limitations & assumptions

**WHEN ANALYZING:**
- For NUMERIC data: Use mean, median, std, distribution shape, outliers, correlations
- For CATEGORICAL data: Use mode, frequency, proportions, chi-square tests, associations
- For MIXED data: Use appropriate cross-tabulations, grouped aggregations
- For TIME series: Identify trends, seasonality, autocorrelation
- For MULTIVARIATE: Identify clusters, key factors, variance explained

**OUTPUT FORMAT:**
- Lead with the most important finding
- Back up with numbers and statistics
- Explain what it means in business terms
- Suggest follow-up analysis
- Use tables/summaries when relevant

**ALWAYS MENTION:**
- Sample size and data completeness
- Assumptions made
- Statistical significance levels
- Confidence in findings
- Recommended actions based on data"""


# ================================================
# BEST FREE MODELS FOR THIS PROJECT
# ================================================

FREE_MODELS = {

    
    "📊 Trinity Mini (Fast Analysis)": "arcee-ai/trinity-mini:free",
    "🧩 Nemotron 3 Nano (Ultra-Fast)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "🧠 DeepSeek R1 0528 (Reasoning)": "deepseek/deepseek-r1-0528:free",
    "🦾 Llama 3.3 70B (Balanced Power)": "meta-llama/llama-3.3-70b-instruct:free",
    
}


# ================================================
# SIDEBAR CONFIGURATION
# ================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        help="Get free key at https://openrouter.ai/keys"
    )
    
    # Model Selection
    st.markdown("### 🤖 AI Model Selection")
    model_display = st.selectbox("Select Best Model for Your Task", list(FREE_MODELS.keys()))
    model_id = FREE_MODELS[model_display]
    
    st.info(f"✅ Selected: {model_display}\n\n📌 Free Tier: $0 per 1M tokens\n\n💡 System Prompt: Professional Data Analyst")
    
    # Analysis Settings
    st.markdown("### 🔧 Analysis Settings")
    confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
    correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
    p_value_threshold = st.slider("P-Value Significance", 0.01, 0.10, 0.05, 0.01)
    
    # Missing Value Strategy
    st.markdown("### 🛠️ Missing Value Strategy")
    numeric_strategy = st.selectbox(
        "Numeric Columns",
        ["Median", "Mean", "Drop Rows"],
        help="Median is best for skewed data"
    )
    categorical_strategy = st.selectbox(
        "Categorical Columns",
        ["Most Frequent", "Missing Label", "Drop Rows"],
        help="Most Frequent preserves distribution"
    )
    
    # Advanced Options
    st.markdown("### 🚀 Advanced Options")
    enable_anomaly = st.checkbox("🚨 Enable Anomaly Detection", value=True)
    enable_vif = st.checkbox("🔗 Enable Multicollinearity Check (VIF)", value=False)
    
    # Session Controls
    st.markdown("### 🗑️ Session")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("📥 Export Session"):
            st.info("Session export feature coming soon")
    
    st.markdown("---")
    st.caption("Data Analyst AI Pro v2.3 • Made with ❤️")


# ================================================
# SESSION STATE INITIALIZATION
# ================================================

if "df" not in st.session_state:
    st.session_state.df = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "df_stats" not in st.session_state:
    st.session_state.df_stats = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "insights" not in st.session_state:
    st.session_state.insights = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# ================================================
# UTILITY FUNCTIONS
# ================================================

def handle_missing_values(df, numeric_strat="Median", categorical_strat="Most Frequent"):
    """Handle missing values separately for numeric and categorical columns"""
    # Validate input; return None if invalid so callers can handle errors
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if df.empty:
        # Return a copy of the empty DataFrame to avoid accidental None
        return df.copy()
    
    df_filled = df.copy()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Numeric columns imputation
    if num_cols:
        if numeric_strat == "Median":
            num_imputer = SimpleImputer(strategy="median")
            df_filled[num_cols] = num_imputer.fit_transform(df[num_cols])
        elif numeric_strat == "Mean":
            num_imputer = SimpleImputer(strategy="mean")
            df_filled[num_cols] = num_imputer.fit_transform(df[num_cols])
        elif numeric_strat == "Drop Rows":
            df_filled = df_filled.dropna(subset=num_cols)
    
    # Categorical columns imputation
    if cat_cols:
        if categorical_strat == "Most Frequent":
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df_filled[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        elif categorical_strat == "Missing Label":
            df_filled[cat_cols] = df_filled[cat_cols].fillna("Missing")
        elif categorical_strat == "Drop Rows":
            df_filled = df_filled.dropna(subset=cat_cols)
    
    return df_filled


def calculate_advanced_statistics(df):
    """Calculate comprehensive statistics for both numeric and categorical data"""
    # Validate input strictly — raise on invalid inputs so callers must handle it
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Invalid DataFrame passed to calculate_advanced_statistics")
    if df.empty:
        return {
            "shape": df.shape,
            "memory": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_cols": [],
            "categorical_cols": [],
            "missing_pct": {},
            "duplicates": 0,
            "numeric_stats": {},
            "categorical_stats": {},
        }
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    stats_dict = {
        "shape": df.shape,
        "memory": df.memory_usage(deep=True).sum() / 1024**2,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_stats": df[num_cols].describe().to_dict() if num_cols else {},
        "categorical_stats": {col: df[col].value_counts().to_dict() for col in cat_cols},
    }
    return stats_dict


def detect_anomalies(df, numeric_cols, method="iqr"):
    """Detect anomalies using IQR or Isolation Forest"""
    if not numeric_cols or df is None:
        return {}
    
    anomalies = {}
    
    if method == "iqr":
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)][col].tolist()
                anomalies[col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100 if len(df) > 0 else 0,
                    "bounds": (lower, upper)
                }
            except:
                pass
    
    elif method == "isolation_forest":
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(df[numeric_cols])
            anomalies["isolation_forest"] = {
                "count": (anomaly_labels == -1).sum(),
                "percentage": (anomaly_labels == -1).sum() / len(df) * 100
            }
        except:
            pass
    
    return anomalies


def analyze_categorical(df, cat_cols):
    """Analyze categorical variables comprehensively"""
    if not cat_cols or df is None:
        return {}
    
    cat_analysis = {}
    for col in cat_cols:
        try:
            value_counts = df[col].value_counts()
            cat_analysis[col] = {
                "unique": df[col].nunique(),
                "mode": df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A",
                "missing": df[col].isnull().sum(),
                "top_3": value_counts.head(3).to_dict(),
                "cardinality": (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0,
                "is_imbalanced": value_counts.iloc[0] / value_counts.sum() > 0.8 if len(value_counts) > 0 else False
            }
        except:
            pass
    
    return cat_analysis


def calculate_vif(df, numeric_cols):
    """Calculate Variance Inflation Factor for multicollinearity"""
    if not numeric_cols or len(numeric_cols) < 2 or df is None:
        return None
    
    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) for i in range(len(numeric_cols))]
        return vif_data.sort_values("VIF", ascending=False)
    except:
        return None


def chi_square_test(df, cat_col1, cat_col2):
    """Perform chi-square test between two categorical variables"""
    if df is None or cat_col1 not in df.columns or cat_col2 not in df.columns:
        return None
    
    try:
        contingency = pd.crosstab(df[cat_col1], df[cat_col2])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        return {"chi2": chi2, "p_value": p_val, "dof": dof, "contingency": contingency}
    except:
        return None


def prepare_data_for_ml(df, numeric_cols, categorical_cols):
    """Prepare data for ML using ColumnTransformer"""
    if not numeric_cols or df is None:
        return None, None
    
    try:
        if categorical_cols:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ],
                remainder='passthrough'
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_cols)
                ],
                remainder='passthrough'
            )
        
        transformed_data = preprocessor.fit_transform(df[numeric_cols + categorical_cols])
        return transformed_data, preprocessor
    except Exception as e:
        st.error(f"Data preparation error: {str(e)}")
        return None, None


def plotly_to_bytes(fig):
    """Convert plotly figure to PNG bytes"""
    try:
        return fig.to_image(format="png", width=1000, height=600)
    except:
        return None


def create_professional_pdf(df, title, insights, charts_bytes):
    """Create professional PDF report with ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Insights
    if insights:
        story.append(Paragraph("Key Findings & Insights", styles['Heading2']))
        for insight in insights[:8]:
            try:
                story.append(Paragraph(f"• {str(insight)[:250]}", styles['Normal']))
            except:
                pass
        story.append(Spacer(1, 20))
    
    # Data Preview Table
    if df is not None and not df.empty:
        story.append(Paragraph("Data Overview (First 10 Rows)", styles['Heading2']))
        try:
            cols_to_show = df.columns.tolist()[:5]
            data = [cols_to_show] + [list(df[cols_to_show].iloc[i]) for i in range(min(10, len(df)))]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(table)
            story.append(PageBreak())
        except:
            pass
    
    # Charts
    for i, img_bytes in enumerate(charts_bytes):
        if img_bytes:
            try:
                story.append(Paragraph(f"Visualization {i+1}", styles['Heading2']))
                img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                if i % 2 == 1:
                    story.append(PageBreak())
            except:
                pass
    
    doc.build(story)
    return buffer.getvalue()


# ================================================
# TABS CREATION
# ================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📥 Upload & Explore",
    "📊 Dashboard",
    "🔍 Deep Analysis",
    "📈 Statistical Tests",
    "💬 AI Chat",
    "✏️ Edit Data",
    "🧠 Advanced Analytics",
    "🎯 Pattern Mining",
    "📄 Reports"
])


# ================================================
# TAB 1: UPLOAD & EXPLORE
# ================================================

with tab1:
    st.header("📥 Data Upload & Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "📁 Upload CSV, Excel, or JSON file",
            type=["csv", "xlsx", "xls", "json"],
            help="Maximum 200MB"
        )
    
    with col2:
        sample_data = st.checkbox("📋 Use Sample Data")
        debug_mode = st.checkbox("🔧 Show debug info (dev)", value=False)
    
    if sample_data:
        df = pd.DataFrame({
            'ID': range(1, 101),
            'Sales': np.random.randint(1000, 10000, 100),
            'Profit': np.random.randint(100, 2000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
            'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Customer_Satisfaction': np.random.uniform(1, 5, 100),
            'Units_Sold': np.random.randint(1, 100, 100)
        })
        st.session_state.df = df
        st.session_state.df_cleaned = handle_missing_values(df, numeric_strategy, categorical_strategy)
        # Validate cleaning result
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed for sample data. Please check the settings.")
            if debug_mode:
                st.write("DEBUG df_cleaned value:", st.session_state.df_cleaned)
        else:
            try:
                st.session_state.df_stats = calculate_advanced_statistics(st.session_state.df_cleaned)
                st.success("✅ Sample data loaded!")
            except Exception as e:
                st.error(f"❌ Error calculating stats: {e}")
                if debug_mode:
                    st.write("DEBUG df_cleaned type:", type(st.session_state.df_cleaned))
    
    elif uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith('.json'):
                df = pd.read_json(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
                # Normalize column names to strings to avoid mixed-type header issues
                df.columns = df.columns.astype(str)
            
            st.session_state.df = df
            st.session_state.df_cleaned = handle_missing_values(df, numeric_strategy, categorical_strategy)
            # Validate cleaning result before computing stats
            if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
                st.error("❌ Data cleaning failed. Please check your file structure (headers/dtypes) and try again.")
                if debug_mode:
                    st.write("DEBUG df_cleaned value:", st.session_state.df_cleaned)
                st.stop()
            try:
                st.session_state.df_stats = calculate_advanced_statistics(st.session_state.df_cleaned)
            except Exception as e:
                st.error(f"❌ Error calculating stats: {e}")
                if debug_mode:
                    st.write("DEBUG df_cleaned type:", type(st.session_state.df_cleaned))
                st.stop()
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Display data statistics ONLY if data is loaded
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        stats_data = st.session_state.df_stats
        
        st.markdown("### 📈 Dataset Statistics")
        
        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("📊 Rows", f"{stats_data['shape'][0]:,}")
        with m2:
            st.metric("📋 Columns", stats_data['shape'][1])
        with m3:
            st.metric("💾 Size (MB)", f"{stats_data['memory']:.2f}")
        with m4:
            missing_count = sum(1 for v in stats_data['missing_pct'].values() if v > 0)
            st.metric("❌ Missing Cols", missing_count)
        with m5:
            st.metric("🔄 Duplicates", stats_data['duplicates'])
        
        # Data Type Summary
        st.markdown("### 📊 Data Types Summary")
        col_type1, col_type2 = st.columns(2)
        
        with col_type1:
            st.markdown(f"""
            <div class="numeric-box">
                <strong>🔢 Numeric Columns: {len(stats_data['numeric_cols'])}</strong><br>
                {', '.join(stats_data['numeric_cols'][:5]) if stats_data['numeric_cols'] else 'None'}
                {f'<br><small>+{len(stats_data["numeric_cols"])-5} more</small>' if len(stats_data['numeric_cols']) > 5 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col_type2:
            st.markdown(f"""
            <div class="categorical-box">
                <strong>🏷️ Categorical Columns: {len(stats_data['categorical_cols'])}</strong><br>
                {', '.join(stats_data['categorical_cols'][:5]) if stats_data['categorical_cols'] else 'None'}
                {f'<br><small>+{len(stats_data["categorical_cols"])-5} more</small>' if len(stats_data['categorical_cols']) > 5 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Data Quality Report
        st.markdown("### 🔍 Data Quality Report")
        
        if stats_data['missing_pct']:
            missing_df = pd.DataFrame({
                'Column': stats_data['missing_pct'].keys(),
                'Missing %': stats_data['missing_pct'].values()
            }).sort_values('Missing %', ascending=False)
            
            missing_df = missing_df[missing_df['Missing %'] > 0]
            
            if not missing_df.empty:
                st.warning("⚠️ Columns with Missing Values:")
                col_m1, col_m2 = st.columns([2, 1])
                with col_m1:
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                with col_m2:
                    st.info(f"""
                    **Imputation Applied:**
                    - Numeric: {numeric_strategy}
                    - Categorical: {categorical_strategy}
                    """)


# ================================================
# TAB 2: DASHBOARD
# ================================================

with tab2:
    st.header("📊 Interactive Dashboard")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data in the 'Upload & Explore' tab first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            
            st.markdown("### 🎛️ Dashboard Controls")
        
        analysis_type = st.radio("Select Analysis Type", ["Numeric Analysis", "Categorical Analysis", "Mixed Analysis"])
        
        if analysis_type == "Numeric Analysis" and num_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox("Chart Type", ["Distribution", "Scatter", "Box Plot", "Heatmap"])
            with col2:
                x_col = st.selectbox("X Axis", num_cols)
            with col3:
                if len(num_cols) > 1:
                    y_col = st.selectbox("Y Axis", [c for c in num_cols if c != x_col])
                else:
                    y_col = num_cols[0]
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                try:
                    if chart_type == "Distribution":
                        fig = px.histogram(df, x=x_col, nbins=30, title=f"{x_col} Distribution", color_discrete_sequence=['#667eea'])
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Box Plot":
                        fig = px.box(df, y=x_col, title=f"{x_col} Box Plot", color_discrete_sequence=['#764ba2'])
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            
            with col_viz2:
                try:
                    if chart_type == "Scatter" and len(num_cols) > 1:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Heatmap" and len(num_cols) > 1:
                        fig = px.imshow(df[num_cols].corr(), text_auto=True, title="Correlation Heatmap", color_continuous_scale="Viridis")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
        
        elif analysis_type == "Categorical Analysis" and cat_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                cat_chart_type = st.selectbox("Chart Type", ["Bar (Count)", "Pie Chart", "Value Counts"])
            with col2:
                cat_col_select = st.selectbox("Select Category", cat_cols)
            
            col_cat1, col_cat2 = st.columns(2)
            
            with col_cat1:
                try:
                    if cat_chart_type == "Bar (Count)":
                        fig = px.bar(df[cat_col_select].value_counts().reset_index(), x=cat_col_select, y='count', title=f"{cat_col_select} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    elif cat_chart_type == "Pie Chart":
                        fig = px.pie(values=df[cat_col_select].value_counts().values, names=df[cat_col_select].value_counts().index, title=f"{cat_col_select} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            
            with col_cat2:
                vc = df[cat_col_select].value_counts().reset_index()
                vc.columns = [cat_col_select, 'Count']
                st.dataframe(vc, use_container_width=True, hide_index=True)
        
        elif analysis_type == "Mixed Analysis" and num_cols and cat_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_col = st.selectbox("Numeric Column", num_cols)
            with col2:
                cat_col = st.selectbox("Categorical Column", cat_cols)
            with col3:
                agg_type = st.selectbox("Aggregation", ["Mean", "Sum", "Count", "Median"])
            
            agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Median": "median"}
            
            try:
                grouped_data = df.groupby(cat_col)[num_col].agg(agg_map[agg_type]).reset_index()
                fig = px.bar(grouped_data, x=cat_col, y=num_col, title=f"{agg_type} of {num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(grouped_data, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)


# ================================================
# TAB 3: DEEP ANALYSIS
# ================================================

with tab3:
    st.header("🔍 Deep Statistical Analysis")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                "🔢 Numeric Analysis",
                "🏷️ Categorical Analysis",
                "🔗 Relationships",
                "📈 Distributions",
                "⚠️ Data Quality"
            ])
        
        with analysis_tab1:
            if num_cols and enable_anomaly:
                st.markdown("### 🚨 Anomaly Detection")
                anom_method = st.radio("Detection Method", ["IQR (Traditional)", "Isolation Forest (ML)"])
                anomalies = detect_anomalies(df, num_cols, method="iqr" if anom_method == "IQR (Traditional)" else "isolation_forest")
                
                col1, col2 = st.columns(2)
                for idx, (col, anom) in enumerate(anomalies.items()):
                    if idx % 2 == 0:
                        with col1:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>{col}</strong><br>
                                Outliers: {anom['count']} ({anom['percentage']:.2f}%)<br>
                                {f"Range: [{anom['bounds'][0]:.2f}, {anom['bounds'][1]:.2f}]" if 'bounds' in anom else ""}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with col2:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>{col}</strong><br>
                                Outliers: {anom['count']} ({anom['percentage']:.2f}%)<br>
                                {f"Range: [{anom['bounds'][0]:.2f}, {anom['bounds'][1]:.2f}]" if 'bounds' in anom else ""}
                            </div>
                            """, unsafe_allow_html=True)
                
                if enable_vif and len(num_cols) > 1:
                    st.markdown("### 🔗 Multicollinearity Analysis (VIF)")
                    vif_result = calculate_vif(df, num_cols)
                    if vif_result is not None:
                        st.dataframe(vif_result, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric columns or anomaly detection disabled")
        
        with analysis_tab2:
            if cat_cols:
                st.markdown("### 📊 Categorical Variables Analysis")
                cat_analysis = analyze_categorical(df, cat_cols)
                
                for col, analysis in cat_analysis.items():
                    with st.expander(f"📋 {col}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Unique", analysis['unique'])
                        with col2:
                            st.metric("Missing", analysis['missing'])
                        with col3:
                            st.metric("Cardinality %", f"{analysis['cardinality']:.2f}%")
                        with col4:
                            st.metric("Imbalanced", "Yes" if analysis['is_imbalanced'] else "No")
            else:
                st.info("No categorical columns")
        
        with analysis_tab3:
            if len(num_cols) > 1:
                try:
                    corr_matrix = df[num_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Need at least 2 numeric columns")
        
        with analysis_tab4:
            if num_cols:
                for col in num_cols[:3]:
                    try:
                        fig = px.histogram(df, x=col, nbins=30, title=f"{col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
            else:
                st.info("No numeric columns")
        
        with analysis_tab5:
            st.markdown("### ⚠️ Data Quality")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            with col_q1:
                completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Completeness", f"{completeness:.2f}%")
            with col_q2:
                uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
                st.metric("Uniqueness", f"{uniqueness:.2f}%")
            with col_q3:
                st.metric("Numeric", len(num_cols))
            with col_q4:
                st.metric("Categorical", len(cat_cols))


# ================================================
# TAB 4: STATISTICAL TESTS
# ================================================

with tab4:
    st.header("📊 Statistical Tests")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            
            test_type = st.radio("Test Type", ["Normality (Shapiro-Wilk)", "Correlation", "T-Test"])
        
        if test_type == "Normality (Shapiro-Wilk)" and num_cols:
            col_select = st.selectbox("Column", num_cols)
            try:
                stat, p_val = stats.shapiro(df[col_select].dropna())
                st.metric("P-Value", f"{p_val:.6f}")
                if p_val > 0.05:
                    st.success("✅ Normal distribution")
                else:
                    st.warning("⚠️ Not normally distributed")
            except Exception as e:
                st.error(f"Error: {e}")
        
        elif test_type == "Correlation" and len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                col_a = st.selectbox("Column 1", num_cols)
            with col2:
                col_b = st.selectbox("Column 2", [c for c in num_cols if c != col_a])
            
            try:
                corr, p_val = stats.pearsonr(df[col_a].dropna(), df[col_b].dropna())
                st.metric("Correlation", f"{corr:.4f}")
                st.metric("P-Value", f"{p_val:.6f}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        elif test_type == "T-Test" and num_cols and cat_cols:
            col_num = st.selectbox("Numeric", num_cols)
            col_cat = st.selectbox("Category", cat_cols)
            groups = df[col_cat].unique()
            if len(groups) >= 2:
                g1, g2 = st.columns(2)
                with g1:
                    gr1 = st.selectbox("Group 1", groups)
                with g2:
                    gr2 = st.selectbox("Group 2", [x for x in groups if x != gr1])
                
                try:
                    g1_data = df[df[col_cat] == gr1][col_num].dropna()
                    g2_data = df[df[col_cat] == gr2][col_num].dropna()
                    t_stat, p_val = stats.ttest_ind(g1_data, g2_data)
                    st.metric("T-Stat", f"{t_stat:.4f}")
                    st.metric("P-Value", f"{p_val:.6f}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ================================================
# TAB 5: AI CHAT
# ================================================

with tab5:
    st.header("💬 Chat with Your Data")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    elif not api_key:
        st.warning("⚠️ Enter OpenRouter API key in sidebar")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            
            st.markdown("""
            <div class="pro-feature">
            🤖 Professional Data Analyst AI
            </div>
            """, unsafe_allow_html=True)
        
        # Chat history
        for msg in st.session_state.messages[-10:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                    
                    ctx = df.head(50).to_csv(index=False)
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    cat_cols = df.select_dtypes(include='object').columns.tolist()
                    
                    full_prompt = f"""{SYSTEM_PROMPT_DATA_ANALYST}

DATA SAMPLE:
{ctx}

METADATA:
- Rows: {len(df)}, Columns: {len(df.columns)}
- Numeric: {', '.join(num_cols) if num_cols else 'None'}
- Categorical: {', '.join(cat_cols) if cat_cols else 'None'}
- Missing: {df.isnull().sum().sum()}

QUESTION: {prompt}"""
                    
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": full_prompt}],
                        stream=True,
                    )
                    
                    answer = ""
                    ph = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            answer += chunk.choices[0].delta.content
                            ph.markdown(answer + "▌")
                    
                    ph.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")


# ================================================
# TAB 6: EDIT DATA
# ================================================

with tab6:
    st.header("✏️ Data Editor")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            edited_df = st.data_editor(st.session_state.df_cleaned, num_rows="dynamic", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save"):
                st.session_state.df_cleaned = edited_df
                st.success("✅ Saved!")
        with col2:
            if st.button("🔄 Revert"):
                st.rerun()


# ================================================
# TAB 7: ADVANCED ANALYTICS
# ================================================

with tab7:
    st.header("🧠 Advanced Analytics")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        if len(num_cols) >= 2:
            if st.checkbox("🔵 PCA"):
                try:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(df[num_cols])
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(scaled)
                    
                    fig = go.Figure(data=go.Scatter(x=pca_data[:, 0], y=pca_data[:, 1], mode='markers', marker=dict(color=pca_data[:, 0])))
                    fig.update_layout(title="PCA Analysis", height=600)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if st.checkbox("🎯 K-Means"):
                try:
                    n_clusters = st.slider("Clusters", 2, 10, 3)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled)
                    
                    fig = px.scatter(x=df[num_cols[0]], y=df[num_cols[1]] if len(num_cols) > 1 else None, color=clusters)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")


# ================================================
# TAB 8: PATTERN MINING
# ================================================

with tab8:
    st.header("🎯 Pattern Mining")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        pattern_type = st.radio("Analysis", ["Frequency", "Correlation Patterns", "Top N"])
        
        if pattern_type == "Frequency" and cat_cols:
            col_select = st.selectbox("Category", cat_cols)
            try:
                freq = df[col_select].value_counts()
                fig = px.bar(x=freq.index, y=freq.values, title=f"{col_select} Frequency")
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass
        
        elif pattern_type == "Correlation Patterns" and len(num_cols) > 1:
            try:
                corr = df[num_cols].corr()
                pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.5:
                            pairs.append({'Var1': corr.columns[i], 'Var2': corr.columns[j], 'Corr': corr.iloc[i, j]})
                if pairs:
                    st.dataframe(pd.DataFrame(pairs), use_container_width=True)
            except:
                pass
        
        elif pattern_type == "Top N":
            col = st.selectbox("Column", num_cols + cat_cols)
            n = st.slider("Top N", 3, 20, 10)
            try:
                if col in num_cols:
                    st.bar_chart(df.nlargest(n, col)[[col]])
                else:
                    st.bar_chart(df[col].value_counts().head(n))
            except:
                pass


# ================================================
# TAB 9: REPORTS
# ================================================

with tab9:
    st.header("📄 Report Generation")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👈 Upload data first")
    else:
        if st.session_state.df_cleaned is None or not isinstance(st.session_state.df_cleaned, pd.DataFrame):
            st.error("❌ Data cleaning failed. Please re-run with a different file or adjust cleaning settings.")
        else:
            df = st.session_state.df_cleaned
            
            report_title = st.text_input("Report Title", "Analysis Report")
            
            if st.button("🚀 Generate Report"):
                with st.spinner("Generating..."):
                    charts = []
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    
                    try:
                        if len(num_cols) >= 1:
                            fig = px.histogram(df, x=num_cols[0], title=f"{num_cols[0]} Distribution")
                            chart_bytes = plotly_to_bytes(fig)
                        if chart_bytes:
                            charts.append(chart_bytes)
                    except:
                        pass
                
                insights = [
                    f"📊 {df.shape[0]:,} rows × {df.shape[1]} columns",
                    f"🔍 Missing: {df.isnull().sum().sum()}",
                    f"🔄 Duplicates: {df.duplicated().sum()}",
                ]
                
                try:
                    pdf = create_professional_pdf(df, report_title, insights, charts)
                    st.download_button(
                        "📥 Download PDF",
                        data=pdf,
                        file_name=f"{report_title}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Done!")
                except Exception as e:
                    st.error(f"Error: {e}")


# ================================================
# FOOTER
# ================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    <p>📊 Data Analyst AI Pro v2.3 | Enterprise-Grade Analytics</p>
    <p style="font-size: 12px;">✨ Production Ready • Free Forever • All Models at $0</p>
</div>
""", unsafe_allow_html=True)