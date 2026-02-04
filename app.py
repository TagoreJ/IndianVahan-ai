import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI
from openai import OpenAI
import os
import warnings

warnings.filterwarnings("ignore")

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="Vahan Data Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# STYLING (From main.py)
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
    
    .stApp {
        background-color: #f8f9fa;
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
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_HTML_CSS, unsafe_allow_html=True)

# ================================================
# DATA MAPPING & UTILS
# ================================================
CATEGORY_EMOJIS = {
    "Transport": "🚛",
    "Non-Transport": "🚗",
    "Construction Equipment Vehicle": "🏗️",
    "Tractor": "🚜",
    "Two Wheeler": "🛵",
    "Light Motor Vehicle": "🚕",
    "Heavy Goods Vehicle": "🚚",
    "Bus": "🚌",
    "Electric": "⚡"
}

def get_emoji(category):
    for key, emoji in CATEGORY_EMOJIS.items():
        if key.lower() in str(category).lower():
            return emoji
    return "🚙"

# ================================================
# DATA LOADING
# ================================================
@st.cache_data
def load_data():
    try:
        # Load vahan.csv
        df = pd.read_csv("vahan.csv")
        
        # Cleaning Logic
        if 'Group' in df.columns:
            df['Group'] = df['Group'].astype(str).apply(lambda x: x.split('\n')[0] if '\n' in x else x)
        
        # Clean Category if needed (remove newlines)
        if 'Category' in df.columns:
             df['Category'] = df['Category'].astype(str).apply(lambda x: x.strip())
        
        # Ensure 'Total' is numeric
        if 'Total' in df.columns:
            df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
            
        # Ensure Year is numeric
        if 'Year' in df.columns:
             df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
             df['Year'] = df['Year'].astype(int)

        # Load vehicle_summary.csv
        df_summary = pd.read_csv("vehicle_summary.csv")
        
        return df, df_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_vahan, df_summary = load_data()

# ================================================
# HEADER
# ================================================
st.markdown("""
<div class="header-gradient">
    <h1>🚗 Vahan Data Analyzer Dashboard</h1>
    <p>Comprehensive Insights into Vehicle Registration Data</p>
</div>
""", unsafe_allow_html=True)

# ================================================
# MAIN TABS
# ================================================
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat with Data"])

# ================================================
# TAB 1: DASHBOARD
# ================================================
with tab1:
    if not df_vahan.empty:
        # Filters
        with st.expander("🔎 Filters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_states = st.multiselect("Select State", options=df_vahan['State'].unique(), default=df_vahan['State'].unique()[:1])
            with c2:
                selected_years = st.multiselect("Select Year", options=sorted(df_vahan['Year'].unique()), default=sorted(df_vahan['Year'].unique()))
            with c3:
                selected_cats = st.multiselect("Select Category", options=df_vahan['Category'].unique())

        # Filter Data
        filtered_df = df_vahan.copy()
        if selected_states:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
        if selected_years:
            filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
        if selected_cats:
            filtered_df = filtered_df[filtered_df['Category'].isin(selected_cats)]

        # KPI Metrics
        total_vehicles = filtered_df['Total'].sum()
        top_state = filtered_df.groupby('State')['Total'].sum().idxmax() if not filtered_df.empty else "N/A"
        
        # Growth Metric (Year over Year if available)
        growth_text = ""
        if len(selected_years) > 1 or not selected_years:
            current_year = filtered_df['Year'].max()
            prev_year = current_year - 1
            curr_val = filtered_df[filtered_df['Year'] == current_year]['Total'].sum()
            prev_val = filtered_df[filtered_df['Year'] == prev_year]['Total'].sum()
            if prev_val > 0:
                growth = ((curr_val - prev_val) / prev_val) * 100
                growth_text = f"{growth:+.1f}% vs {prev_year}"
            else:
                growth_text = "N/A"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vehicles Registered", f"{total_vehicles:,.0f}", delta=growth_text if growth_text else None)
        col2.metric("Top State (in selection)", top_state)
        col3.metric("Data Rows", f"{len(filtered_df):,.0f}")

        # Charts
        st.markdown("### 📈 Visualizations")
        
        # Row 1
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.subheader("Yearly Trend 📅")
            yearly_trend = filtered_df.groupby('Year')['Total'].sum().reset_index()
            fig_year = px.line(yearly_trend, x='Year', y='Total', markers=True, title='Total Registrations per Year')
            fig_year.update_traces(line_color='#1f77b4')
            st.plotly_chart(fig_year, use_container_width=True)
            
        with r1c2:
            st.subheader("Category Distribution 🍭")
            cat_dist = filtered_df.groupby('Category')['Total'].sum().reset_index()
            # Add emojis to labels
            cat_dist['Label'] = cat_dist['Category'].apply(lambda x: f"{get_emoji(x)} {x}")
            fig_cat = px.pie(cat_dist, values='Total', names='Label', title='Share by Vehicle Category', hole=0.4)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Row 2
        st.subheader("State-wise Analysis 🗺️")
        state_dist = filtered_df.groupby('State')['Total'].sum().reset_index().sort_values('Total', ascending=False)
        fig_state = px.bar(state_dist, x='State', y='Total', color='Total', title='Registrations by State', height=500)
        st.plotly_chart(fig_state, use_container_width=True)

        
    else:
        st.warning("No data available. Please check 'vahan.csv'.")


# ================================================
# TAB 2: CHAT WITH DATA
# ================================================
with tab2:
    st.header("💬 Chat with Data AI")
    
    # API Key Retrieval
    try:
        api_key = st.secrets["secrets"]["OPENROUTER_API_KEY"]
    except Exception:
        api_key = None
        
    if not api_key:
        st.error("🔑 API Key not found! Please add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
        st.stop()

    mode = st.radio("Select AI Mode", ["📊 Graph Maker (PandasAI)", "🧠 Auto Analyst (Expert Chat)"], horizontal=True)
    
    user_query = st.text_area("Ask about your data:", help="e.g., 'Show me the trend of EVs' or 'Why did sales drop in 2020?'")

    
    if st.button("🚀 Analyze"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            if "Graph Maker" in mode:
                st.subheader("Generated Graph")
                try:
                    llm = PandasAI_OpenAI(api_token=api_key, model="openai/gpt-4o-mini") # Using a generic model mapping for OpenRouter if compatible or standard OpenAI
                    # Getting OpenRouter Base URL if needed, but PandasAI defaults to OpenAI. 
                    # For OpenRouter with PandasAI, we might need custom config. 
                    # Let's try standard instantiated LLM first or custom.
                    
                    # Correct configuration for OpenRouter in PandasAI can be tricky.
                    # We will use the direct OpenAI client compatible formatting if available, 
                    # or standard Generic LLM. 
                    
                    # For simplicity in this reliable demo, let's assume standard OpenAI structure works or use the 'openai' param 
                    # with base_url if supported. PandasAI 2.0+ supports custom connectors.
                    
                    # Workaround: PandasAI might not support OpenRouter out of the box easily without custom wrapper.
                    # I will use a simple custom wrapper or try defining base_url if the library allows.
                    # Checking PandasAI docs logic: it allows `OpenAI(api_token=..., base_url=...)`
                    
                    # LLM configuration
                    llm_config = {
                        "api_key": api_key,
                        "openai_proxy": "https://openrouter.ai/api/v1",
                        "model": "google/gemini-2.0-flash-001" # Using a fast model
                    }
                    
                    # Re-instantiate with proxy
                    agent = SmartDataframe(df_vahan, config={"llm": PandasAI_OpenAI(**llm_config)})
                    
                    response = agent.chat(user_query)
                    
                    # PandasAI usually returns a path to an image or the result
                    if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
                        st.image(response)
                    else:
                        st.write(response)
                        
                except Exception as e:
                    st.error(f"PandasAI Error: {e}")
            
            else: # Auto Analyst
                st.subheader("Analyst Insight")
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                
                # Context building
                data_summary = df_vahan.describe(include='all').to_string()
                
                system_prompt = f"""You are an Expert Automobile Industry Data Analyst.
                You have access to a dataset with the following summary:
                {data_summary}
                
                User question: {user_query}
                
                Provide a detailed, professional answer using automobile domain knowledge. 
                Explain trends, seasonality, or anomalies if relevant. Use emojis for categories.
                """
                
                try:
                    completion = client.chat.completions.create(
                        model="google/gemini-2.0-flash-001",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ]
                    )
                    st.markdown(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

