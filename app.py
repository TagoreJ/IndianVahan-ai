import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import io
import os
import warnings
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

# State Lat/Lon for Map
STATE_COORDS = {
    "Andhra Pradesh": [15.9129, 79.7400], "Arunachal Pradesh": [28.2180, 94.7278], "Assam": [26.2006, 92.9376],
    "Bihar": [25.0961, 85.3131], "Chhattisgarh": [21.2787, 81.8661], "Goa": [15.2993, 74.1240],
    "Gujarat": [22.2587, 71.1924], "Haryana": [29.0588, 76.0856], "Himachal Pradesh": [31.1048, 77.1734],
    "Jharkhand": [23.6102, 85.2799], "Karnataka": [15.3173, 75.7139], "Kerala": [10.8505, 76.2711],
    "Madhya Pradesh": [22.9734, 78.6569], "Maharashtra": [19.7515, 75.7139], "Manipur": [24.6637, 93.9063],
    "Meghalaya": [25.4670, 91.3662], "Mizoram": [23.1645, 92.9376], "Nagaland": [26.1584, 94.5624],
    "Odisha": [20.9517, 85.0985], "Punjab": [31.1471, 75.3412], "Rajasthan": [27.0238, 74.2179],
    "Sikkim": [27.5330, 88.5122], "Tamil Nadu": [11.1271, 78.6569], "Telangana": [18.1124, 79.0193],
    "Tripura": [23.9408, 91.9882], "Uttar Pradesh": [26.8467, 80.9462], "Uttarakhand": [30.0668, 79.0193],
    "West Bengal": [22.9868, 87.8550], "Delhi": [28.7041, 77.1025], "Puducherry": [11.9416, 79.8083],
    "Ladakh": [34.1526, 77.5771], "Jammu and Kashmir": [33.7782, 76.5762]
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

        # Enhance State for Map
        if 'State' in df.columns:
            # Simple clean: remove codes like (84) from "Andhra Pradesh(84)"
            df['State_Clean'] = df['State'].apply(lambda x: x.split('(')[0].strip())
            # Add Lat/Lon
            df['Lat'] = df['State_Clean'].map(lambda x: STATE_COORDS.get(x, [None, None])[0])
            df['Lon'] = df['State_Clean'].map(lambda x: STATE_COORDS.get(x, [None, None])[1])

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
# ================================================
# MAIN TABS
# ================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat with Data", "📈 Advanced Analytics", "📄 Generate Report"])

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
        col_map, col_bar = st.columns([1, 1])
        # Row 2
        col_map, col_bar = st.columns([1, 1])
        with col_map:
            st.subheader("Geo Sales Map 🗺️")
            
            # For Map: specific logic to show ALL India but highlight selection
            # distinct from 'filtered_df' which drives the metrics
            map_base = df_vahan.copy()
            if selected_years:
                map_base = map_base[map_base['Year'].isin(selected_years)]
            if selected_cats:
                map_base = map_base[map_base['Category'].isin(selected_cats)]
                
            # Aggregation for Map
            map_data = map_base.groupby(['State_Clean', 'Lat', 'Lon'])['Total'].sum().reset_index()
            
            # Color Logic
            if selected_states:
                # Highlight selected states
                map_data['Highlight'] = map_data['State_Clean'].apply(
                    lambda x: 'Selected' if x in [s.split('(')[0].strip() for s in selected_states] else 'Others'
                )
                color_col = 'Highlight'
            else:
                map_data['Highlight'] = 'All States'
                color_col = 'Total' # Default to heatmap-like coloring if nothing specifically selected

            fig_map = px.scatter_geo(
                map_data,
                lat='Lat', lon='Lon',
                size='Total', 
                color=color_col,
                hover_name='State_Clean',
                scope='asia',
                center={'lat': 20.5937, 'lon': 78.9629}, # India Center
                projection='natural earth',
                title='Geographic Distribution (Highlighting Selection)',
                color_discrete_map={'Selected': '#ff7f0e', 'Others': '#1f77b4'} if selected_states else None
            )
            fig_map.update_geos(fitbounds="locations", visible=False) # Auto-zoom
            st.plotly_chart(fig_map, use_container_width=True)

        with col_bar:
            st.subheader("State-wise Analysis 🎢")
            state_dist = filtered_df.groupby('State')['Total'].sum().reset_index().sort_values('Total', ascending=False).head(10)
            fig_state = px.bar(state_dist, x='State', y='Total', color='Total', title='Top 10 States by Registrations', height=400)
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
        # Check for direct key or nested key (handles both local [secrets] section and Cloud's flat secrets)
        if "OPENROUTER_API_KEY" in st.secrets:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        elif "secrets" in st.secrets and "OPENROUTER_API_KEY" in st.secrets["secrets"]:
            api_key = st.secrets["secrets"]["OPENROUTER_API_KEY"]
        else:
            api_key = None
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
                    # LAZY IMPORT PANDASAI
                    # Using standard import now that we pinned pandasai<3.0.0
                    from pandasai import SmartDataframe
                    from pandasai.llm import OpenAI as PandasAI_OpenAI
                    
                    # LLM configuration
                    llm_config = {
                        "api_key": api_key,
                        "openai_proxy": "https://openrouter.ai/api/v1",
                        "model": "google/gemini-2.0-flash-001" 
                    }
                    
                    # Re-instantiate with proxy
                    agent = SmartDataframe(df_vahan, config={"llm": PandasAI_OpenAI(**llm_config)})
                    
                    with st.spinner("🤖 Generative AI is thinking... (This may take up to 30 seconds)"):
                        response = agent.chat(user_query)
                    
                    # PandasAI usually returns a path to an image or the result
                    if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
                        st.image(response)
                    else:
                        st.write(response)
                        
                except Exception as e:
                    st.error(f"PandasAI Error (Check compatibility): {e}")
            
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
                    with st.spinner("🧠 Analyst is processing insights..."):
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

# ================================================
# TAB 3: ADVANCED ANALYTICS
# ================================================
with tab3:
    st.header("📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    # 1. FORECASTING
    with col1:
        st.subheader("🔮 Sales Forecasting")
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            forecast_df = df_vahan.groupby('Year')['Total'].sum().reset_index()
            
            if len(forecast_df) > 1:
                X = forecast_df[['Year']]
                y = forecast_df['Total']
                
                model = LinearRegression()
                model.fit(X, y)
                
                future_years = [2024, 2025, 2026]
                predictions = model.predict(pd.DataFrame(future_years, columns=['Year']))
                
                future_df = pd.DataFrame({'Year': future_years, 'Total': predictions, 'Type': 'Forecast'})
                forecast_df['Type'] = 'Actual'
                
                combined_df = pd.concat([forecast_df, future_df])
                
                fig_forecast = px.line(combined_df, x='Year', y='Total', color='Type', 
                                    markers=True, title='Projected Registrations (Linear Regression)')
                fig_forecast.add_annotation(x=2026, y=predictions[-1], text="2026 Prediction", showarrow=True, arrowhead=1)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.info(f"Projected Growth: The model predicts a steady trend based on historical data.")
            else:
                st.warning("Not enough data points for forecasting.")
        except ImportError:
            st.error("Sklearn not installed.")
        except Exception as e:
            st.error(f"Forecasting Error: {e}")

    # 2. SEASONALITY
    with col2:
        st.subheader("📅 Seasonal Heatmap")
        
        if 'Month' in df_vahan.columns:
            # Aggregate
            heatmap_data = df_vahan.groupby(['Year', 'Month'])['Total'].sum().reset_index()
            
            if not heatmap_data.empty:
                # Pivot for Heatmap
                pivot_table = heatmap_data.pivot(index='Year', columns='Month', values='Total').fillna(0)
                
                # Sort Months logically if possible (requires mapping)
                month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                # Reindex columns if they match standard names (or subset of them)
                existing_months = [m for m in month_order if m in pivot_table.columns]
                pivot_table = pivot_table[existing_months]
                
                fig_heat = px.imshow(pivot_table, text_auto=True, aspect="auto", 
                                    color_continuous_scale='Viridis', title='Monthly Registration Intensity')
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("No data for heatmap.")
        else:
            st.warning("Month column not found for seasonality analysis.")

# ================================================
# TAB 4: REPORT GENERATION
# ================================================
with tab4:
    st.header("📄 Generate Intelligence Report")
    
    st.write("Click below to generate a professional PDF summary of the current dataset.")
    
    if st.button("📥 Download Report"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            
            # Title
            c.setFont("Helvetica-Bold", 24)
            c.drawString(50, height - 50, "Vahan Business Intelligence Report")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
            
            # Key Metrics
            c.line(50, height - 90, width - 50, height - 90)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 120, "Key Performance Indicators")
            
            c.setFont("Helvetica", 14)
            total_reg = df_vahan['Total'].sum()
            c.drawString(50, height - 150, f"Total Registrations: {total_reg:,.0f}")
            
            if not df_vahan.empty:
                top_st = df_vahan.groupby('State')['Total'].sum().idxmax()
                c.drawString(50, height - 175, f"Top Performing State: {top_st}")
                
                top_cat = df_vahan.groupby('Category')['Total'].sum().idxmax()
                c.drawString(50, height - 200, f"Dominant Category: {top_cat}")
            
            # Insight Text
            c.line(50, height - 220, width - 50, height - 220)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 250, "Automated Insights")
            
            c.setFont("Helvetica", 12)
            text_lines = [
                "1. The dataset shows a significant concentration of vehicles in the top states.",
                "2. Seasonality analysis indicates potential peaks during festival months.",
                "3. Linear forecasts suggest continued growth in the upcoming years.",
                "4. EV adoption trends should be monitored in the 'Electric' category."
            ]
            y_pos = height - 280
            for line in text_lines:
                c.drawString(50, y_pos, line)
                y_pos -= 25
                
            c.showPage()
            c.save()
            
            buffer.seek(0)
            st.success("Report generated successfully!")
            st.download_button(
                label="Download PDF",
                data=buffer,
                file_name="vahan_analytics_report.pdf",
                mime="application/pdf"
            )
            
        except ImportError:
            st.error("ReportLab not installed.")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
