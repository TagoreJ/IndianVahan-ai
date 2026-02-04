# 🚗 Vahan Data Analyzer Dashboard

**Version 1.0** | *More updates coming soon!* 🚀

A comprehensive Streamlit dashboard for analyzing Indian Vehicle Registration Data (Vahan). This tool provides insights into vehicle sales trends, category distribution, and state-wise performance using interactive visualizations and AI-powered analysis.

![Vahan Dashboard](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌟 Features

*   **📊 Interactive Dashboard**:
    *   **Monthly Sales Trends**: Visualize seasonality and MOM growth.
    *   **Geo-Spatial Map**: Interactive map of India showing state-wise vehicle density (Mapbox styled).
    *   **Category Analysis**: Breakdown of vehicle types (EVs, Transport, Two-wheelers) with growth metrics.
    *   **Dynamic Filters**: Filter by State (supports "All India"), Year, and Category.
*   **🧠 Expert Auto Analyst (AI)**:
    *   Powered by LLMs (via OpenRouter) to answer complex questions about the data.
    *   Context-aware: Knows exactly "which state sold more in 2024" vs "2023".
*   **📈 Advanced Analytics**:
    *   **Forecasting**: Linear Regression models to predict future registration trends.
    *   **Seasonality Heatmap**: Visualize peak sales months.
*   **📄 Intelligence Reports**: Generate downloadable PDF summaries of the current data view.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tagorej/indianvahan-ai.git
    cd indianvahan-ai
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Secrets**:
    *   Create a file `.streamlit/secrets.toml`:
    ```toml
    [secrets]
    OPENROUTER_API_KEY = "your_api_key_here"
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 🚀 Deployment (Streamlit Cloud)

1.  Push this code to your GitHub repository.
2.  Connect your repo to [Streamlit Cloud](https://streamlit.io/cloud).
3.  In the **Advanced Settings**, paste your API Key into the **Secrets** area:
    ```toml
    OPENROUTER_API_KEY = "your_api_key_here"
    ```
4.  Deploy! The `runtime.txt` ensures the correct Python version (3.11) is used.

## 📁 Project Structure

*   `app.py`: Main application code.
*   `vahan.csv`: Core dataset (Vehicle Registrations).
*   `requirements.txt`: Python dependencies.
*   `runtime.txt`: Configuration for Streamlit Cloud (Python 3.11).

## 👨‍💻 Author

**Tagore J**  
[LinkedIn](https://www.linkedin.com/in/tagorej/)

---
*Note: Data extracted from Vahan Parivahan Dashboard via automation. Subject to extraction limitations.*
