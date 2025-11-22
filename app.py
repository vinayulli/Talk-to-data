# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from datetime import datetime
from openai import OpenAI
import textwrap

# ---------- CONFIG ----------
MODEL_NAME = "gpt-4o"   # replace with your model identifier if different

# Initialize session state for API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

def get_openai_client():
    """Get OpenAI client if API key is set"""
    if st.session_state.openai_api_key:
        try:
            return OpenAI(api_key=st.session_state.openai_api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None

# ---------- HELPERS ----------
def load_file(uploaded_file):
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif fname.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

def detect_column_types(df):
    types = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_timedelta64_dtype(df[c]):
            types[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[c]):
            # Don't convert numeric columns to datetime - they should stay numeric
            types[c] = "numeric"
        else:
            # Only try to parse non-numeric columns as datetime
            # Check if values look like date strings, not numeric timestamps
            try:
                # Sample a few values to check if they're date-like strings
                sample = df[c].dropna().head(10)
                if len(sample) > 0:
                    # Only parse if values are strings that look like dates
                    if sample.dtype == 'object':
                        # Try parsing as datetime, but be strict
                        parsed = pd.to_datetime(df[c], errors='coerce', format='mixed')
                        non_null = parsed.notnull().sum()
                        # Only convert if most values are valid dates and they're not all the same timestamp
                        if non_null >= max(1, len(df) * 0.5) and parsed.nunique() > 1:
                            # Additional check: make sure it's not just numeric values interpreted as timestamps
                            # If all parsed dates are from 1970, it's likely a mistake
                            unique_years = parsed.dt.year.unique()
                            if len(unique_years) > 1 or (len(unique_years) == 1 and unique_years[0] != 1970):
                                df[c] = parsed
                                types[c] = "datetime"
                                continue
            except Exception:
                pass
            # small cardinality? treat as categorical
            nunique = df[c].nunique(dropna=True)
            if nunique <= 20:
                types[c] = "categorical"
            else:
                types[c] = "text"
    return types

def create_visualization_prompt(question, df_sample, col_types):
    """Create prompt for LLM to determine visualization and data processing"""
    schema = "\n".join([f"- {c}: {col_types[c]}" for c in col_types])
    sample_rows = df_sample.head(20).to_dict(orient="records")
    sample_text = "\n".join([str(r) for r in sample_rows])
    
    prompt = textwrap.dedent(f"""
    You are a data analysis assistant. Analyze the user's question and determine:
    1. What visualization type to use
    2. Which columns to use
    3. What data processing/aggregation is needed
    4. How to prepare the data for visualization

    User Question: "{question}"

    Available Data Schema:
    {schema}

    Sample Data (first 20 rows):
    {sample_text}

    Based on the question, schema, and sample data, provide a JSON response with the following structure:
    {{
        "chart_type": "bar|line|pie|scatter|histogram|table",
        "x_column": "column_name or null",
        "y_column": "column_name or null",
        "aggregation": "sum|mean|count|max|min|none",
        "group_by": "column_name or null",
        "time_grouping": "month|year|day|week|none",
        "filter_conditions": {{"column": "value"}} or null,
        "sort_by": "column_name or null",
        "sort_order": "asc|desc",
        "explanation": "Brief explanation of your choices"
    }}

    Guidelines:
    - For questions about "which month/year", use bar chart with time_grouping
    - For "total" or "sum" queries, use aggregation: "sum"
    - For "average" queries, use aggregation: "mean"
    - For "count" queries, use aggregation: "count"
    - Match column names from the schema to what the user is asking about
    - If the question asks about categories, use the categorical column
    - If the question asks about amounts/sales/values, use the numeric column
    - For time-based questions, identify the datetime column
    - Use pie charts for small categorical distributions (<=6 categories)
    - Use bar charts for comparisons
    - Use line charts for trends over time
    - Use scatter for relationships between two numeric variables

    Return ONLY valid JSON, no additional text.
    """)
    return prompt

def call_llm_for_visualization(prompt, model=MODEL_NAME, max_tokens=500):
    """Call LLM to get visualization configuration"""
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system", "content":"You are a data visualization expert. Always respond with valid JSON only."},
                {"role":"user", "content":prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        st.error(f"Error calling LLM: {str(e)}")
        return None

def process_data_for_visualization(df, config):
    """Process dataframe based on LLM configuration"""
    try:
        df_processed = df.copy()
        
        # Apply filters if specified
        if config.get("filter_conditions"):
            for col, value in config["filter_conditions"].items():
                if col in df_processed.columns:
                    df_processed = df_processed[df_processed[col] == value]
        
        # Select relevant columns - ensure we keep all needed columns for grouping/aggregation
        cols_to_keep = []
        group_by_col = config.get("group_by") or config.get("x_column")
        
        if config.get("x_column") and config["x_column"] in df_processed.columns:
            if config["x_column"] not in cols_to_keep:
                cols_to_keep.append(config["x_column"])
        if config.get("y_column") and config["y_column"] in df_processed.columns:
            if config["y_column"] not in cols_to_keep:
                cols_to_keep.append(config["y_column"])
        if group_by_col and group_by_col in df_processed.columns:
            if group_by_col not in cols_to_keep:
                cols_to_keep.append(group_by_col)
        
        # Only select columns if we have specific ones to keep, otherwise keep all
        if cols_to_keep:
            # Remove duplicates while preserving order
            cols_to_keep = list(dict.fromkeys(cols_to_keep))
            df_processed = df_processed[cols_to_keep].dropna()
        
        # Handle time grouping
        x_col = config.get("x_column")
        if x_col and x_col in df_processed.columns:
            if config.get("time_grouping") and pd.api.types.is_datetime64_any_dtype(df_processed[x_col]):
                time_group = config["time_grouping"]
                if time_group == "month":
                    df_processed['_time_group'] = df_processed[x_col].dt.to_period('M').astype(str)
                    df_processed[x_col] = df_processed['_time_group']
                    df_processed = df_processed.drop(columns=['_time_group'])
                elif time_group == "year":
                    df_processed['_time_group'] = df_processed[x_col].dt.year.astype(str)
                    df_processed[x_col] = df_processed['_time_group']
                    df_processed = df_processed.drop(columns=['_time_group'])
                elif time_group == "day":
                    df_processed['_time_group'] = df_processed[x_col].dt.date.astype(str)
                    df_processed[x_col] = df_processed['_time_group']
                    df_processed = df_processed.drop(columns=['_time_group'])
        
        # Apply aggregation
        agg_type = config.get("aggregation", "none")
        group_by_col = config.get("group_by") or config.get("x_column")
        y_col = config.get("y_column")
        
        # Ensure group_by_col exists and is a valid column
        if group_by_col:
            # Convert to string if it's not already
            if not isinstance(group_by_col, str):
                group_by_col = str(group_by_col)
            
            if group_by_col not in df_processed.columns:
                # Try to find it in original df
                if group_by_col in df.columns:
                    df_processed[group_by_col] = df[group_by_col]
                else:
                    # Column doesn't exist, can't proceed with grouping
                    group_by_col = None
        
        if agg_type == "count" and group_by_col:
            # For count, we just count rows per group
            if group_by_col in df_processed.columns:
                count_col_name = y_col if y_col and y_col in df_processed.columns else "count"
                df_processed = df_processed.groupby(group_by_col, as_index=False).size()
                df_processed.columns = [group_by_col, count_col_name]
        elif agg_type != "none" and group_by_col and y_col:
            if group_by_col in df_processed.columns and y_col in df_processed.columns:
                # Ensure group_by_col is a single column (not a list)
                if isinstance(group_by_col, (list, tuple)):
                    group_by_col = group_by_col[0]
                
                if agg_type == "sum":
                    df_processed = df_processed.groupby(group_by_col, as_index=False)[y_col].sum()
                elif agg_type == "mean":
                    df_processed = df_processed.groupby(group_by_col, as_index=False)[y_col].mean()
                elif agg_type == "max":
                    df_processed = df_processed.groupby(group_by_col, as_index=False)[y_col].max()
                elif agg_type == "min":
                    df_processed = df_processed.groupby(group_by_col, as_index=False)[y_col].min()
        
        # Apply sorting
        if config.get("sort_by") and config["sort_by"] in df_processed.columns:
            ascending = config.get("sort_order", "desc") == "asc"
            df_processed = df_processed.sort_values(by=config["sort_by"], ascending=ascending)
        
        return df_processed
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error processing data: {str(e)}")
        with st.expander("Error Details (click to expand)"):
            st.text(f"Config: {config}")
            st.text(f"DataFrame columns: {list(df.columns)}")
            st.text(f"Full error: {error_details}")
        return df.copy()

def render_chart_from_config(chart_type, df, config):
    """Render chart based on LLM configuration"""
    try:
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        
        if chart_type == "bar":
            if y_col and x_col:
                fig = px.bar(df, x=x_col, y=y_col, title=config.get("explanation", ""))
            elif x_col:
                # Count chart
                fig = px.bar(df, x=x_col, y=df.columns[-1], title=config.get("explanation", ""))
            else:
                st.write("Bar chart requires at least x_column")
                return
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "line":
            if x_col and y_col:
                fig = px.line(df, x=x_col, y=y_col, markers=True, title=config.get("explanation", ""))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Line chart requires both x_column and y_column")
                
        elif chart_type == "pie":
            if x_col and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=config.get("explanation", ""))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Pie chart requires both x_column and y_column")
                
        elif chart_type == "scatter":
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=config.get("explanation", ""))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Scatter chart requires both x_column and y_column")
                
        elif chart_type == "histogram":
            if y_col:
                fig = px.histogram(df, x=y_col, title=config.get("explanation", ""))
                st.plotly_chart(fig, use_container_width=True)
            elif x_col:
                fig = px.histogram(df, x=x_col, title=config.get("explanation", ""))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Histogram requires a column")
                
        elif chart_type == "table":
            st.dataframe(df.head(200))
        else:
            st.write(f"Unknown chart type: {chart_type}")
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")

def create_insight_prompt(question, df_result, config):
    """Create prompt for LLM to generate insights from results"""
    result_summary = df_result.head(50).to_dict(orient="records")
    result_text = "\n".join([str(r) for r in result_summary])
    
    # Calculate some basic statistics
    stats = {}
    if config.get("y_column") and config["y_column"] in df_result.columns:
        y_col = config["y_column"]
        if pd.api.types.is_numeric_dtype(df_result[y_col]):
            stats = {
                "total": float(df_result[y_col].sum()) if config.get("aggregation") == "sum" else None,
                "mean": float(df_result[y_col].mean()),
                "max": float(df_result[y_col].max()),
                "min": float(df_result[y_col].min()),
                "count": len(df_result)
            }
    
    prompt = textwrap.dedent(f"""
    You are a data insights assistant. The user asked: "{question}"

    Visualization Configuration:
    - Chart Type: {config.get("chart_type")}
    - X Column: {config.get("x_column")}
    - Y Column: {config.get("y_column")}
    - Aggregation: {config.get("aggregation")}

    Result Data (first 50 rows):
    {result_text}

    Statistics:
    {stats}

    Task:
    1) In 2-3 sentences, describe what the visualization shows and answer the user's question.
    2) Give 3-4 concise insights based on the actual data (use exact numbers from the results).
    3) Suggest 2-3 follow-up queries the user might ask based on this data.

    IMPORTANT: Use only the numbers and values present in the result data. Be specific with numbers.

    Output format:
    Summary: ...
    Insights: 
    1) ...
    2) ...
    3) ...
    4) ...
    Follow-up Questions:
    1) ...
    2) ...
    3) ...
    """)
    return prompt

def call_llm_for_insights(prompt, model=MODEL_NAME, max_tokens=500):
    """Call LLM to generate insights"""
    client = get_openai_client()
    if client is None:
        return "LLM disabled. Please enter your OpenAI API key in the sidebar to enable insights."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system", "content":"You are a helpful, concise data insights assistant."},
                {"role":"user", "content":prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Converse with your data", layout="wide")
st.title("Converse with your data")
st.markdown("Upload CSV or Excel, ask a question, and get a table, chart and insights powered by an LLM.")

with st.sidebar:
    st.header("OpenAI API Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key to enable LLM-powered insights and visualization selection.",
        placeholder="sk-..."
    )
    if api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key_input
    
    if st.session_state.openai_api_key:
        st.success("✓ API key set")
    else:
        st.warning("⚠ API key required for LLM features")
    
    st.markdown("---")
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xls","xlsx"])
    sample_mode = st.checkbox("Use internal sample (if no file)", value=False)
    show_schema = st.checkbox("Show detected schema", True)
    max_rows = st.slider("Rows to preview", min_value=5, max_value=500, value=100, step=5)

if uploaded:
    df = load_file(uploaded)
elif sample_mode:
    @st.cache_data
    def load_sample():
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=12, freq="ME"),
            "category": ["A","B","A","C","B","A","A","C","B","A","B","C"],
            "sales": np.random.randint(100, 1000, 12),
            "orders": np.random.randint(1,20,12)
        })
    df = load_sample()
else:
    st.info("Upload a file or enable sample mode in the sidebar.")
    df = None

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(max_rows))

    # detect column types
    col_types = detect_column_types(df)
    if show_schema:
        st.markdown("**Detected schema:**")
        st.write(col_types)

    st.subheader("Ask a question about your data")
    question = st.text_input("e.g. 'Show monthly sales for 2024' or 'Which category has highest average sales?'")
    
    if st.button("Run query") and question.strip():
        client = get_openai_client()
        if client is None:
            st.error("OpenAI API key not set. Please enter your API key in the sidebar to enable LLM features.")
        else:
            with st.spinner("Analyzing question and preparing visualization..."):
                # Step 1: LLM determines visualization and data processing
                viz_prompt = create_visualization_prompt(question, df, col_types)
                config = call_llm_for_visualization(viz_prompt)
                
                if config:
                    st.markdown(f"**Visualization:** {config.get('chart_type', 'unknown')}")
                    if config.get('explanation'):
                        st.caption(f"*{config['explanation']}*")
                    
                    # Step 2: Process data based on LLM configuration
                    df_processed = process_data_for_visualization(df, config)
                    
                    # Step 3: Render visualization
                    render_chart_from_config(config.get("chart_type", "table"), df_processed, config)
                    
                    # Step 4: Generate insights from results
                    with st.spinner("Generating insights..."):
                        insight_prompt = create_insight_prompt(question, df_processed, config)
                        insight_text = call_llm_for_insights(insight_prompt)
                        st.subheader("LLM Insights")
                        st.write(insight_text)
                    
                    # Step 5: Show processed data
                    st.subheader("Returned Data (first 200 rows)")
                    st.dataframe(df_processed.head(200))
                    
                    # Provide downloadable CSV
                    csv = df_processed.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download result CSV", data=csv, file_name="query_result.csv", mime="text/csv")
                else:
                    st.error("Failed to get visualization configuration from LLM. Please try again.")

    st.markdown("---")
    st.markdown("**Notes**: The LLM analyzes your question and automatically selects the best visualization and data processing approach.")
