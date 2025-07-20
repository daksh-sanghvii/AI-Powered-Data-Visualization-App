import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
import base64
import numpy as np
from dotenv import load_dotenv
import io
import contextlib
from io import StringIO

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Configure the Gemini API
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Load the Gemini model
def get_gemini_model():
    return genai.GenerativeModel(model_name="gemini-1.5-pro")

# Safe execution context
@contextlib.contextmanager
def capture_stdout():
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    yield mystdout
    sys.stdout = old_stdout

def generate_code_from_prompt(model, prompt, df_info):
    """Generate Python code based on user's question and dataframe information"""
    full_prompt = f"""
You are a Python data analyst expert. Generate code to answer the user's question about their data.

DataFrame Information:
{df_info}

User Question: "{prompt}"

Generate complete Python code using pandas, matplotlib, and seaborn to answer this question.
Include detailed visualizations where appropriate.
Make the visualizations attractive with proper titles, labels, and colors.
Use plt.figure(figsize=(10, 6)) for better visualization size.
Don't include markdown, only Python code.
Assume the DataFrame is already loaded as 'df'.
"""
    response = model.generate_content(full_prompt)
    code_text = response.text
    # Clean up code blocks if present
    code_text = code_text.replace("```python", "").replace("```", "")
    return code_text.strip()

def run_code_on_dataframe(df, code):
    """Execute the generated code on the dataframe"""
    # Create a local environment with the dataframe and necessary libraries
    local_env = {
        "df": df, 
        "pd": pd, 
        "plt": plt, 
        "sns": sns,
        "np": np
    }
    
    # Capture stdout and execute code
    with capture_stdout() as output:
        try:
            # Create a figure for matplotlib and add it to local_env
            fig = plt.figure(figsize=(10, 6))
            local_env["plt"] = plt  # Ensure plt is in the local environment
            
            # Execute the code
            exec(code, globals(), local_env)  # Use globals() to ensure all imports are available
            
            # Check if the code created any plots
            if plt.get_fignums():
                fig = plt.gcf()  # Get current figure
                return output.getvalue(), fig
            else:
                return output.getvalue(), None
        except Exception as e:
            return f"Error executing code: {str(e)}", None

def get_dataframe_info(df):
    """Get comprehensive information about the dataframe"""
    buffer = StringIO()
    
    # Basic dataframe info
    buffer.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
    
    # Column names and types
    buffer.write("Columns:\n")
    for col in df.columns:
        buffer.write(f"- {col} (Type: {df[col].dtype})\n")
    buffer.write("\n")
    
    # Sample data (first 5 rows)
    buffer.write("Sample Data (first 5 rows):\n")
    buffer.write(df.head().to_string())
    buffer.write("\n\n")
    
    # Statistical summary
    buffer.write("Statistical Summary:\n")
    buffer.write(df.describe().to_string())
    
    return buffer.getvalue()

def analyze_data_with_ai(model, df):
    """Get AI analysis of the dataframe"""
    df_info = get_dataframe_info(df)
    
    prompt = f"""
    Analyze this dataset and provide:
    1. A summary of what this data represents
    2. Key insights and patterns
    3. Potential correlations between variables
    4. Recommendations for further analysis
    
    Dataset Information:
    {df_info}
    """
    
    response = model.generate_content(prompt)
    return response.text

def get_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("📊 Data Visualization & Analysis")
    st.write("Upload your CSV or Excel file and ask questions to get instant visualizations and analysis.")
    
    # API key input
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Enter your Gemini API Key", type="password")
    if api_key:
        configure_genai(api_key)
        model = get_gemini_model()
    else:
        st.warning("Please enter your Gemini API key to enable AI analysis")
        model = None
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a data file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load the data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Quick statistics
            st.subheader("Quick Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Shape:**")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Columns: {df.shape[1]}")
                
                st.write("**Missing Values:**")
                missing = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Values': missing.values
                })
                st.dataframe(missing_df)
            
            with col2:
                st.write("**Data Types:**")
                dtypes_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtypes_df)
            
            # Automatic Analysis
            st.subheader("Automatic Analysis")
            if model:
                if st.button("Generate AI Analysis"):
                    with st.spinner("Analyzing data..."):
                        analysis = analyze_data_with_ai(model, df)
                        st.markdown(analysis)
            
            # Basic visualizations
            st.subheader("Basic Visualizations")
            viz_type = st.selectbox(
                "Select visualization type",
                ["Histogram", "Scatter Plot", "Bar Chart", "Correlation Heatmap", "Box Plot"]
            )
            
            if viz_type == "Histogram":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    col = st.selectbox("Select column", numeric_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[col], kde=True, ax=ax)
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns available for histogram")
            
            elif viz_type == "Scatter Plot":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) >= 2:
                    col_x = st.selectbox("Select X-axis column", numeric_cols)
                    col_y = st.selectbox("Select Y-axis column", [c for c in numeric_cols if c != col_x])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
                    plt.title(f"{col_y} vs {col_x}")
                    plt.xlabel(col_x)
                    plt.ylabel(col_y)
                    st.pyplot(fig)
                else:
                    st.info("Need at least 2 numeric columns for scatter plot")
            
            elif viz_type == "Bar Chart":
                if len(df.columns) >= 2:
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    
                    if len(cat_cols) > 0 and len(num_cols) > 0:
                        cat_col = st.selectbox("Select category column", cat_cols)
                        num_col = st.selectbox("Select value column", num_cols)
                        
                        # Get top 10 categories by value
                        top_cats = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10).index
                        filtered_df = df[df[cat_col].isin(top_cats)]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=filtered_df, x=cat_col, y=num_col, ax=ax)
                        plt.title(f"{num_col} by {cat_col} (Top 10)")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Need at least one categorical and one numeric column for bar chart")
                else:
                    st.info("Need at least 2 columns for bar chart")
            
            elif viz_type == "Correlation Heatmap":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    plt.title("Correlation Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Need at least 2 numeric columns for correlation heatmap")
            
            elif viz_type == "Box Plot":
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    col = st.selectbox("Select column for box plot", numeric_cols)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(y=df[col], ax=ax)
                    plt.title(f"Box Plot of {col}")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns available for box plot")
            
            # Custom Analysis with AI
            st.subheader("Ask Questions About Your Data")
            if model:
                question = st.text_area("Enter your question:", 
                                        placeholder="Example: What is the correlation between columns? Show me a histogram of the main numeric columns.")
                
                # Suggested questions
                st.write("**Suggested Questions:**")
                suggestions = [
                    "Show me the distribution of values in each numeric column",
                    "What are the correlations between numeric columns?",
                    "Create a scatter plot of [column1] vs [column2]",
                    "What are the top 10 values in [column]?",
                    "Show me a summary of missing values in the dataset"
                ]
                
                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"sugg_{i}"):
                        question = suggestion
                
                if question:
                    with st.spinner("Analyzing your data..."):
                        # Get dataframe information
                        df_info = get_dataframe_info(df)
                        
                        # Generate code based on the question
                        code = generate_code_from_prompt(model, question, df_info)
                        
                        # Show the generated code
                        with st.expander("Generated Code", expanded=False):
                            st.code(code, language="python")
                        
                        # Execute the code
                        output, fig = run_code_on_dataframe(df, code)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Display any text output
                        if output and output.strip():
                            st.text(output)
                        
                        # Display the figure if one was created
                        if fig:
                            st.pyplot(fig)
            else:
                st.info("Enter your Gemini API key to ask questions about your data")
            
            # Download options
            st.subheader("Download Options")
            st.markdown(get_download_link(df, filename=uploaded_file.name), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample datasets when no file is uploaded
        st.info("Upload a CSV or Excel file to get started, or try one of our sample datasets:")
        
        sample_datasets = {
            "Iris Flower Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Titanic Passenger Data": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "Housing Prices": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/housing.csv"
        }
        
        selected_sample = st.selectbox("Select a sample dataset:", list(sample_datasets.keys()))
        
        if st.button("Load Sample Dataset"):
            with st.spinner(f"Loading {selected_sample}..."):
                try:
                    df = pd.read_csv(sample_datasets[selected_sample])
                    st.success(f"Successfully loaded {selected_sample} with {df.shape[0]} rows and {df.shape[1]} columns.")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Store the dataframe in session state
                    st.session_state.df = df
                    st.session_state.dataset_name = selected_sample
                    
                    # Rerun to refresh the page with the loaded dataset
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")

if __name__ == "__main__":
    main()