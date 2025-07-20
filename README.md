# Data Visualization Pipeline Using LLMs

## Project Overview

This project creates an intelligent data visualization pipeline using Google's Gemini LLM. Users can upload CSV or Excel files and ask natural language questions about their data. The LLM generates Python code to analyze the data and create visualizations, which are then executed and displayed in a Streamlit web application.

## 🎥 Demo Video

▶️ [Click here to watch the demo video](https://drive.google.com/file/d/1-ERrh345D8JXNA_JdPr4er6UykIWaZMy/view?usp=share_link)

## Key Features

- **File Upload**: Support for CSV and Excel files
- **Natural Language Queries**: Ask questions in plain English
- **AI-Powered Analysis**: Gemini LLM generates Python code for data analysis
- **Interactive Visualizations**: Automatic generation of charts and graphs
- **Sample Datasets**: Built-in sample datasets for testing
- **Download Options**: Export analyzed data as CSV

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **AI/LLM**: Google Gemini 1.5 Pro
- **Environment**: Python 3.8+

## Project Architecture

### Core Components

1. **Streamlit Interface** (`main()` function)

   - File upload functionality
   - User input for questions
   - Display of results and visualizations

2. **LLM Integration** (`generate_code_from_prompt()`)

   - Sends user questions to Gemini
   - Receives Python code as response
   - Formats code for execution

3. **Code Execution Engine** (`run_code_on_dataframe()`)

   - Safely executes generated code
   - Captures output and visualizations
   - Handles errors gracefully

4. **Data Analysis Functions**
   - `get_dataframe_info()`: Extracts comprehensive data info
   - `analyze_data_with_ai()`: Performs automatic AI analysis
   - Basic visualization functions for common chart types

## How It Works

### 1. Data Upload

- User uploads CSV/Excel file through Streamlit interface
- Data is loaded into a Pandas DataFrame
- Basic statistics and preview are displayed

### 2. Question Processing

- User asks a question in natural language
- Question is sent to Gemini LLM along with data information
- LLM generates appropriate Python code to answer the question

### 3. Code Generation

The LLM receives:

- User's question
- DataFrame information (shape, columns, data types, sample data, statistics)
- Instructions to generate Python code using pandas, matplotlib, seaborn

### 4. Code Execution

- Generated code is executed in a controlled environment
- Output (text and visualizations) is captured
- Results are displayed to the user

### 5. Visualization Display

- Matplotlib figures are captured and displayed
- Text output is shown
- Generated code is available for inspection

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 3. Set Environment Variable

Create a `.env` file in the project directory:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
streamlit run data_analyzer.py
```
