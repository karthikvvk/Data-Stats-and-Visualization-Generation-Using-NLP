# Data Insights and Visualization Generation

## Overview

This project allows users to upload a CSV file, query insights from the dataset, and get visualization recommendations based on those queries. It leverages advanced NLP and generative AI technologies to extract meaningful insights, predict the need for visual representation (charts/graphs), and generate Python code that visualizes the data using libraries like `matplotlib`.

### Key Features:
- **Data Analysis**: The ability to generate summary statistics and data insights from the uploaded CSV file.
- **Natural Language Processing**: A spaCy model is used to extract relevant keywords from user queries.
- **Generative AI**: Google Gemini (through the `google-generativeai` API) is used to generate Python code for data visualization based on user queries and dataset features.
- **Visualization**: The generated code is executed to visualize the dataset in an interactive manner using `matplotlib`.
- **Model Training**: The project also includes a training pipeline for a custom classifier that predicts the need for visualizations based on queries, using the `Roberta` model from Hugging Face.

## Requirements

Before running the project, ensure that all dependencies are installed. You can install them using the `requirements.txt` file provided.

### Install Dependencies

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This will install all necessary packages listed in the `requirements.txt` file, including:
- `pandas`
- `spacy`
- `torch`
- `transformers`
- `google-generativeai`
- `streamlit`
- `matplotlib`
- `datasets`

Additionally, the `spaCy` model `en_core_web_sm` will be automatically downloaded.

```bash
python -m spacy download en_core_web_sm
```

### Setup Google Generative AI API Key

To use the Google Gemini model, you'll need to provide your own API key. Replace the placeholder `KEY` in the script with your actual API key.

```python
KEY = 'YOUR_GOOGLE_API_KEY_HERE'
```

## Project Structure

```plaintext
├── requirements.txt         # List of required Python packages
├── train.py                 # Script to train a custom classification model
├── data.csv                 # Sample data CSV (for querying and training)
├── train_data.csv           # Data for training the classification model
├── app.py                   # Main Streamlit app for visualizations and querying
└── plt.py                   # Generated Python code for visualizations (temporary)
```

### Files Explained:
1. **`train.py`**: This script trains a custom classification model using a dataset (`train_data.csv`). It uses the Hugging Face `Roberta` transformer model to classify whether a graph is needed based on the query.
2. **`app.py`**: The main Streamlit app where users can upload a CSV file, input a query, and generate insights or visualizations based on that query. It uses the `google-generativeai` API to generate code and then runs that code to display the chart.
3. **`data.csv`**: A sample CSV file used for data analysis and visualization. Users can upload their own CSV files.
4. **`train_data.csv`**: A CSV file with queries and corresponding labels used to train the model to predict whether visual representation (graphs) is needed for a given query.

## How to Run the Project

### Step 1: Install Dependencies

Make sure to install all dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Also, download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

### Step 2: Train the Model

Before using the application, you need to train the custom model that classifies whether a graph is required. To do this, run the following command:

```bash
python train.py
```

This will train the model using `train_data.csv` and save the trained model (`trained_roberta`) to a folder.

### Step 3: Run the Streamlit App

Once the model is trained, you can start the interactive Streamlit app by running:

```bash
streamlit run app.py
```

This will start a local Streamlit server and open a browser window where you can upload a CSV file, enter a query, and get insights or visualizations based on your dataset.

### Step 4: Upload Your Data and Query

1. **Upload a CSV file**: Choose a CSV file to analyze. It will be previewed in the app.
2. **Enter your query**: Type a natural language query, such as "Which car sold more quantity?" or "What are the trends in sales?"
3. **View the Insights & Visualization**: The app will use AI to analyze the dataset and generate Python code that visualizes the data. The generated plot will be displayed in the app.

### Step 5: View the Generated Code (Optional)

You can also view the Python code generated by the AI model by clicking on the "Show Code" button after submitting your query. This allows you to see the code that was used to generate the visualization.

## Example Use Case

### Query 1: 
**User**: "Which car sold more quantity?"

- The app will analyze the dataset, extract relevant keywords (e.g., "car", "sold", "quantity"), and decide if a graph is needed.
- The app will generate the corresponding Python code for a visualization (e.g., bar chart or pie chart showing car sales).
- The generated plot will be displayed, and the Python code will be available for review.

### Query 2: 
**User**: "What is the distribution of sales over time?"

- The app will create a time series plot or similar chart to represent sales trends.

## Model Training Pipeline

The training pipeline consists of the following steps:
1. Load the `train_data.csv` which contains queries, keywords, and whether the graph is required.
2. Tokenize the data using the `RobertaTokenizer`.
3. Fine-tune the `RobertaForSequenceClassification` model.
4. Split the data into training and testing datasets.
5. Train the model using the `Trainer` API from Hugging Face's `transformers` library.
6. Save the trained model and tokenizer for future use.

## Contributing

Feel free to open issues, fork the repository, and submit pull requests. Any contributions are welcome!
