import os
import pandas as pd
import runpy
import spacy
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import google.generativeai as genai
import streamlit as st
import matplotlib.pyplot as plt


#runpy.run_path('train.py')


KEY = 'YOUR_GOOGLE_API_KEY_HERE'
genai.configure(api_key=KEY)
generation_config = {
"temperature": 1,
"top_p": 0.95,
"top_k": 40,
"max_output_tokens": 8192,
"response_mime_type": "text/plain"}
model = genai.GenerativeModel( model_name="gemini-1.5-flash", generation_config=generation_config,)
chat_session = model.start_chat(history=[])

nlp = spacy.load("en_core_web_sm")
transformer_model = RobertaForSequenceClassification.from_pretrained("./trained_roberta")
tokenizer = RobertaTokenizer.from_pretrained("./trained_roberta")


def extract_keywords_and_predict_graph_requirement(query):
    doc = nlp(query)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = transformer_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    graph_requirement = "needed" if prediction == 1 else "not needed"
    return keywords, graph_requirement


st.title("Data Insights and Visualization Generation")
st.write("Upload a CSV file to analyze its contents and get graph recommendations based on your query.")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", dataset.head())
    with st.form(key="query_form"):
        user_query = st.text_input("Enter your query:", "What insights can be drawn from the dataset?")
        submit_button = st.form_submit_button("Submit Query")
        if submit_button:
            if user_query:
                keywords_query, graph_flag_query = extract_keywords_and_predict_graph_requirement(user_query)
                query = f"""
                dataset insights:
                {dataset.info()}
                {dataset.describe(include='all')}

                Dataset:
                {dataset}

                query: "{user_query}"
                graph/visual_representation: {graph_flag_query} (generate python matplotlib code accordingly)
                Use keywords: {keywords_query} to estimate the type of graph.

                NOTE: Do not include citations or sources and decorative explanations.
                Just return the code.
                Full dataset available at 'csv_path = '{uploaded_file}'
                """

                response = chat_session.send_message(query)
                python_code = response.text.replace("```python", "\n").replace("```", "\n")
                with open("plt.py", "w") as fh:
                    fh.write(python_code)

                runpy.run_path('plt.py')
                st.pyplot(plt.gcf())
                st.code(python_code, language="python")

else:
    st.write("Please upload a CSV file to get started.")

