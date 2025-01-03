import os
#os.system('pip install -U -q "google-generativeai>=0.8.2" pandas==2.2.2 python-dotenv matplotlib spacy transformers torch python -m spacy download en_core_web_sm scikit-learn datasets pip install transformers[torch]')

import google.generativeai as genai
import pandas as pd
import runpy
import spacy
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


KEY = 'AIzaSyBX2e7LY5v1C5E_OfEvZMEERYevcBP_zy8'

genai.configure(api_key=KEY)
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)
csv_path = 'data.csv'
dataset = pd.read_csv(csv_path)
dataset_str = dataset.head(20).to_string(index=False)
user_query = input("Enter your query: ")


nlp = spacy.load("en_core_web_sm")

model = RobertaForSequenceClassification.from_pretrained("./trained_roberta")
tokenizer = RobertaTokenizer.from_pretrained("./trained_roberta")

def extract_keywords_and_predict_graph_requirement(query):
    doc = nlp(query)
    
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
            keywords.append(token.text)
    
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    graph_requirement = "needed" if prediction == 1 else "not needed"
    
    return keywords, graph_requirement

keywords_query, graph_flag_query = extract_keywords_and_predict_graph_requirement(user_query)


query = f"""
dataset insights:
{dataset.info()}
{dataset.describe(include='all')}

Dataset:
{dataset_str}

query: "{user_query}"
graph/visual_representation: {graph_flag_query} (generate python matplotlib code acoordinglly)
Use keywords: {keywords_query} to estimate the type of graph.


NOTE:Do not include citations or sources and decorative explanations.
Just return the code.
Full dataset available at 'csv_path = 'data.csv'
"""

response = chat_session.send_message(query)

fh = open("plt.py", "w")
fo = response.text
fo = fo.replace("```python", "\n").replace("```", "\n")


fh.write(fo)
fh.close()
runpy.run_path('plt.py')


print(response.text)
