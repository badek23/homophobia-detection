import streamlit as st
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

@st.cache_resource
def get_model():
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = RobertaForSequenceClassification.from_pretrained("gvbl92/HomphobiaDetection-roBERTa")
    return tokenizer, model

tokenizer, model = get_model()

# Define labels dictionary
d = {
    1: 'Homophobic',
    0: 'Not Homophobic'
}

# Function to predict using the model
def predict(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return logits, predicted_class  # Return logits along with predicted class

# Streamlit app
def main():
    st.title("Homophobia Detector")

    # Input text area and button
    user_input = st.text_area('Enter Text to Analyze')
    button = st.button("Analyze")

    if user_input and button:
        logits, prediction = predict(user_input)
        st.write("Logits: ", logits)
        st.write("It's giving... ", d[prediction])

if __name__ == "__main__":
    main()