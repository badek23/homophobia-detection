import streamlit as st
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load tokenizer and model
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

    # Get logits from the model output
    logits = outputs.logits[0]  # Assuming batch size is 1

    # Softmax operation is already applied during model training for this model
    probabilities = torch.softmax(logits, dim=0)
    
    # Get predicted class
    predicted_class = torch.argmax(logits, dim=0).item()

    return probabilities, predicted_class  # Return probabilities along with predicted class

# Streamlit app
def main():
    st.title("Homophobia Detector")

    # Input text area and button
    user_input = st.text_area('Enter Text to Analyze')
    button = st.button("Analyze")

    if user_input and button:
        probabilities, prediction = predict(user_input)
        
        # Get the score (probability) for the predicted class
        score = probabilities[prediction].item()
        
        st.write("Score: ", score)
        st.write("It's giving... ", d[prediction])

if __name__ == "__main__":
    main()
