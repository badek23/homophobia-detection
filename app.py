import streamlit as st
import numpy 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch

@st.cache_resource
def get_model():
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = RobertaForSequenceClassification.from_pretrained("gvbl92/HomphobiaDetection-roBERTa")
    return tokenizer, model

tokenizer, model = get_model()

# Define labels dictionary
d = {
    1: 'homophobic.',
    0: 'not homophobic.'
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

    # Get likelihood of prediction
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    likelihood = classifier(input_text)
    score = likelihood[0]["score"]

    return logits, predicted_class, score  # Return logits along with predicted class, and % likelihood of prediction

# Streamlit app
def main():
    
    st.title("Anti-Gaydar")

    with st.container(border=True):

        # Input text area and button
        user_input = st.text_area('Paste the Tweet here to analyze:')
        button = st.button("Analyze")

        if user_input and button:
            logits, prediction, score = predict(user_input)
            st.write(f"It's giving... **{d[prediction]}** There is a {round(score*100,2)}% change that that statement was {d[prediction]}")

            if d[prediction] == "not homophobic.":
                st.write(":balloon::balloon::rainbow-flag::rainbow-flag::innocent::innocent::face_with_cowboy_hat::face_with_cowboy_hat:")
            elif d[prediction] == "homophobic.": 
                st.write(":thumbsdown::thumbsdown::japanese_ogre::japanese_ogre::broken_heart::broken_heart::disappointed::disappointed:")

if __name__ == "__main__":
    main()