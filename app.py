import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
#from datasets import load_dataset

st.title('Homophobia Detector')

st.markdown(
    """
    This app will detect whether Tweets are homophobic. As hate speech spreads, our aim is to be able to more reliably detect
     when it is present on social media. To test the Tweet, input it into the text box below.
    """
)


# MODEL

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

# Model
model = AutoModelForSequenceClassification.from_pretrained("my_model_hate")
model.eval()

# Function to predict homophobic sentiment
def predict_phrase(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return predicted_class_id


# User-input Tweet
Tweet = st.text_input("Copy the Tweet here:")

predicted_label = predict_phrase(Tweet)

st.markdown(predicted_label)



