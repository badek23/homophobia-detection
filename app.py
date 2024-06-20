import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Sidebar

# Main content
st.title('Homophobia Detector')
st.markdown(
    """
    This app will detect whether Tweets are homophobic. As hate speech spreads, our aim is to be able to more reliably detect
     when it is present on social media. To test the Tweet, input it into the text box below.
    """
)

Tweet = st.text_input("Copy the Tweet here:")

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('JoshMcGiff/HomophobiaDetectionTwitterX') # Pre-trained model
    model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert") #Fine-tuned model -> We should consider pushing to HuggingFace
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
