import streamlit as st

from datasets import load_dataset

st.title('Homophobia Detector')

st.markdown(
    """
    This app will detect whether Tweets are homophobic. As hate speech spreads, our aim is to be able to more reliably detect
     when it is present on social media. To test the Tweet, input it into the text box below.
    """
)

Tweet = st.text_input("Copy the Tweet here:")

