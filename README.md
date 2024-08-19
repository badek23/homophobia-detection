# Homophobia Detection with Twitter-roBERTa

This repository contains the implementation of a model for detecting homophobic content on Twitter. The model is based on the `cardiffnlp/twitter-roberta-base-sentiment-latest` pre-trained model from Hugging Face and fine-tuned on a custom dataset. The project also includes data preprocessing, augmentation, and model evaluation.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Data Augmentation](#data-augmentation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)

## Installation

To run this project, you'll need to install the necessary Python libraries:

- transformers
- datasets
- torch
- scikit-learn
- nltk
- matplotlib

## Dataset

The dataset used for training and evaluation is the "HomophobiaDetectionTwitterX" dataset, which can be loaded from the Hugging Face Hub.

## Data Preprocessing

Data preprocessing includes the following steps:

1. **Loading and Cleaning Data:** The dataset is cleaned by removing URLs, mentions, emojis, and special characters. Stopwords are removed, and lemmatization is applied.

2. **Keyword Analysis:** The dataset is analyzed to identify the most common keywords in homophobic and non-homophobic tweets, which can reveal potential biases.

## Data Augmentation

To balance the dataset, data augmentation is performed by replacing frequently occurring keywords with synonyms or similar words, generating new variations of tweets.

## Model Training

The model is fine-tuned using Hugging Face's `Trainer` class. Training configurations include:

- **Epochs:** 3
- **Batch Size:** 4 for training, 8 for evaluation
- **Warmup Steps:** 500
- **Weight Decay:** 0.01

## Model Evaluation

The model is evaluated on a test dataset of unseen tweets, achieving an accuracy of 87%. The evaluation also considers metrics like F1 score and ROC-AUC.

## Deployment

The trained model can be pushed to the Hugging Face Hub and deployed using Streamlit.

1. **Push to Hugging Face Hub:**

- Authenticate with Hugging Face
- Push the model to the Hub

2. **Deploy on Streamlit:** You can deploy the model on Streamlit to make it accessible for real-time inference. Our deployed app: https://homophobia-detector.streamlit.app/

## Acknowledgments

- This project leverages the `cardiffnlp/twitter-roberta-base-sentiment-latest` model and the `HomophobiaDetectionTwitterX` dataset from Hugging Face.
- Special thanks to the all group members who helped in the development and evaluation of this model.

---

This README provides a comprehensive overview of the project, including installation instructions, dataset details, and steps for data preprocessing, augmentation, model training, evaluation, and deployment. For further details, refer to the code files in this repository.

