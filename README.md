# Personalized Medicine: Tailoring Treatment Plans with Generative AI

This project uses the T5 transformer model to predict disease and treatment based on user-input symptoms. It employs the Hugging Face `transformers` library for model training and `gradio` for the user interface.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Gradio Interface](#gradio-interface)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

>>Install the required libraries directly in a Google Colab notebook:


!pip install transformers==4.30.2 accelerate==0.21.0 gradio pandas numpy scikit-learn

>>Load your dataset (Training_with_Treatments.csv):

import pandas as pd

data = pd.read_csv('/content/Training_with_Treatments.csv')

>> Import Libraries

import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

>> Initialize Tokenizer and Model

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

>>Data Cleaning and Preprocessing
Prepare input-output pairs and split data into training and validation sets.

>>Create Dataset Class
Create a custom SymptomsDataset class to handle data for training.

>>Define Training Arguments and Initialize Trainer
Set up training arguments and initialize the Trainer object.

>>Train the Model
Train the model using the prepared dataset and training arguments.

>>Prediction
Define Functions for Prediction
Define functions to convert symptoms to a binary vector and predict disease and treatment.

>>Gradio Interface
Create and Launch Gradio Interface
Create a Gradio interface to input symptoms and get predictions.

>>Usage
Load the Data: Upload Training_with_Treatments.csv to the appropriate directory in Google Colab.
Run the Notebook: Execute the cells in the Google Colab notebook sequentially.
Interact with the Interface: Use the Gradio link to enter symptoms and receive predictions.


