import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
import numpy as np
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
zero_shot_model = AutoModelForSequenceClassification.from_pretrained(f'C:/wdst/model-to-be-used').to(device)
zero_shot_tokenizer = AutoTokenizer.from_pretrained(f'C:/wdst/model-to-be-used')
available_tasks = ["pie chart", "median", "mean", "bar graph", "correlation", "standard deviation", 'line plot', 'histogram']
data = pd.read_excel(f'C:\wdst\data.xlsx')
