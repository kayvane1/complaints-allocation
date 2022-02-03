import shap
import numpy as np
import scipy as sp
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import logging
 
# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'
# load a BERT sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = DistilBertForSequenceClassification.from_pretrained(
    "Kayvane/distilbert-complaints-product"
)

# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

def handler(event, context):
    
    text = event.get("text")

    shap_values = explainer(text, fixed_context=1)
    response = {
        "statusCode": 200,
        "body": shap_values
    }
    return response
