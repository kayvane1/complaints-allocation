from transformers import AutoTokenizer, pipeline , DistilBertForSequenceClassification
import torch
import shap
import numpy as np
import scipy as sp 
import logging

def explain_predictions(tokenizer_name, model_name, text, chart_type="text", return_all_predictions = False):

  # load a DistilBERT sentiment analysis model
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  model = DistilBertForSequenceClassification.from_pretrained(
      model_name
  )
  pred = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=return_all_predictions)

  # define a prediction function
  def f(x):
      tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
      outputs = model(tv)[0].detach().cpu().numpy()
      scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
      val = sp.special.logit(scores[:,1]) # use one vs rest logit units
      return val

  # build an explainer using a token masker
  explainer = shap.Explainer(pred)

  # explain the model's predictions on incorrect examples
  assert type(text) in {list, tuple, np.ndarray}
  shap_values = explainer(text)

  assert chart_type in ['text', 'waterfall', 'bar']
  if chart_type == 'text':
    chart = shap.plots.text(shap_values.abs.max(0))
  elif chart_type == 'waterfall':
    chart = shap.plots.waterfall(shap_values.abs.max(0))
  elif chart_type == 'bar':
    chart = shap.plots.bar(shap_values.abs.max(0))
  else:
    raise ValueError("Invalid chart type")
  
  return shap_values, chart