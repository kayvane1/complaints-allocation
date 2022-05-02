from transformers import AutoTokenizer, pipeline , DistilBertForSequenceClassification
import torch
from torch.nn.functional import cross_entropy
import shap
import numpy as np
import scipy as sp 
import logging
import pandas as pd
from typing import Any, Dict
from datasets.dataset_dict import DatasetDict

class ModelAnalyser:
  """
  class to analyse model's performance
  """
  def __init__(self, dataset, tokenizer, model, int2str):
    self.dataset = dataset
    self.tokeniser = tokenizer
    self.model = model
    self.int2str = int2str
    self.vocab_size = tokenizer.vocab_size
    self.model_max_length = tokenizer.model_max_length
    self.model_input_names = tokenizer.model_input_names
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def forward_pass_with_labels(self, batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(self.device) for k, v in batch.items() if k in self.model_input_names}

    #Run model in inference mode
    with torch.no_grad():
      output = self.model(**inputs)
      pred_label = torch.argmax(output.logits, axis=-1)
      loss = cross_entropy(output.logits, batch["label"].to(self.device), reduction="none")

      return {
        "loss": loss.cpu().numpy(),
        "predicted_label": pred_label.cpu().numpy()
      }

  def return_loss_df(self): 
  
    # Convert the dataset to PyTorch tensors
    self.dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Comput loss values 
    assert len(self.dataset>0)
    prediction_with_loss = self.dataset.map(self.forward_pass_with_labels, batched=True, batch_size=16)

    # Change to pandas for df output
    self.dataset.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_test = self.dataset[:][cols]
    df_test["label"] = df_test["label"].apply
    return



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