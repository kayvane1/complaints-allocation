import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_metric
import numpy as np
import torch

class LabelAnalyser:
  """
  Helper class to understand a dataset's label distribution.
  Generates some basic statistics about the labels.
  Utilities to visualises the distribution of labels in the dataset.
  Utilities to set-up the model for training - int2str & str2int
  Utilities to generate class weights for imbalanced datasets.
  """
  def __init__(self, df: pd.DataFrame):
    assert (df.columns == 'labels').any()
    self.df = df
    self.label_counts = df.labels.value_counts()
    self.label_names = df.labels.unique().tolist()
    self.num_labels = len(self.label_names)
    self.str2int = {name: i for i, name in enumerate(self.label_names)}
    self.int2str = {v: k for k, v in self.str2int.items()}
    self.mean = np.mean(self.label_counts)
    self.std = np.std(self.label_counts)
    self.min_max_ratio = np.max(self.label_counts) / np.min(self.label_counts)
    self.min_max_std = (np.max(self.label_counts) - np.min(self.label_counts) )/ self.std
  
  def __repr__(self):
    return f"Labels(num_classes: {self.num_labels}, mean: {self.mean}, std : {self.std}, min_max_ratio: {self.min_max_ratio}, min_max_std: {self.min_max_std})"

  def one_std(self):
    """
    Returns a list of values that are within 1 standard deviations of the mean.
    """
    return [x for x in self.label_counts if (x < self.mean + self.std) & (x > self.mean - self.std)]

  def two_std(self):
    """
    Returns a list of values that are within 2 standard deviations of the mean.
    """
    return [x for x in self.label_counts if (x < self.mean + 2 * self.std) & (x > self.mean - 2 * self.std)]
  
  def visualise_top_20(self):
    """
    Visualises the top 20 labels in the dataset.
    """
    return self.df.labels.value_counts().sort_values(ascending=False)[:20].plot(kind = 'barh')
  
  def visualise_bottom_20(self):
    """
    Visualises the bottom 20 labels in the dataset.
    """
    return self.df.labels.value_counts().sort_values(ascending=True)[:20].plot(kind = 'barh')

  def visualise_distribution(self):
    """
    Visualises dataset distribution.
    """
    return self.df.hist()

  def label_distribution(self):
    """
    Assesses the distribution of labels in the dataset
    And returns a description of the distribution.

    TODO: To be improved

    Returns:
        [str]: Returns either 'balanced', 'imbalanced' or very_imbalanced'
    """

    if self.std == 0:
      return "balanced"
    elif self.min_max_ratio >= 2 and self.min_max_ratio <= 5 : 
      return "imbalanced"
    elif self.min_max_ratio > 5:
      return "very_imbalanced"
    else:
      return "undetermined" 
  
  def generate_class_weights(self):
    """
    Generates class weights for the labels in the dataset.
    To be used when the data is imbalanced.

    Returns:
        [list]: ordered list of floats
    """
    class_weights = compute_class_weight(
                                          class_weight = "balanced",
                                          classes = self.label_names,
                                          y = self.df.labels                                                    
                                      )
    class_weights = dict(zip(np.unique(self.df.labels), class_weights))

    return [class_weights[v].astype(np.float32) for k, v in self.int2str.items()]


def sampling_strategy(text,labels,n_samples, t='majority'):
  """
  Helper function to identify classes above of below the input target samples
  TODO: To be improved
  """
  
  if t == 'majority':
      target_classes = labels.value_counts() > n_samples
      other_classes = labels.value_counts() < n_samples

  elif t == 'minority':
      target_classes = labels.value_counts() < n_samples
      other_classes = labels.value_counts() > n_samples

  tc = target_classes[target_classes == True].index
  oc = other_classes[other_classes == True].index

  sampling_strategy = {target: n_samples for target in tc}

  oversampling_class = sampling_strategy
  return oversampling_class, oc

def undersample_df(df, t, other_classes):
  """
  Simplified sampler function to undersample the dataframe.
  """
  undersampled_df = df.head(0)
  for k, v in t.items():
    sampled = df[df.labels == k].sample(v)
    undersampled_df = undersampled_df.append(sampled)
  for i in other_classes:
    sampled = df[df.labels == i]
    undersampled_df = undersampled_df.append(sampled)
  return undersampled_df

def compute_metrics(eval_pred):
  # define metrics and metrics function
  f1_metric = load_metric("f1")
  accuracy_metric = load_metric( "accuracy")
  recall_metric = load_metric("recall")
  precision_metric = load_metric("precision")
  
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  acc = accuracy_metric.compute(predictions=predictions, references=labels)
  recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
  f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
  precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")

  return {
      "accuracy": acc["accuracy"],
      "f1": f1["f1"],
      "recall": recall["recall"],
      "precision" : precision["precision"]
  }

