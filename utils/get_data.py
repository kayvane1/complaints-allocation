from datasets import load_dataset

def get_training_data(text_column, target_column):

  # Loading consumer complaints dataset
  text_dataset = load_dataset("consumer-finance-complaints")

  # Splitting the dataset into training and validation datasets
  text_dataset = text_dataset['train'].train_test_split(test_size=0.2,seed=0)

  # Extracting the target column
  columns = ['Date Received', 'Product','Sub Product', 'Issue', 'Sub Issue', 'Company Public Response', 'Company', 'State', 'Zip Code', 'Tags', 'Consumer Consent Provided', 'Submitted via', 'Date Sent To Company', 'Company Response To Consumer', 'Timely Response', 'Consumer Disputed', 'Complaint ID']
  assert target_column in columns, "Target column not found in dataset"
  columns.remove(target_column)
  text_dataset = text_dataset.remove_columns(columns)

  class_mapping = {
      idx: text_dataset['train'].features[target_column].int2str(idx)
      for idx, names in enumerate(
          text_dataset['train'].features[target_column].names)
  }
  label2id = {
      text_dataset['train'].features[target_column].int2str(idx): idx
      for idx, names in enumerate(
          text_dataset['train'].features[target_column].names)
  }
  text_dataset = text_dataset.rename_column(text_column, "text")
  text_dataset = text_dataset.rename_column(target_column, "labels")

  number_classes = text_dataset['train'].features['labels'].num_classes

  # Filtering out empty/no-text complaints
  text_dataset = text_dataset.filter(lambda example: len(example['text'])>0)

  return text_dataset, class_mapping, label2id, number_classes
