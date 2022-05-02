import torch
import logging
import os
import time
import argparse
import wandb
import sys
from pathlib import Path
from datasets import load_dataset, Dataset
from model_utils import LabelAnalyser, sampling_strategy, undersample_df, compute_metrics
from torch import tensor, nn, device, cuda
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.trainer_callback import EarlyStoppingCallback
from huggingface_hub import HfFolder
from datasets.dataset_dict import DatasetDict

def get_training_data(text_column:str, target_column:str):

  """
  Fetches the Complaints dataset from the HuggingFace Hub and generates id and class mappings
  """  

  # Loading consumer complaints dataset
  text_dataset = load_dataset("consumer-finance-complaints", ignore_verifications=True)
  assert type(text_dataset) == DatasetDict
  # Splitting the dataset into training and validation datasets
  text_dataset = text_dataset['train'].train_test_split(test_size=0.2,seed=0)

  # Extracting the target column
  columns = text_dataset['train'].column_names
  assert target_column in columns, "Target column not found in dataset"
  keep_cols = [e for e in columns if e not in (text_column, target_column)]
  text_dataset = text_dataset.remove_columns(keep_cols)

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train new HuggingFace Model for a given two column dataset')
    parser.add_argument('--text_col', dest='text_col', type=str, help='The name of the text column in the dataset')
    parser.add_argument('--feature_col', dest='feature_col', type=str, help='The name of the text column in the dataset')
    parser.add_argument('--model_id', dest='model_id', type=str, default='distilbert-base-uncased', help='Name of the HF model to use')
    parser.add_argument('--dataset_id', dest='dataset_id', type=str, default='consumer-complaints', help='Name of the HF Hub dataset to use')
    parser.add_argument('--experiment_name', dest='experiment_name', default='complaints_dataset', type=str, help='path to the csv file')
    parser.add_argument('--from_checkpoint', dest='from_checkpoint', type=str, help='If training should start from a specific checkpoint')
    parser.add_argument('--evaluate', dest='evaluate', type=str, help='If training should evaluate the final model performance')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    text_dataset, id2label, label2id, number_classes = get_training_data(args.text_col, args.feature_col)

    logging.info("Analysing Label Distribution")
    train_dataset = text_dataset['train'].to_pandas()
    label_info = LabelAnalyser(train_dataset)
    logging.info(label_info)

    # If the dataset is imbalanced, we want to generate class weights for the loss function to penalise more for rarer classes
    if label_info.label_distribution != 'balanced':
      logging.info("Dataset is imbalanced, generating class weights and undersampling majority class")
      # Sampling majority to the mean value
      target_classes, other_classes = sampling_strategy(train_dataset['text'],train_dataset['labels'],round(train_dataset.labels.value_counts().mean()),t='majority')
      df = undersample_df(train_dataset,target_classes,other_classes)
      logging.info("Recalculating label distribution")
      label_info = LabelAnalyser(df)
      logging.info(label_info)
      weights = label_info.generate_class_weights()
      # The weights need to be converted to a torch tensor and loaded into the GPU to be accessible by the Trainer
      allocated_weights = tensor(weights)
      weights = allocated_weights.to(device)
    else:
      weights = None

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # tokenizer helper function
    def tokenize(batch):
        tokenized_batch = tokenizer(batch['text'], padding='max_length', truncation=True)
        tokenized_batch["labels"] = [label_info.str2int[label] for label in batch["labels"]]
        return tokenized_batch

    # tokenize dataset
    train_dataset = text_dataset['train'].map(tokenize, batched=True)
    test_dataset = text_dataset['test'].map(tokenize, batched=True)

    # set format for pytorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    if device == "cuda":
      fp_16 = True
    else:
      fp_16 = False

    logging.info("Setting up Trainer Args")
    
    output_dir = Path("/opt/ml/output/data")
        
    logging.info(f'Created Directory')
    training_args = TrainingArguments(
    output_dir=output_dir.as_posix(),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    warmup_steps=args.warmup_steps,
    fp16=fp_16,
    learning_rate=float(args.learning_rate),
    # logging & evaluation strategies
    logging_dir=f"{output_dir.as_posix()}/logs",
    logging_steps=50, 
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="wandb",
    # push to hub parameters
    push_to_hub=args.push_to_hub,
    hub_strategy=args.hub_strategy,
    hub_model_id=args.hub_model_id,
    hub_token=args.hub_token,
)

    # define data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logging.info("Initializing Model")
    model = AutoModelForSequenceClassification.from_pretrained(
    args.model_id, num_labels=number_classes, label2id=label2id, id2label=id2label
)
    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )

    logging.info('Starting Model Training')
    trainer.train()

    logging.info('Computing Evaluation F1')
    outputs = trainer.evaluate()
    
    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_dir.as_posix(), "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(outputs.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")
    logging.info(f'{outputs}')
    logging.info(f"Eval Accuracy: {outputs['eval_f1']}")
    logging.info('Saving Model')
    trainer.save_model()
    wandb.finish()

    # save best model, metrics and create model card
    if args.push_to_hub:
        trainer.create_model_card(model_name=args.hub_model_id)
        # wait for asynchronous pushes to finish
        time.sleep(180)
        trainer.push_to_hub()
        
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])