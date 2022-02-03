from datasets import load_metric
from utils import get_data
from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification
import torch
import logging
import numpy as np
import os

def create_complaints_model(text_column, feature_column, output_dir="distilbert-complaints",from_checkpoint=None, evaluate=False):

    # Set-up directories
    model_name = f"{output_dir}-{feature_column}"
    if not os.path.exists(f"{output_dir}"):
        os.mkdir(os.path.join(os.getcwd(),output_dir))
        logging.info(f'Created Directory')
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    text_dataset, class_mapping, label2id, number_classes = get_data(text_column, feature_col)

    # tokenizer used in preprocessing
    tokenizer_name = 'distilbert-base-uncased'

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # tokenize dataset
    train_dataset = text_dataset['train'].map(tokenize, batched=True)
    test_dataset = text_dataset['test'].map(tokenize, batched=True)

    # set format for pytorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    metric = load_metric("accuracy")

    training_args = TrainingArguments(
        output_dir= model_name,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=f"{output_dir}/logs",            # directory for storing logs
        logging_steps=250,
        do_eval=True,
        push_to_hub=True,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=number_classes, label2id= label2id, id2label = class_mapping).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    )
    if from_checkpoint is not None:
        logging.info(f'Starting Model Training from checkpoint {from_checkpoint}')
        trainer.train(f"output_dir/{model_name}/{from_checkpoint}")
    else:
        logging.info('Starting Model Training')
        trainer.train()

    if evaluate:
        logging.info('Computing Evaluation Accuracy')
        outputs = trainer.evaluate()
        logging.info(f'{outputs}')

    logging.info(f"Eval Accuracy: {outputs['eval_accuracy']}")
    logging.info('Saving Model')
    trainer.save_model()

    trainer.push_to_hub()