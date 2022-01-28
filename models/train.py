from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import logging

 
# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# set format for pytorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

training_args = TrainingArguments(
    output_dir= args.output_dir,          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    do_eval=True
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le_name_mapping), label2id=le_name_mapping)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)


logging.info('Starting Model Training')
trainer.train()

logging.info('Computing Evaluation Accuracy')
outputs = trainer.evaluate()

logging.info(f'{outputs}')
logging.info(f'Eval Accuracy: {outputs['eval_accuracy']})
logging.info('Saving Model')
trainer.save_model()