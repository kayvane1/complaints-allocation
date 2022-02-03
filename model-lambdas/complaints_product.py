import json
from transformers import pipeline

pipe = pipeline(task="text-classification", model="Kayvane/distilvert-complaints-product", )

#TODO: Expand response to include model-id, all predictions and scores

def handler(event, context):
    response = {
        "statusCode": 200,
        "body": pipe(event['text'])[0]
    }
    return response

