import json
from transformers import pipeline

pipe = pipeline(task="text-classification", model="Kayvane/distilvert-complaints-sub-product", )

def handler(event, context):
    response = {
        "statusCode": 200,
        "body": pipe(event['text'])[0]
    }
    return response

