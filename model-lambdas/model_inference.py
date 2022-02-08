import json
from transformers import pipeline

def load_model(model_name):
    """Load a model from a model name
    
    Args:
        model_name (str): The name of the model to load
    
    Returns:
        model (transformers.pipeline.Pipeline): The loaded model
    """
    return pipeline(
        task="text-classification",
        model=model_name,
    )


#TODO: Expand response to include model-id, all predictions and scores

def handler(event, context):
    
    pipe = load_model(event['model_name'])
    return {
        "statusCode": 200,
        "body": pipe(event['text'])[0]
    }
