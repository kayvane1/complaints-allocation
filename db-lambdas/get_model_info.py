import boto3
import random
from boto3.dynamodb.conditions import Key, Attr

def get_active_models():
    """Get all active models from dynamodb table 'model_info' .

    Returns:
        [list]: A list of dictionaries with metadata related to each active model
    """    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('model-info')
    response = table.scan(FilterExpression=Attr('active').eq("True"))
    return response['Items']


def get_model_deployment_strategy(model_task: str):
    """Get deployment strategy where there are more than 1 active models for a single modelling task

    Args:
        model_task (str): A model task is a modelling objective of the pipeline

    Returns:
        [dict]: A dictionary with the deployment strategy for a given model task
    """    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('model-strategy')
    response = table.scan(FilterExpression=Attr('task').eq(model_task))
    return response['Items']

def canary_deployment_strategy(models: dict, model_dict: dict, deployment_strategy: dict):
    """Filter models by a deployment strategy .

    Args:
        models (list): List of models which will be returned by the handler for the downstream process
        model_dict (dict): List of active models fetched by the handler
        deployment_strategy (dict): Deployment strategy for a given model task

    Returns:
        [list]: List of models which will be returned by the handler for the downstream process
    """    

    # Get the model with the highest allocation score
    dominant_model = max(deployment_strategy['allocation'], key=lambda x:x['allocation_perc'])
    
    for allocation_strategy in deployment_strategy['allocation']: # list of dict of model_name and allocation percentage
        # generate random int between 0 and 100
        random_int = random.randint(0, 100)
        model_name = allocation_strategy['model_id']
        allocation_percentage = allocation_strategy['allocation_perc']
        
        # if random_int is less than allocation_percentage, then return model_name
        if random_int < allocation_percentage:
            models.extend([model for model in model_dict if model['model_id'] == model_name])
        else:
            continue
    # If both random generations are greater than allocation_percentage, then return the dominant model_name
    if model_name is None:
            models.extend([model for model in model_dict if model['model_id'] == dominant_model['model_id'] ])
    return models


def handler(event: dict, context: dict):  # sourcery skip: remove-pass-elif
    """Creates a list of models to be used by the downstream lambda process.
    The function fetches the active models from the dynamodb table 'model_info' and the deployment strategy for a given model task from the dynamodb table 'model_strategy'.
    The final list of models is then processed parrallel using a Mapped Operation by the Step-Function

    Args:
        event (dict): event passed by the Step-Function containing the complaint text to be resolved by the pipeline
        context (dict): event context passed by the Step-Function

    Returns:
        [dict]: event containing the complaint text and the models which need to process the text
    """    
    model_dict = get_active_models()
    
    print(model_dict)

    # Pull model tasks from list of active models to infer deployment strategies to use if there is more than 1 active model for a task
    model_tasks = [model['task'] for model in model_dict]
    tasks = {i: model_tasks.count(i) for i in model_tasks}

    # Model dictionary with deployment strategy for each task - each dictionary gets mapped to a model inference lambda through a mapped step-function parrallelisation
    models = []

    for task, value in tasks.items():
        if value > 1:
            deployment_strategy = get_model_deployment_strategy(task)[0]

            if deployment_strategy['strategy'] == 'canary':
                models = canary_deployment_strategy(models, model_dict, deployment_strategy)
            
            elif deployment_strategy['strategy'] == 'shadow':
                #TODO
                pass
        else:
            model_info = [model for model in model_dict if model['task'] == task]
            models.append(model_info[0])

    event['models'] = models
    return event



