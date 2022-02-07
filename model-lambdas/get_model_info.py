import json
import boto3
import numpy as np

# Get all active models from dynamodb table 'model_info'
def get_active_models():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('model_info')
    response = table.scan(FilterExpression=Attr('active').eq(True))
    return response['Items']

# Get deployment strategy where there are more than 1 active models for a single modelling task
def get_model_deployment_strategy(model_task):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('model_strategy')
    response = table.scan(FilterExpression=Attr('task').eq(model_task))
    return response['Item']

def handler(event, context):

    model_dict = get_active_models()

    # Pull model tasks from list of active models to infer deployment strategies to use if there is more than 1 active model for a task
    model_tasks = [model['task'] for model in model_dict]
    tasks = {i: model_tasks.count(i) for i in model_tasks}
    
    for task, value in tasks.items():
        if value > 1:
            deployment_strategy = get_model_deployment_strategy(task)
            
            if deployment_strategy['strategy'] == 'canary':
                dominant_model = max(deployment_strategy, key=lambda x:x['allocation'])
                for allocation_strategy in deployment_strategy['allocation'] # list of dict of model_name and allocation percentage
                    # generate random int between 0 and 100
                    random_int = int(np.random.uniform(0, 100))
                    model_name = allocation_strategy['model_name']
                    allocation_percentage = allocation_strategy['allocation_percentage']
                    
                    # if random_int is less than allocation_percentage, then return model_name
                    if random_int < allocation_percentage:
                        return model_name
                    else:
                        continue
                # If both random generations are greater than allocation_percentage, then return the dominant model_name
                if model_name is None:
                    return dominant_model['model_name']

            elif deployment_strategy == 'shadow':
                #TODO - implement shadow deployment strategy
                pass
            else:
                #TODO - implement other deployment strategies
                pass
        else:
            model_info = filter(lambda model: model_dict['task'] == task, model_dict)
            return model_info['model_name']



