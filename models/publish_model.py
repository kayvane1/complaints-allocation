from boto3 import TypeSerializer, resource, Attr, 

# Class to publish model metadata to dynamodb

class Model:
    def __init__(self, model_name: str, model_task: str, model_metrics: dict, model_version: int):
        self.model_name = model_name
        self.model_task = model_task
        self.model_description = None
        self.active = False
        self.model_version = None
        self.model_metrics = None
        self.dynamodb = resource('dynamodb')
        self.table = self.dynamodb.Table('model_info')
        self.serializer = TypeSerializer()
    
    def publish(self):
        item = {
            'model_name': self.model_name,
            'model_task': self.model_task,
            'model_description': self.model_description,
            'active': self.active,
            'model_version': self.model_version,
        }
        dyn_item = {key: self.serializer.serialize(value) for key, value in item.items()}
        self.table.put_item(Item=dyn_item)
    
    def get_champion_model(self):
      # get current active model for the same model task
      response = self.table.scan(FilterExpression=Attr('model_task').eq(self.model_task) & Attr('active').eq(True))
      return response['Items'] if response is not None else None
    
    def _get_model_version(self):
      # get current active model for the same model task
      response = self.table.scan(FilterExpression=Attr('model_task').eq(self.model_task) & Attr('active').eq(True))
      self.model_version =  int(response['Items']['model_version']) + 1 if response is not None else 1

    
    

    
    