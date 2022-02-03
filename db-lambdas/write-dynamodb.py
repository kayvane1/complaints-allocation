import json
import boto3
from boto3.dynamodb.types import TypeSerializer

serializer = TypeSerializer()
dynamodb = boto3.client('dynamodb')

def handler(event, context):
    table_name = event.get("table_name")
    item = event.get("item")
    dyn_item = {key: serializer.serialize(value) for key, value in item.items()}
    dynamodb.put_item(TableName=table_name, Item=dyn_item)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }
