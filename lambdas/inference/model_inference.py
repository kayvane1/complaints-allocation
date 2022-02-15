from http import HTTPStatus
from typing import Any, Dict
import os
from transformers import pipeline
from aws_lambda_powertools.logging.logger import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext

os.environ["PYTHONIOENCODING"]="utf8"

def load_model(model_id: str) -> pipeline:
    """Load a model from a model name

    Args:
        model_name (str): The name of the model to load

    Returns:
        model (transformers.pipeline.Pipeline): The loaded model
    """
    return pipeline(
        task="text-classification",
        model=model_id,
    )


# JSON output format, service name can be set by environment variable "POWERTOOLS_SERVICE_NAME"
logger: Logger = Logger(service='model-inference-lambda')


def handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:

    logger.set_correlation_id(context.aws_request_id)
    logger.debug(
        f"{event['model_id']} model being used by lambda function for inference")

    pipe = load_model(event['model_id'])
    event['prediction'] = pipe(event['text'])[0]

    return {'statusCode': HTTPStatus.OK,
            'headers': {'Content-Type': 'application/json'},
            'body': event
            }
