from http import HTTPStatus
from lambdas.inference.model_inference import handler
from aws_lambda_powertools.utilities.typing import LambdaContext

def generate_context() -> LambdaContext:
    context = LambdaContext()
    context._aws_request_id = '888888'
    return context

def test_handler_200_ok():
    response = handler({"model_id": "Kayvane/distilbert-complaints-product", "text": "Coinbase stole all of my money" }, generate_context())
    assert response['statusCode'] == HTTPStatus.OK