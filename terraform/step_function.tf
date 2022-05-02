
# Need to parametrise with Terraform functions
module "step_function" {
  source = "terraform-aws-modules/step-functions/aws"

  name       = "ComplaintsInferrence"
  definition = <<EOF
{
  "Comment": "A description of my state machine",
  "StartAt": "Get Model Names",
  "States": {
    "Get Model Names": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "OutputPath": "$.Payload",
      "Parameters": {
        "Payload.$": "$",
        "FunctionName": "arn:aws:lambda:us-east-1:987199941948:function:get-models:$LATEST"
      },
      "Retry": [
        {
          "ErrorEquals": [
            "Lambda.ServiceException",
            "Lambda.AWSLambdaException",
            "Lambda.SdkClientException"
          ],
          "IntervalSeconds": 2,
          "MaxAttempts": 6,
          "BackoffRate": 2
        }
      ],
      "Next": "Map"
    },
    "Map": {
      "Type": "Map",
      "Iterator": {
        "StartAt": "Call Inference Model Lambdas",
        "States": {
          "Call Inference Model Lambdas": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "OutputPath": "$.Payload",
            "Parameters": {
              "Payload.$": "$",
              "FunctionName": "arn:aws:lambda:us-east-1:987199941948:function:model_inference:$LATEST"
            },
            "Retry": [
              {
                "ErrorEquals": [
                  "Lambda.ServiceException",
                  "Lambda.AWSLambdaException",
                  "Lambda.SdkClientException"
                ],
                "IntervalSeconds": 2,
                "MaxAttempts": 6,
                "BackoffRate": 2
              }
            ],
            "End": true
          }
        }
      },
      "ItemsPath": "$.models",
      "MaxConcurrency": 5,
      "Next": "write to DynamoDB Table",
      "InputPath": "$",
      "ResultPath": "$.text"
    },
    "write to DynamoDB Table": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "OutputPath": "$.Payload",
      "Parameters": {
        "Payload.$": "$",
        "FunctionName": "arn:aws:lambda:us-east-1:987199941948:function:dynamodb-write:$LATEST"
      },
      "Retry": [
        {
          "ErrorEquals": [
            "Lambda.ServiceException",
            "Lambda.AWSLambdaException",
            "Lambda.SdkClientException"
          ],
          "IntervalSeconds": 2,
          "MaxAttempts": 6,
          "BackoffRate": 2
        }
      ],
      "End": true
    }
  }
}
EOF

  service_integrations = {
    dynamodb = {
      dynamodb = ["arn:aws:dynamodb:eu-west-1:052212379155:table/Test"]
    }

    lambda = {
      lambda = ["arn:aws:lambda:eu-west-1:123456789012:function:test1", "arn:aws:lambda:eu-west-1:123456789012:function:test2"]
    }
  }

  type = "STANDARD"

  tags = {
    Module = "complaints-model-inference"
  }
}