provider "aws" {
  region = "us-east-1"
}

module "deep-learning-lambda" {
  source  = "kayvane1/deep-learning-lambda/aws"
  version = "0.0.4"
  region  = "us-east-1"
  runtime = "python3.8"
  project = "terraform-huggingface-lambda"
  lambda_dir = "lambdas/inference"
  memory =  "4096"
  timeout = "300"
  lambda_mount_path =  "/mnt"
  lambda_transformers_cache = "/mnt/hf_models_cache"
  ecr_repository_name = "huggingface-container-registry"
  ecr_container_image = "transformers-lambda-container"
  ecr_image_tag =  "latest"
  subnet_public_cidr_block = "10.0.0.0/21"
  subnet_private_cidr_block= "10.0.8.0/21"
  vpc_cidr_block = "10.0.0.0/16"
  efs_permissions = "777"
  efs_root_directory = "/mnt"
}

