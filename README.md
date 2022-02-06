# Blueprint: End to End Complaints Management


This project aims to show how to build an end to end solution for complaints management, hosted in AWS. It will go through all the key stages of the delivery lifecycle.

Tools & Infrastructure:

**Infrastructure Provisioning**: Terraform  
**Orchestration** : AWS Step Functions  
**Modelling**: Transformers   
**Model Management**: HuggingFace Hub  
**Front End**: Budibase  

TODO List:
- Write Requirements Framework for ML Projects 🚧
- Write Model Training Script ✅
- Make Dataset publicly available through HuggingFace Datasets - PR Open ✅
- Write Terraform based Serverless Model Inference Endpoints ✅
- Use HuggingFace Hub as Model Registry ✅
- Build ML Ops layer for model retraining, champion vs. challenger assessment, stretch-goal: canary deployment 🚧
- Build Step Function Orchestrations 🚧
- Build Model Explainability using Shap 🚧
- Build Complaints Data Store on DynamoDB ✅
- Build Front-end on Budibase 🚧
- Write Budibase deployment scripts in Terraform
- Write Blog posts for End to End 🚧

# Problem Description

The complaints department in the fictional _MegaBank_ recieves customer complaints and enquiries which need to be triaged to the right teams. In order to do this there is a triage team which screens the incoming complaints and routes them to the right product team. Each product team has it's own recurring issues it is aware of and has policies and procedures to resolve them in a systematic way, in addition to those, new issues may arise where a complaints analyst will need to use their best judgement to address the complaint. _MegaBank_ also has a commitment to the regulator to ensure vulnerable customers (the elderly / ex-service people) follow a different customer journey to ensure they are seen to by qualified analysts.  
  
The complaints department has collected a large amount of complaints which have labels to help route them appropriately to the right product team, highlighting the most probable issue and with corresponding labels for vulnerable customers where applicable.  

The Business would like to explore re-deploying the triage team to the product teams and to use an ML Based Triage system in it's place.

# Success Metrics

By analysing the re-allocation rate from one product team to another, it has been inferred that the accuracy rate of the triage team when sending complaints to the product team is 80%. Similarly, when comparing the initial issue to the final issue stated on file, a accuracy rate of 71% has been recorded. Finally, the team is able to identify vulnerable customers with a 60% recall based on the first interaction.

Meeting the team's existing performance with a confidence interval ± 3% has been agreed with the business as model which meets the success criteria, which can be deployed to production.

# Requirements & Constraints

## Functional Requirements

- The implementation must not change the current ways of working of the product team
- The implementation must be designed with a feedback loop in place so it can adapt to new issues
- Decision explainability is not required at the time of allocation but should be available if requested by the regulator

