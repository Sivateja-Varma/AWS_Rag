import boto3 
import json 


bedrock=boto3.client('bedrock-runtime')

model_id='mistral.mistral-7b-instruct-v0:2'

prompt='''<s>you are a helpful assisstant that gives answers in a neat and formatted way
[INST]what is the best local model? [/INST]</s> '''

body={
    "prompt": prompt,
    "max_tokens" : 400,   
    "temperature": 0.7,
    "top_p": 0.5,
    "top_k": 4
}
json_body=json.dumps(body)

response=bedrock.invoke_model(body=json_body,modelId=model_id)
output=json.loads(response.get('body').read())
out=output.get('outputs')
for i in out:
  print(i['text'])