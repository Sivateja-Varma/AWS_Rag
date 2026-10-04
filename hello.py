# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to generate text using a Mistral AI model.
"""
import json
import boto3
from botocore.exceptions import ClientError



def generate_text(model_id, body):
    """
    Generate text using a Mistral AI model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        JSON: The response from the model.
    """


    bedrock = boto3.client(service_name='bedrock-runtime')

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id
    )

    return response


def main():
    """
    Entrypoint for Mistral AI example.
    """



    try:
        model_id = 'mistral.mistral-7b-instruct-v0:2'

        prompt = """<s>[INST] whats aws bedrock [/INST]"""

        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50
        })

        response = generate_text(model_id=model_id,
                                 body=body)

        response_body = json.loads(response.get('body').read())

        outputs = response_body.get('outputs')

        for index, output in enumerate(outputs):

            print(f"Output {index + 1}\n----------")
            print(f"Text:\n{output['text']}\n")
            print(f"Stop reason: {output['stop_reason']}\n")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        print("A client error occured: " +
              format(message))
    else:
        print(f"Finished generating text with Mistral AI model {model_id}.")


if __name__ == "__main__":
    main()
