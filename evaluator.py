import json

import os

from huggingface_hub import InferenceClient

from config import HUGGINGFACE_API_TOKEN



# 1. Initialize the Hugging Face Client

hf_client = InferenceClient(token=HUGGINGFACE_API_TOKEN)



def evaluate_response(prompt, response, criteria):

    # 2. Define the exact JSON structure we want in the prompt

    # DeepSeek-V3 is exceptionally good at following JSON instructions

    json_schema_instruction = """

    You must respond ONLY with a valid JSON object matching this exact schema:

    {

        "accuracy": <integer 1-10>,

        "brevity": <integer 1-10>,

        "tone": <integer 1-10>,

        "overall": <integer 1-10>

    }

    """



    evaluation_prompt = f"""

    You are a strict evaluator.

    Criteria: {criteria}



    Original Prompt: {prompt}

    Response to Evaluate: {response}



    Score from 1-10 on Accuracy, Brevity, and Tone based on the criteria.

    {json_schema_instruction}

    """



    # 3. Call DeepSeek-V3 via Hugging Face Chat Completions

    result = hf_client.chat.completions.create(

        model="deepseek-ai/DeepSeek-V3-0324",

        messages=[

            {"role": "system", "content": "You are a helpful assistant designed to output strict JSON."},

            {"role": "user", "content": evaluation_prompt}

        ],

        # Force the endpoint to return a JSON object

        response_format={"type": "json_object"},

        # Keep temperature low for deterministic, consistent evaluations

        temperature=0.1,

        max_tokens=500

    )



    # 4. Extract the JSON string from the response

    json_string = result.choices[0].message.content



    return json_string



# Example Usage:

# eval_json = evaluate_response("What is 2+2?", "The answer is 4.", "Be precise.")

# print(json.loads(eval_json))