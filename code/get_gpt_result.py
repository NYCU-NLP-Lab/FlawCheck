import requests
import json
import logging
import re
import os
import concurrent.futures
from pathlib import Path

logging.basicConfig(filename='gpt_result.log', level=logging.ERROR, format='[%(threadName)s] %(message)s')

url = "https://api.openai.com/v1/chat/completions"
key = os.environ['openai_token']


def get_response(prompt):
    payload={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type":"application/json"
    }
    try:
        res = requests.post(url=url, data=json.dumps(payload), headers=headers)
        res = json.loads(res.text)
        return res['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f'Error: {e}, Response: {res}')
        return ""
    

def get_bias_prompt(claim, review, biases):
    selected_bias = ""
    for i, bias in enumerate(biases):
        if i == len(biases) - 1:
            selected_bias += bias
        else:
            selected_bias += bias + ", "
    return f"""According to the expert's verdict, what biases exist in the provided claim.
Selected Biases: {selected_bias}
Claim: {claim}
Expert's verdict: {str(review)[:2500]}
Consistently generate a numbered list of results for selected biases, even if you think it's hard to determine."""


def get_aspect_prompt(claim, review):
    return f"""According to the expert's verdict, enumerate four distinct aspects that were employed to verify the authenticity of the claim.
Claim: {claim}
Expert's verdict: {str(review)[:2500]}
Consistently generate a numbered list of four distinct aspects, even if you think it's hard to determine."""


def get_dict_result(response):
    pattern = re.compile(r"\d+\.[^\w]*([\w\s]+)[^\w]*(.*)")

    matches = pattern.findall(response)
    dict_result = {item.strip(): explanation.strip() for item, explanation in matches}
    
    return dict_result


def handle_by_id(id):
    review = json.load(open(f"./Datasets/FlawCheck/{split}/review/{id}", "r"))
    claim = json.load(open(f"./Datasets/FlawCheck/{split}/claim/{id}", "r"))
    biases = ["Contradicting facts", "Exaggeration", 'Understatement', "Occasional faltering", "Insufficient support", "Problematic assumptions", "Existence of alternative explanations"]
    
    bias_prompt = get_bias_prompt(claim, review, biases)
    aspect_prompt = get_aspect_prompt(claim, review)

    bias_response = ""
    while bias_response == "":
        bias_response = get_response(bias_prompt)
    json.dump(bias_response, open(f"./openai/bias/{id}", "w"))
    bias_dict = get_dict_result(bias_response)
    if bias_dict == {}:
        logging.error(f"Error: Can't get bias dict - {id}")
    else:
        json.dump(bias_dict, open(f"./Datasets/FlawCheck/{split}/bias/{id}", "w"))

    aspect_response = ""
    while aspect_response == "":
        aspect_response = get_response(aspect_prompt)
    json.dump(aspect_response, open(f"./openai/aspect/{id}", "w"))
    aspect_dict = get_dict_result(aspect_response)
    if aspect_dict == {}:
        logging.error(f"Error: Can't get aspect dict - {id}")
    else:
        json.dump(aspect_dict, open(f"./Datasets/FlawCheck/{split}/aspect/{id}", "w"))

    return

if __name__ == "__main__":
    for split in ['test']:
        path = Path(f"./Datasets/FlawCheck/{split}/review/")
        sub = list(path.glob('pub_*.json'))

        ids = []
        for sub_path in sub:
            id = str(sub_path).split('/')[-1]
            import os
            if os.path.exists(f"./openai/aspect/{id}"):
                continue
            ids.append(id)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(handle_by_id, ids)
