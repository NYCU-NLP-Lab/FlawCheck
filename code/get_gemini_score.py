import json
import os

wat_dict = json.load(open("../wat_id.json", "r"))

store = {}
for w in wat_dict.items():
    if not os.path.exists(f"../output/flaw_7/{w[0]}.json"):
        continue
    if w[1] not in store:
        store[w[1]] = [w[0]]
    # elif len(store[w[1]]) < 1:
    store[w[1]].append(w[0])

for i in range(4):
    print(len(list(store.values())[i]))

for k in store.keys():
    for v in store[k]:
        if not os.path.exists(f"../output/flaw_7/{v}.json"):
            print(f"None {v}")

import requests
import json
import re

secret = os.environ["gemeni_token"]
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={secret}"
headers = {"Content-Type": "application/json"}


def get_response(prompt):
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "safety_settings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ],
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = json.loads(response.text)
        if "blockReason" in result["promptFeedback"]:
            print("Blocked!")
            return "Blocked"
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except Exception as e:
        # print(f'Error: {e}')
        return ""


def get_prompt(review, output):
    return f"""**Fact-Checking Quality Assessment**

**Reference Review:**  
{str(review)[:5000]}

**Model's Fact-Checking Output:**  
{str(output)[:2000]}

**Evaluation Criteria:**

**Correctness:**
- Compare the model's output to the reference review.
- Identify any inaccuracies or errors in the output.

**Completeness:**
- Determine if the output addresses all key points from the reference review.
- Note any significant omissions or gaps in the information provided.
- Evaluate the sufficiency of context and detail for understanding the fact-checked claims.

**Scoring:**
- Rate the correctness of the output on a scale of 0-1 (0: Incorrect, 1: Fully correct).
- Rate the completeness of the output on a scale of 0-1 (0: Incomplete, 1: Fully complete).

**Format:**
- Output the scoring follow this format: 
- Correctness: 0.5, Completeness: 0.5

Please submit your scores for both correctness and completeness based on the criteria above.
"""


import re


def get_score(text, id):
    # text = 'Correctness: 0.75\nCompleteness: 0.75'
    pattern = r"Correctness:\s*(\d(?:\.\d+)?)[\s,]*Completeness:\s*(\d(?:\.\d+)?)"

    match = re.search(pattern, text)
    if match:
        scores = (float(match.group(1)), float(match.group(2)))
        # print(scores)  # Output will be: (1.0, 0.7)
        return scores
    else:
        print(f"No match found: {id}, {text}")
        return None


import numpy as np

for model in ["baseline", "aspect", "flaw_3", "flaw_5", "flaw_7"]:
    for label in store.keys():
        print(model, label)
        correct = []
        complete = []
        for id in store[label]:
            review = json.load(open(f"../dataset/test/review/{id}.json", "r"))
            output = json.load(open(f"../output/{model}/{id}.json", "r"))
            prompt = get_prompt(review, output)
            s = None
            while s == None:
                response = ""
                while response == "":
                    response = get_response(prompt=prompt)
                if response == "Blocked":
                    s = "Pass"
                    continue
                s = get_score(response, id)
            if s == "Pass":
                continue
            correct.append(s[0])
            complete.append(s[1])
        output = {"correct": correct, "complete": complete}
        json.dump(output, open(f"./gemini_{model}_{label}.json", "w"))
        correct = np.array(correct)
        complete = np.array(complete)
        print(correct.mean(), complete.mean())
