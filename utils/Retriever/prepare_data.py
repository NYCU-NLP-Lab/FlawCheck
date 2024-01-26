from pathlib import Path
import random
import json

channel = "train"

output = []
path = Path(f"../../dataset/{channel}/review")
sub = list(path.glob("*.json"))
for pos in sub:
    datapoint = {}
    neg = random.choice(sub)
    while neg == pos:
        neg = random.choice(sub)
    info = str(pos).replace("review", "info")
    with open(info, "r") as file:
        claim = json.load(file)
    datapoint["question"] = claim["claim"]
    datapoint["answers"] = []
    datapoint["negative_ctxs"] = []
    datapoint["positive_ctxs"] = []
    datapoint["hard_negative_ctxs"] = []
    with open(str(pos), "r") as file:
        pos_review = json.load(file)
    with open(str(neg), "r") as file:
        neg_review = json.load(file)
    start = False
    text = ""
    for j, content in enumerate(pos_review):
        if "true" in content:
            start = True
        if "Our Sources" in content:
            break
        if start:
            text += content
    datapoint["positive_ctxs"].append({"title": "", "text": text})
    start = False
    text = ""
    for j, content in enumerate(neg_review):
        if "true" in content:
            start = True
        if "Our Sources" in content:
            break
        if start:
            text += content
    datapoint["hard_negative_ctxs"].append({"title": "", "text": text})
    output.append(datapoint)
with open(f"./data/{channel}.json", "w") as file:
    json.dump(output, file)
