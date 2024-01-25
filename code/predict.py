import torch
from pathlib import Path
import json
from tqdm import tqdm
import torch
from fastchat.model import load_model, get_conversation_template, add_model_args

model_path = "lmsys/vicuna-7b-v1.5"
revision = "main"
device = "cuda"
gpus = None
num_gpus = 1
max_gpu_memory = None
load_8bit = True
cpu_offloading = False
gptq_ckpt = None
gptq_wbits = 16
gptq_groupsize = -1
gptq_act_order = False
awq_ckpt = None
awq_wbits = 16
awq_groupsize = -1
temperature = 0.7
repetition_penalty = 1.0
max_new_tokens = 512
debug = False

model, tokenizer = load_model(
    model_path,
    device=device,
    num_gpus=num_gpus,
    max_gpu_memory=max_gpu_memory,
    load_8bit=load_8bit,
    cpu_offloading=cpu_offloading,
    revision=revision,
    debug=debug,
)


@torch.inference_mode()
def main(msg):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # print(f"{conv.roles[0]}: {msg}")
    # print(f"{conv.roles[1]}: {outputs}")
    return outputs


import evaluate

rouge_scorer = evaluate.load("rouge")
bleu_scorer = evaluate.load("bleu")
bertscore_scorer = evaluate.load("bertscore")


def get_baseline_prompt(claim, evidence):
    return f"""Validate the veracity of the given claim based on the provided evidence and generate a detailed explanation.
Claim: {claim}
Collected Evidence: {str(evidence)[:2500]}"""


rouge1 = {}
rouge2 = {}
rougeL = {}
rougeLsum = {}
bleu = {}
precision = {}
recall = {}
f1 = {}

path = Path(f"./Datasets/RefuteClaim/test/evidence/")
sub = path.glob("*")
for file in tqdm(list(sub)):
    id = str(file).split("/")[-1]
    name = id.split(".")[0]
    claim = json.load(open(f"./Datasets/RefuteClaim/test/claim/{id}", "r"))
    evidence = json.load(open(f"./Datasets/RefuteClaim/test/evidence/{id}", "r"))
    review = json.load(open(f"./Datasets/RefuteClaim/test/review/{id}", "r"))
    review = str(review)
    prompt = get_baseline_prompt(claim, evidence)

    with torch.cuda.device("cuda"):
        response = main(prompt)
    json.dump(response, open(f"./output/baseline/{id}", "w"))
    score = rouge_scorer.compute(predictions=[response], references=[review])
    rouge1[name] = score["rouge1"]
    rouge2[name] = score["rouge2"]
    rougeL[name] = score["rougeL"]
    rougeLsum[name] = score["rougeLsum"]

    score = bleu_scorer.compute(predictions=[response], references=[review])
    bleu[name] = score["bleu"]

    score = bertscore_scorer.compute(
        predictions=[response], references=[review], lang="en"
    )

    precision[name] = score["precision"]
    recall[name] = score["recall"]
    f1[name] = score["f1"]

output = {
    "rouge1": rouge1,
    "rouge2": rouge2,
    "rougeL": rougeL,
    "rougeLsum": rougeLsum,
    "bleu": bleu,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

with open(f"./baseline_result.json", "w") as file:
    json.dump(output, file)
