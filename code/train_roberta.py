from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import numpy as np
import json
from pathlib import Path

new_wat_labels = {
    "correct": "correct",
    "correct.": "correct",
    "correct attribution": "correct",
    "correct for england and wales.": "correct",
    "this is correct, based on new research in the lancet.": "correct",
    "true": "correct",
    "correct (although so does leaving).": "correct",
    "correct, for the nhs in england.": "correct",
    "this is correct.": "correct",
    "this is broadly correct.": "correct",
    "this is correct as of 11 may.": "correct",
    "this is correct, based on new research in the lancet.": "correct",
    "that's correct.": "correct",
    "that's right.": "correct",
    "that’s correct.": "correct",
    "correct, for undergraduate students from the uk who took up their place between 2015 and 2017.": "correct",
    "mostly correct": "correct",
    "mostly-true": "correct",
    "broadly correct.": "correct",
    "mostly true": "correct",
    "exagerated": "partly false",
    "exaggerates": "partly false",
    "understated/exaggerated": "partly false",
    "partly false": "partly false",
    "half-true": "partly false",
    "out of context": "partly false",
    "outdated": "partly false",
    "outdated figure": "partly false",
    "mixed": "partly false",
    "mixture": "partly false",
    "half true": "partly false",
    "miscaptioned": "incorrect",
    "this is false, and the website making the claim is impersonating bbc news.": "incorrect",
    "this is not true.": "incorrect",
    "false attribution": "incorrect",
    "misattributed": "incorrect",
    "distorts the facts": "incorrect",
    "spins the facts": "incorrect",
    "fake": "incorrect",
    "false": "incorrect",
    "faux": "incorrect",
    "hoax": "incorrect",
    "incorrect": "incorrect",
    "incorrect.": "incorrect",
    "mostly false": "incorrect",
    "barely-true": "incorrect",
    "baseless claim": "incorrect",
    "false.": "incorrect",
    "wrong": "incorrect",
    "false headline": "incorrect",
    "labeled satire": "incorrect",
    "satire": "incorrect",
    "pants-fire": "incorrect",
    "scam": "incorrect",
    "twists the facts": "incorrect",
    "doctored image": "incorrect",
    "misleading": "incorrect",
    "cherry picks": "incorrect",
    "not the whole story": "incorrect",
    "altered": "incorrect",
    "altered video": "incorrect",
    "disputed": "incorrect",
    "mueller report refutes that": "incorrect",
    "experts say it won't": "incorrect",
    "false. the extent of antarctic sea ice did peak in 2015, before hitting a low in 2016, but arctic sea ice and the land ice sheets at both poles have been steadily shrinking for decades.": "incorrect",
    "false: distorts facts": "incorrect",
    "falso": "incorrect",
    "it's a doctored video": "incorrect",
    "legal experts disagree": "incorrect",
    "not based in science": "incorrect",
    "spinning the facts": "incorrect",
    "lacks context": "unproven",
    "missing context": "unproven",
    "needs context": "unproven",
    "no evidence": "unproven",
    "there is no evidence for this.": "unproven",
    "there is no evidence of this.": "unproven",
    "there’s no evidence that this works as a preventative or a cure for the virus.": "unproven",
    "there is no evidence for this in the report cited by the claimant, nor elsewhere.": "unproven",
    "there is no evidence for this.": "unproven",
    "there is no evidence he said this.": "unproven",
    "there is no evidence of this.": "unproven",
    "there is no evidence to suggest this test can show if you have the new coronavirus.": "unproven",
    "there’s no evidence that this works as a preventative or a cure for the virus.": "unproven",
    "unproven": "unproven",
    "unsupported": "unproven",
    "legend": "unproven",
    "false: no evidence": "incorrect",
    "lost legend": "unproven",
    "research in progress": "unproven",
}

id2label = {0: "correct", 1: "partly false", 2: "incorrect", 3: "unproven"}
label2id = {"correct": 0, "partly false": 1, "incorrect": 2, "unproven": 3}

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-large", num_labels=4, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )


data = json.load(open("../WatClaimCheck/train.json", "r"))
wat__labels = []
wat__id = []
for d in data:
    wat__labels.append(d["label"]["original_rating"])
    wat__id.append(d["label"]["id"])
wat__labels = np.array(wat__labels, dtype=str)
wat__id = np.array(wat__id, dtype=str)
wat_labels = []
for l in wat__labels:
    wat_labels.append(new_wat_labels[l])
wat_dict = dict(zip(wat__id, wat_labels))

path = Path(f"../dataset/train/review/")
sub = path.glob("*")
sub = list(sub)

dataset = []
for s in sub:
    id = str(s).split("/")[-1]
    cls = id.split("_")[0]
    if cls != "wat":
        continue
    name = id.split("_")[1]
    name = name.split(".")[0]
    review = json.load(open(s, "r"))
    claim = json.load(open(f"../dataset/train/claim/{id}", "r"))
    dataset.append(
        {
            "text": f"Claim: {str(claim)}\nReview:{str(review)}",
            "label": label2id[wat_dict[name]],
        }
    )

dataset = Dataset.from_list(dataset)
dataset.shuffle(seed=42)
encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset, test_dataset = encoded_dataset.train_test_split(test_size=0.1).values()

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir=f"../models/roberta",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)


import evaluate

accuracy = evaluate.load("accuracy")
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")

    metrics = accuracy.compute(predictions=predictions, references=labels)
    metrics["macro_f1"] = macro_f1

    return metrics


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
