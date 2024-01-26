from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import concurrent.futures
import json
from tqdm import tqdm
import os

retriever = DensePassageRetriever.load(load_dir="./models", document_store=None)


def get_ruling(claim, review):
    docs = []
    for s in review:
        docs.append({"content": s})
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    retriever.document_store = document_store
    top_k_documents = retriever.retrieve(claim, top_k=30)

    result = []
    candi = []
    for doc in top_k_documents:
        if doc.score < 0.6:
            break
        candi.append(doc.content)
    for s in review:
        if s in candi:
            result.append(s)
    return result


def get_evidence(claim, evidence):
    docs = []
    for s in evidence:
        docs.append({"content": s})
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    retriever.document_store = document_store
    top_k_documents = retriever.retrieve(claim, top_k=50)

    candi = []
    for doc in top_k_documents:
        if doc.score < 0.6:
            break
        candi.append(doc.content)
    result = []
    for s in evidence:
        if s in candi:
            result.append(s)
    return result


# WatClaimCheck
def run_thread_wat(x):
    id = x["label"]["id"]
    if not os.path.exists(f"../../WatClaimCheck/review/{id}.json"):
        return
    with open(f"../../WatClaimCheck/review/{id}.json", "r") as file:
        review = json.load(file)
    if review == []:
        return

    if input_split == "valid":
        output_split = "dev"
    else:
        output_split = input_split

    json.dump(
        review, open(f"../../dataset/{output_split}/review/wat_{id}.json", "w")
    )
    claim = x["metadata"]["claim"]
    ruling = get_ruling(claim, review)
    files = x["metadata"]["premise_articles"]
    if output_split == "test":
        evidence_pool = []
        for article in files.values():
            with open(f"../../WatClaimCheck/articles/{article}", "r") as file:
                evidence = json.load(file)
            evidence_pool += evidence
        select_evidence = get_evidence(claim, evidence_pool)

        with open(
            f"../../dataset/{output_split}/claim/wat_{id}.json", "w"
        ) as file:
            json.dump(claim, file)
        with open(
            f"../../dataset/{output_split}/evidence/wat_{id}.json", "w"
        ) as file:
            json.dump(select_evidence, file)

    with open(
        f"../../dataset/{output_split}/selected_review/wat_{id}.json", "w"
    ) as file:
        json.dump(ruling, file)


for input_split in ["train", "valid", "test"]:
    with open(f"../../WatClaimCheck/{input_split}.json", "r") as file:
        data = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        tqdm(executor.map(run_thread_wat, data), total=len(data))
