import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from sentence_transformers import SentenceTransformer, util

dataset = load_dataset("ag_news", split="train")
id2label = {str(i): label for i, label in enumerate(dataset.features["label"].names)}
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForTokenClassification.from_pretrained('gpt2')
model2 = SentenceTransformer('all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_text(examples):
    inputs = tokenizer(
        examples["text"], padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs["attention_mask"])
    return {"embedding": pooled_embeds.cpu().numpy()}


small_set = (
    dataset.shuffle(42) # randomly shuffles the data, 42 is the seed
           .select(range(1000)) # we'll take 1k rows
           .map(embed_text, batched=True, batch_size=128) # and apply our function above to 128 articles at a time
)

n_rows = range(len(small_set))
small_set = small_set.add_column("idx", n_rows)

def convertToLabel(label_num):
    return id2label[str(label_num)]

labels = list(map(convertToLabel, small_set['label']))
small_set = small_set.add_column("label_names", labels)
dim_size = len(small_set[0]['embedding'])

client = QdrantClient("localhost", port=6333)

collection_name = "news_embeddings"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=dim_size, distance=models.Distance.COSINE),
)

payloads = small_set.select_columns(["label_names", "text"]).to_pandas().to_dict(orient="records")

client.upsert(
    collection_name=collection_name,
    points=models.Batch(
        ids=small_set["idx"],
        vectors=small_set["embedding"],
        payloads=payloads
    )
)

client.scroll(
    collection_name=collection_name, 
    limit=10,
    with_payload=False, # change to True to see the payload
    with_vectors=False  # change to True to see the vectors
)

print("what news u lookiing for")
query = input()
query_embed = np.array(embed_text({"text":query})["embedding"]).flatten()

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embed,
    limit=3
)

print()
print()
print(search_result)

def get_st_embedding(example):
    example['embedding_2'] = model2.encode(example['text'])
    return example

small_set = small_set.map(get_st_embedding)

second_collection = "better_news"
client.recreate_collection(
    collection_name=second_collection,
    vectors_config=models.VectorParams(
        size=len(small_set[0]['embedding_2']),
        distance=models.Distance.COSINE
    )
)

client.upsert(
    collection_name=second_collection,
    points=models.Batch(
        ids=small_set['idx'],
        vectors=small_set['embedding_2'],
        payloads=payloads
    )
)


print("what news u lookiing for pt 2")
query2 = input()
query_embed2 = model2.encode(query2)

search_result = client.search(
    collection_name=second_collection,
    query_vector=query_embed2,
    limit=3
)

print()
print()
print(search_result)
