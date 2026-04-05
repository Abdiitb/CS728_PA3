from __future__ import annotations
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import math
import random
from angle_emb import Prompts
from tqdm import tqdm

def load_model_tokenizer(model_name, device, dtype = torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                output_attentions = True,
                                                dtype=dtype,  
                                                local_files_only = True, # set True when the model is already downloaded
                                                )
    model.to(device)
    model.eval()
    return tokenizer, model

class PromptUtils:
    def __init__(self, tokenizer, doc_ids, dict_all_docs):
        self.dict_doc_name_id = {key:idx for idx, key in enumerate(doc_ids)}
        self.tokenizer = tokenizer
        self.prompt_seperator = " \n\n"
        user_header = '<|start_header_id|>user<|end_header_id|>'
        asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self.item_instruction = f" Here are all the available tools:"
        self.prompt_prefix = user_header + self.item_instruction
        self.prompt_suffix = asst_header
        self.prompt_prefix_length = len(tokenizer(self.prompt_prefix, add_special_tokens=False).input_ids)
        self.prompt_suffix_length = len(tokenizer(self.prompt_suffix, add_special_tokens=False).input_ids)
        
        self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
        self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."

        (
            self.all_docs_info_string, 
            self.doc_names_str, 
            self.doc_lengths,
            self.doc_spans
        ) = self.create_doc_pool_string(doc_ids, dict_all_docs)
        self.add_text1_length = len(tokenizer(self.add_text1, add_special_tokens=False).input_ids)

    
    def create_prompt(self, query):
        query_prompt = f"Query: {query}"+ "\nCorrect tool_id:"
        prompt = self.prompt_prefix + \
                self.all_docs_info_string + \
                self.prompt_seperator + \
                self.add_text1 + \
                self.prompt_seperator + \
                query_prompt + \
                self.prompt_suffix
        return prompt
        

    def create_doc_pool_string(self, shuffled_keys, all_docs):
        doc_lengths = []
        doc_list_str = []
        map_docname_id, map_id_docname = {}, {}
        all_schemas = ""
        doc_spans = []
        doc_st_index = self.prompt_prefix_length + 1 # inlcudes " \n\n"
        for ix, key in enumerate(shuffled_keys):
            value = all_docs[key]
            doc_list_str.append(key)
            text = self.prompt_seperator
            doc_text = self.doc_text(idx=self.dict_doc_name_id[key] + 1, doc_name=key, doc_info=value).strip()
            doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)
            text += doc_text
            doc_spans.append((doc_st_index, doc_st_index + doc_text_len))
            doc_st_index =  doc_st_index + 1 + doc_text_len
            doc_lengths.append(doc_text_len)
            all_schemas += text
            if ix == len(shuffled_keys)-1:
                end_of_docs_index = doc_st_index
        doc_list_str = ", ".join(doc_list_str)    
        return all_schemas, doc_list_str, doc_lengths, doc_spans

class BM25SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents: dict[str, str]) -> None:
        self.documents = documents # tools.json -> contains the tool descriptions -> dict with tool name as key and description as value
        self.N: int = len(documents)
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = {} # dictionary where the key is the term and the value is the number of documents containing the term
        self.doc_lengths: dict[str, int] = {} # dictionary where the key is the document id and the value is the length of the document in terms of number of words
        self.term_freqs: dict[str, dict[str, int]] = {} # nested dictionary where the outer key is the document id and the inner key is the term and the value is the frequency of the term in the document

        # Calculate df(t) -> number of documents containing term t and store in self.doc_freqs
        for doc_id, doc in tqdm(documents.items(), desc="Processing documents for BM25 index"):
            self.avgdl += len(doc.split())
            self.doc_lengths[doc_id] = len(doc.split())
            for term in doc.split():
                # Calculate tf(t, d) -> number of times term t appears in document d and store in self.term_freqs
                if doc_id not in self.term_freqs:
                    self.term_freqs[doc_id] = {}
                if term not in self.term_freqs[doc_id]:
                    self.term_freqs[doc_id][term] = 0
                self.term_freqs[doc_id][term] += 1
                # Calculate df(t) -> number of documents containing term t and store in self.doc_freqs
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                if self.term_freqs[doc_id][term] == 1:  # Only count the document frequency once per document
                    self.doc_freqs[term] += 1

        self.avgdl /= self.N

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _tf(self, term: str, doc_id: str) -> int:
        return self.term_freqs.get(doc_id, {}).get(term, 0)

    def _score(self, query: str, doc_id: str) -> float:
        score_dq = 0.0
        for term in query.split():
            tf = self._tf(term, doc_id)
            if tf == 0:
                continue
            idf = self._idf(term)
            doc_length = self.doc_lengths[doc_id]
            score_dq += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))
        return score_dq
    
    def predict_query(self, query: str) -> dict[str, float]:
        scores = {}
        for doc_id in self.documents:
            scores[doc_id] = self._score(query, doc_id)
        # sort scores in descending order
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return scores
    
    def predict(self, queries: list[dict[str, str | int]]) -> dict[str, dict[str, float]]:
        results = {}
        for query in queries:
            results[query["qid"]] = self.predict_query(query["text"])
        return results

    def save(self, path: str):
        # path contains the directory where the index will be saved
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, "bm25_index.json")
        with open(file_name, "w") as f:
            json.dump({
                "k1": self.k1,
                "b": self.b,
                "N": self.N,
                "avgdl": self.avgdl,
                "doc_freqs": self.doc_freqs,
                "doc_lengths": self.doc_lengths,
                "term_freqs": self.term_freqs
            }, f)

    def load(self, path: str, documents: dict[str, str] = None):
        file_name = os.path.join(path, "bm25_index.json")
        with open(file_name, "r") as f:
            data = json.load(f)
            self.k1 = data["k1"]
            self.b = data["b"]
            self.N = data["N"]
            self.avgdl = data["avgdl"]
            self.doc_freqs = data["doc_freqs"]
            self.doc_lengths = data["doc_lengths"]
            self.term_freqs = data["term_freqs"] 
        if documents is not None:
            self.documents = documents

class DenseRetriever:
    def __init__(self, model_name: str):
        if model_name in ['sentence-transformers/msmarco-MiniLM-L12-v3', 'WhereIsAI/UAE-Large-V1']:
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported. Please choose from 'sentence-transformers/msmarco-MiniLM-L12-v3' or 'WhereIsAI/UAE-Large-V1'.")
        self.model_name = model_name

    def _encode(self, text: str) -> torch.Tensor:
        return self.model.encode(text, convert_to_numpy=True)

    def fit(self, documents: dict[str, str]) -> None:
        self.doc_ids: dict[str, int] = {}
        self.doc_embeddings: np.ndarray = np.zeros((len(documents), self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        for idx, (doc_id, doc_text) in enumerate(documents.items()):
            self.doc_ids[doc_id] = idx
            self.doc_embeddings[idx] = self._encode(doc_text)

    def _score(self, query_embedding: np.ndarray, doc_id: str) -> float:
        doc_idx = self.doc_ids[doc_id]
        doc_embedding = self.doc_embeddings[doc_idx]
        return util.dot_score(query_embedding, doc_embedding)[0].item()
    
    def predict_query(self, query: str) -> dict[str, float]:
        if self.model_name == 'WhereIsAI/UAE-Large-V1':
            query_embedding = self._encode(Prompts.C.format(text=query))
        else:
            query_embedding = self._encode(query)
        scores = {}
        for doc_id in self.doc_ids:
            scores[doc_id] = self._score(query_embedding, doc_id)
    
        # sort scores in descending order
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return scores
    
    def predict(self, queries: list[dict[str, str | int]]) -> dict[str, dict[str, float]]:
        results = {}
        for query in queries:
            results[query["qid"]] = self.predict_query(query["text"])
        return results
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "doc_embeddings.npy"), self.doc_embeddings)
        with open(os.path.join(path, "doc_ids.json"), "w") as f:
            json.dump(self.doc_ids, f)

    def load(self, path: str, documents: dict[str, str] = None):
        self.doc_embeddings = np.load(os.path.join(path, "doc_embeddings.npy")).astype(np.float32)
        with open(os.path.join(path, "doc_ids.json"), "r") as f:
            self.doc_ids = json.load(f)
        if documents is not None:
            self.documents = documents

def get_queries_and_items_check():
    tool_path = "/scratch/deekshak/datasets/MetaTool/dataset/data/all_clean_data.csv"   
    tool_desc_path = "/scratch/deekshak/datasets/MetaTool/dataset/plugin_des.json"
    df =  pd.read_csv(tool_path)
    with open(tool_desc_path) as f:
        dbs = json.load(f)
    queries = []
    map_tool_count = {key: 0 for key in dbs}
    for idx in range(len(df)):
        row = df.iloc[idx]
        queries.append({
            "text": row["Query"],
            "gold_tool_name": row["Tool"],
            "qid": idx
            }
        )
        map_tool_count[row["Tool"]] += 1
    
    tools100 = sorted(map_tool_count.items(), key= lambda x: x[1], reverse=True)[:100]
    tools100 = [i[0] for i in tools100]
    queries_filtered = [i for i in queries if i["gold_tool_name"] in tools100]
    random.shuffle(queries_filtered)
    dbs_filtered = {i:dbs[i] for i in dbs if i in tools100}
    with open("data/test_queries.json", "w") as f: json.dump(queries_filtered[:5000], f)
    with open("data/train_queries.json", "w") as f: json.dump(queries_filtered[5000: 6500], f)
    with open("data/tools.json", "w") as f: json.dump(dbs_filtered, f)
    return queries_filtered, dbs_filtered

def get_queries_and_items():
    with open("data/test_queries.json", "r") as f: test_queries = json.load(f)
    with open("data/train_queries.json", "r") as f: train_queries  = json.load(f)
    with open("data/tools.json", "r") as f: tools = json.load(f)
    return train_queries, test_queries, tools         