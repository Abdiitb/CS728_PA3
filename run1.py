import json
import os
from utils import get_queries_and_items, BM25SparseRetriever, DenseRetriever

def calculate_recall_at_k(results: dict[str, dict[str, int]], gold_tools: list[dict], k: int) -> float:
     recall_sum = 0.0
     for query_id, scores in results.items():
          for tool_id, _ in list(scores.items())[:k]: # get top k results
               if tool_id == gold_tools[query_id]:
                    recall_sum += 1.0
                    break
     return recall_sum / len(results)

if __name__ == "__main__":
     # load queries and tools
     train_queries, test_queries, tools = get_queries_and_items()

     # extract gold tool names for train and test queries
     gold_train_tools = {q["qid"]: q["gold_tool_name"] for q in train_queries}
     gold_test_tools = {q["qid"]: q["gold_tool_name"] for q in test_queries}

     # Initialize the retrievers
     bm25_retriever = BM25SparseRetriever()
     msmarco_minilm_retriever = DenseRetriever(model_name="sentence-transformers/msmarco-MiniLM-L12-v3")
     uae_retriever = DenseRetriever(model_name="WhereIsAI/UAE-Large-V1")

     # Fit the retrievers on the tools
     if os.path.exists("saved_models/bm25_retriever"):
          bm25_retriever.load("saved_models/bm25_retriever", documents=tools)
     else:
          bm25_retriever.fit(tools)
     if os.path.exists("saved_models/msmarco_minilm_retriever"):
          msmarco_minilm_retriever.load("saved_models/msmarco_minilm_retriever", documents=tools)
     else:
          msmarco_minilm_retriever.fit(tools)
     if os.path.exists("saved_models/uae_retriever"):
          uae_retriever.load("saved_models/uae_retriever", documents=tools)
     else:
          uae_retriever.fit(tools)

     # Save the retrievers for later use
     bm25_retriever.save("saved_models/bm25_retriever")
     msmarco_minilm_retriever.save("saved_models/msmarco_minilm_retriever")
     uae_retriever.save("saved_models/uae_retriever")

     # Prediction on train and test queries
     train_queries_results_bm25 = bm25_retriever.predict(train_queries)
     test_queries_results_bm25 = bm25_retriever.predict(test_queries)

     train_queries_results_ms_marco_minilm = msmarco_minilm_retriever.predict(train_queries)
     test_queries_results_ms_marco_minilm = msmarco_minilm_retriever.predict(test_queries)

     train_queries_results_uae = uae_retriever.predict(train_queries)
     test_queries_results_uae = uae_retriever.predict(test_queries)

     # Save the results for later evaluation
     with open("results/q1/train_queries_results_bm25.json", "w") as f:
          json.dump(train_queries_results_bm25, f)
     with open("results/q1/test_queries_results_bm25.json", "w") as f:
          json.dump(test_queries_results_bm25, f)
     with open("results/q1/train_queries_results_ms_marco_minilm.json", "w") as f:
          json.dump(train_queries_results_ms_marco_minilm, f)
     with open("results/q1/test_queries_results_ms_marco_minilm.json", "w") as f:
          json.dump(test_queries_results_ms_marco_minilm, f)
     with open("results/q1/train_queries_results_uae.json", "w") as f:
          json.dump(train_queries_results_uae, f)
     with open("results/q1/test_queries_results_uae.json", "w") as f:
          json.dump(test_queries_results_uae, f)

     # Calulate recall@1 and recall@5 for the retrievers on train and test queries
     k_values = [1, 5]
     for k in k_values:
          recall_train_bm25 = calculate_recall_at_k(train_queries_results_bm25, gold_tools=gold_train_tools, k=k)
          recall_test_bm25 = calculate_recall_at_k(test_queries_results_bm25, gold_tools=gold_test_tools, k=k)
          recall_train_ms_marco_minilm = calculate_recall_at_k(train_queries_results_ms_marco_minilm, gold_tools=gold_train_tools, k=k)
          recall_test_ms_marco_minilm = calculate_recall_at_k(test_queries_results_ms_marco_minilm, gold_tools=gold_test_tools, k=k)
          recall_train_uae = calculate_recall_at_k(train_queries_results_uae, gold_tools=gold_train_tools, k=k)
          recall_test_uae = calculate_recall_at_k(test_queries_results_uae, gold_tools=gold_test_tools, k=k)

          # Save the recall results
          with open(f"results/q1/recall_at_{k}.json", "w") as f:
               json.dump({
                    "bm25": {
                         "train": recall_train_bm25,
                         "test": recall_test_bm25
                    },
                    "ms_marco_minilm": {
                         "train": recall_train_ms_marco_minilm,
                         "test": recall_test_ms_marco_minilm
                    },
                    "uae": {
                         "train": recall_train_uae,
                         "test": recall_test_uae
                    }
               }, f)
