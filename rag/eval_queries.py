# Retrieval Evaluation Script with 5 Sample Queries

from rag.retriever import retrieve

def run_retrieval_evaluation():
    """
    Runs 5 sample retrieval-quality test queries against the persistent Chroma vector store,
    printing query metadata and chunk contents for manual evaluation.
    """
    eval_queries = [
        "best time of year to visit Nuwara Eliya",
        "train options between Kandy and Ella",
        "budget accommodation in Mirissa",
        "safety advice for solo travelers",
        "national parks good for wildlife safaris"
    ]

    print("=======================================================================")
    print("      SRI LANKA TOURISM PLANNER — RAG RETRIEVAL QUALITY EVALUATION     ")
    print("=======================================================================\n")

    for idx, query in enumerate(eval_queries, 1):
        print(f"--- Query {idx}: '{query}' ---")
        results = retrieve(query, k=3)
        
        for rank, res in enumerate(results, 1):
            print(f"  [Chunk #{rank}]")
            print(f"  Category   : {res['category']}")
            print(f"  Source File: {res['source_file']}")
            print(f"  Content    :\n{res['text']}\n")
        
        # Relevance placeholder comment for review assessment
        # Relevance: [to be filled in after reviewing output]
        print("-----------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_retrieval_evaluation()
