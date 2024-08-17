import pyterrier as pt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
if not pt.started():
    pt.init()

dataset = pt.get_dataset("irds:beir/hotpotqa")
indexer = pt.IterDictIndexer("./index")
index_ref = indexer.index(dataset.get_corpus_iter())

# Define retrieval models
def tfidf_retrieval(index_ref, query):
    tfidf = pt.transformer.TfIdf(index_ref)
    return tfidf.transform(query)

def bm25_retrieval(index_ref, query):
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    return bm25.transform(query)


# Main function to handle queries
def retrieve_documents(query, model="tfidf"):
    if model == "tfidf":
        return tfidf_retrieval(index_ref, query)
    elif model == "bm25":
        return bm25_retrieval(index_ref, query)
    else:
        raise ValueError("Unsupported model type. Choose from 'tfidf' or 'bm25'")


if __name__ == "__main__":
    query = "multi-hop QA"
    results = retrieve_documents(query, model="bm25")
    print(results)
