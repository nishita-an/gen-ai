from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, docs):

        self.docs = docs

        self.texts = [
            d.page_content.split()
            for d in docs
        ]

        self.bm25 = BM25Okapi(self.texts)


    def retrieve(self, query, k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(

            list(zip(self.docs, scores)),

            key=lambda x: x[1],

            reverse=True

        )

        return [doc for doc, _ in ranked[:k]]