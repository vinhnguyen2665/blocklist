from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")  # ~400MB

    def question(self, prompt):
        X = self.embedder.encode([prompt])
        print(X)
