import pandas as pd
import pathlib
from abc import ABC, abstractmethod
import time
from typing import List, Tuple
import numpy as np

# Import different libraries for each approach
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

DATA_PATH = pathlib.Path(__file__).parent / "data" / "denials.csv"

class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies"""

    def __init__(self, name: str):
        self.name = name
        self.setup_time = 0
        self.query_time = 0

    @abstractmethod
    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        """Initialize the retrieval system"""
        pass

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        """Retrieve relevant documents"""
        pass

class BM25Strategy(RetrievalStrategy):
    """Pure BM25 retrieval using rank_bm25"""

    def __init__(self):
        super().__init__("BM25")
        self.bm25 = None
        self.df = None
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]

    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        start_time = time.time()

        self.df = df
        tokenized_docs = [self.tokenize(doc) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        self.setup_time = time.time() - start_time

    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()

        tokenized_query = self.tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        self.query_time = time.time() - start_time
        return self.df.iloc[top_indices], scores[top_indices]

class TFIDFStrategy(RetrievalStrategy):
    """Enhanced TF-IDF with n-grams"""

    def __init__(self):
        super().__init__("Enhanced TF-IDF")
        self.vectorizer = None
        self.doc_vectors = None
        self.df = None

    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        start_time = time.time()

        self.df = df
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
        self.doc_vectors = self.vectorizer.fit_transform(docs)

        self.setup_time = time.time() - start_time

    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()

        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        self.query_time = time.time() - start_time
        return self.df.iloc[top_indices], similarities[top_indices]

class EmbeddingStrategy(RetrievalStrategy):
    """Pure sentence transformer embeddings"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(f"Embeddings ({model_name})")
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.df = None

    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        start_time = time.time()

        self.df = df
        self.doc_embeddings = self.model.encode(docs)

        self.setup_time = time.time() - start_time

    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()

        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(question_embedding, self.doc_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        self.query_time = time.time() - start_time
        return self.df.iloc[top_indices], similarities[top_indices]

class HybridStrategy(RetrievalStrategy):
    """BM25 filtering + Embedding re-ranking"""

    def __init__(self, bm25_candidates: int = 50, final_top_k: int = 15):
        super().__init__(f"Hybrid (BM25→Embeddings)")
        self.bm25_strategy = BM25Strategy()
        self.embedding_strategy = EmbeddingStrategy()
        self.bm25_candidates = bm25_candidates
        self.final_top_k = final_top_k

    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        start_time = time.time()

        # Setup both strategies
        self.bm25_strategy.setup(docs, df)
        self.embedding_strategy.setup(docs, df)

        self.setup_time = time.time() - start_time

    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()

        # Step 1: BM25 filtering to get candidates
        bm25_results, bm25_scores = self.bm25_strategy.retrieve(question, self.bm25_candidates)
        candidate_indices = bm25_results.index.tolist()

        # Step 2: Re-rank candidates using embeddings
        candidate_docs = [f"Department: {row['department']}. Denial reason: {row['denial_reason']}. Patient age: {row['patient_age']}. Payer: {row['payer']}. Amount: ${row['amount']}"
                         for _, row in bm25_results.iterrows()]

        candidate_embeddings = self.embedding_strategy.model.encode(candidate_docs)
        question_embedding = self.embedding_strategy.model.encode([question])

        rerank_similarities = cosine_similarity(question_embedding, candidate_embeddings).flatten()
        final_top_indices = rerank_similarities.argsort()[-top_k:][::-1]

        # Get final results
        final_df_indices = [candidate_indices[i] for i in final_top_indices]
        final_similarities = rerank_similarities[final_top_indices]

        self.query_time = time.time() - start_time
        return self.bm25_strategy.df.loc[final_df_indices], final_similarities

class EnsembleStrategy(RetrievalStrategy):
    """Ensemble voting from multiple strategies"""

    def __init__(self, strategies: List[RetrievalStrategy], weights: List[float] = None):
        super().__init__("Ensemble")
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        self.df = None

    def setup(self, docs: List[str], df: pd.DataFrame) -> None:
        start_time = time.time()

        self.df = df
        for strategy in self.strategies:
            strategy.setup(docs, df)

        self.setup_time = time.time() - start_time

    def retrieve(self, question: str, top_k: int = 15) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()

        # Get results from all strategies
        all_scores = np.zeros(len(self.df))

        for strategy, weight in zip(self.strategies, self.weights):
            results, scores = strategy.retrieve(question, len(self.df))  # Get all results
            result_indices = results.index.tolist()

            # Normalize scores to 0-1 range
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized_scores = scores

            # Add weighted scores
            for idx, score in zip(result_indices, normalized_scores):
                all_scores[idx] += weight * score

        # Get top-k from ensemble scores
        top_indices = all_scores.argsort()[-top_k:][::-1]

        self.query_time = time.time() - start_time
        return self.df.iloc[top_indices], all_scores[top_indices]

def load_docs():
    """Load and prepare documents"""
    df = pd.read_csv(DATA_PATH)
    docs = df.apply(lambda r: f"Department: {r['department']}. Denial reason: {r['denial_reason']}. Patient age: {r['patient_age']}. Payer: {r['payer']}. Amount: ${r['amount']}", axis=1).tolist()
    return docs, df