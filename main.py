import os
import json
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

load_dotenv()


class MongoDBHandler:
    """Handles MongoDB connections and operations."""
    
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_db = os.getenv("MONGO_DB", "embedding_db")
        self.mongo_collection = os.getenv("MONGO_COLLECTION", "cross-encoder_embeddings")

        if not self.mongo_uri:
            raise ValueError("MongoDB URI not found in environment variables.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
        self.collection = self.db[self.mongo_collection]

    def store_embedding(self, question, context, embedding):
        document = {
            "question": question,
            "context": context,
            "embedding": embedding.tolist()
        }
        self.collection.insert_one(document)

    def fetch_all_embeddings(self):
        return list(self.collection.find({}, {"_id": 0}))


class DataLoaderHandler:
    """Handles data loading and preparation."""
    
    @staticmethod
    def load_data(questions_file, contexts_file):
        with open(questions_file, "r") as q_file:
            questions = q_file.readlines()
        with open(contexts_file, "r") as c_file:
            contexts = c_file.readlines()
        return [q.strip() for q in questions], [c.strip() for c in contexts]


class EmbeddingHandler:
    """Handles embedding generation and similarity computations."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, questions, contexts):
        q_embeddings = self.model.encode(questions, convert_to_tensor=True)
        c_embeddings = self.model.encode(contexts, convert_to_tensor=True)
        return q_embeddings, c_embeddings

    @staticmethod
    def compute_similarity(q_embeddings, c_embeddings):
        return torch.nn.functional.cosine_similarity(q_embeddings, c_embeddings).tolist()


class CrossEncoderTrainer:
    """Handles training of the cross-encoder model."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name, num_labels=1)
    
    def prepare_training_data(self, questions, contexts, similarity_scores):
        return [InputExample(texts=[q, c], label=s) for q, c, s in zip(questions, contexts, similarity_scores)]

    def train(self, train_data, output_path="./fine-tuned-cross-encoder"):
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
        train_loss = losses.MSELoss(model=self.model)
        self.model.fit(train_dataloader=train_dataloader, epochs=10, warmup_steps=500, scheduler='warmupcosine', optimizer_params={'lr': 1e-5}, output_path=output_path)
        print("Model training completed and saved successfully!")

    
    def save_model(self, path):
        self.model.save(path)
    
    @staticmethod
    def load_fine_tuned_model(path="./fine-tuned-cross-encoder"):
        return CrossEncoder(path)


class Predictor:
    """Handles model inference and evaluation."""
    
    @staticmethod
    def make_predictions(model, questions, contexts):
        return model.predict([[q, c] for q, c in zip(questions, contexts)])

    @staticmethod
    def save_results_to_json(results, output_file="fine_tuned_results.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Fine-tuned results have been saved in {output_file}")

    @staticmethod
    def compare_scores(test_questions, test_contexts, original_scores, fine_tuned_scores):
        for i, (q, c) in enumerate(zip(test_questions, test_contexts)):
            print(f"Question: {q}\nContext: {c}")
            print(f"Original Model Score: {original_scores[i]:.4f} | Fine-Tuned Model Score: {fine_tuned_scores[i]:.4f}\n")
    
    @staticmethod
    def prepare_results_for_json(test_questions, test_contexts, original_scores, fine_tuned_scores):
        return [
            {"question": q, "context": c, "original_model_score": float(o), "fine_tuned_model_score": float(f)}
            for q, c, o, f in zip(test_questions, test_contexts, original_scores, fine_tuned_scores)
        ]


def main():
    mongo_handler = MongoDBHandler()
    data_loader = DataLoaderHandler()
    embedding_handler = EmbeddingHandler()
    cross_encoder_trainer = CrossEncoderTrainer()
    predictor = Predictor()

    questions, contexts = data_loader.load_data("assets/questions.txt", "assets/contexts.txt")

    q_embeddings, c_embeddings = embedding_handler.generate_embeddings(questions, contexts)
    for q, c, emb in zip(questions, contexts, q_embeddings):
        mongo_handler.store_embedding(q, c, emb)

    similarity_scores = embedding_handler.compute_similarity(q_embeddings, c_embeddings)
    train_data = cross_encoder_trainer.prepare_training_data(questions, contexts, similarity_scores)
    cross_encoder_trainer.train(train_data)
    cross_encoder_trainer.save_model("./fine-tuned-cross-encoder")

    fine_tuned_model = CrossEncoderTrainer.load_fine_tuned_model()

    test_questions = ["What is AI?", "How do neural networks learn?"]
    test_contexts = [
        "AI is a branch of computer science.",
        "Neural networks learn by adjusting weights through backpropagation."
    ]

    original_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    original_scores = predictor.make_predictions(original_model, test_questions, test_contexts)
    fine_tuned_scores = predictor.make_predictions(fine_tuned_model, test_questions, test_contexts)


    predictor.compare_scores(test_questions, test_contexts, original_scores, fine_tuned_scores)
    results = predictor.prepare_results_for_json(test_questions, test_contexts, original_scores, fine_tuned_scores)
    predictor.save_results_to_json(results)


if __name__ == "__main__":
    main()
