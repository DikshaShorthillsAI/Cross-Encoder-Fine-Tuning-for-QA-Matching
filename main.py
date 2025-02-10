import json
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

load_dotenv()

class DataLoaderHandler:
    """Handles data loading and preparation."""
    
    @staticmethod
    def load_data(json_file):
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if "questions" not in data or "contexts" not in data:
            raise KeyError("JSON file must contain 'questions' and 'contexts' keys.")

        return data["questions"], data["contexts"]

class EmbeddingHandler:
    """Handles embedding generation and similarity computations."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, batch_size=64, show_progress_bar=True)

    @staticmethod
    def compute_similarity(q_embeddings, c_embeddings, threshold=0.5):
        """Compute binary similarity scores: 1 if similar, 0 if not."""
        q_embeddings = torch.nn.functional.normalize(q_embeddings, p=2, dim=1)
        c_embeddings = torch.nn.functional.normalize(c_embeddings, p=2, dim=1)
        cosine_similarities = torch.sum(q_embeddings * c_embeddings, dim=1)
        
        # Convert similarity scores to binary labels
        binary_labels = (cosine_similarities > threshold).float().tolist()
        return binary_labels

class CrossEncoderTrainer:
    """Handles training of the cross-encoder model."""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, num_labels=1)
        self.model.to(self.device)

    def prepare_training_data(self, questions, contexts, similarity_scores):
        return [InputExample(texts=[q, c], label=s) for q, c, s in zip(questions, contexts, similarity_scores)]

    def train(self, train_data, output_path="./output/fine-tuned-cross-encoder"):
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)

        # train_loss = losses.SoftmaxLoss(
        #     model=self.model, 
        #     sentence_embedding_dimension=768, 
        #     num_labels=2
        # )
        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=20,
            warmup_steps=2000,
            scheduler="warmupcosine",
            optimizer_params={"lr": 1e-5}, 
            output_path=output_path
        )

        print("Model training completed and saved successfully!")

    def save_model(self, path):
        self.model.save(path)

    @staticmethod
    def load_fine_tuned_model(path="./output/fine-tuned-cross-encoder"):
        return CrossEncoder(path)

class Predictor:
    """Handles model inference and evaluation."""
    
    @staticmethod
    def make_predictions(model, questions, contexts):
        return model.predict([[q, c] for q, c in zip(questions, contexts)], batch_size=64, convert_to_tensor=True)

    @staticmethod
    def save_results_to_json(results, output_file="./output/fine_tuned_results.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Fine-tuned results have been saved in {output_file}")

    @staticmethod
    def prepare_results_for_json(test_questions, test_contexts, original_scores, fine_tuned_scores):
        return [
            {"question": q, "context": c, "original_model_score": float(o), "fine_tuned_model_score": float(f)}
            for q, c, o, f in zip(test_questions, test_contexts, original_scores, fine_tuned_scores)
        ]

def load_test_data(filepath):
    """Loads test questions and contexts from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["test_questions"], data["test_contexts"]

def main():
    data_loader = DataLoaderHandler()
    embedding_handler = EmbeddingHandler()
    cross_encoder_trainer = CrossEncoderTrainer()
    predictor = Predictor()

    print("Loading training data...")
    questions, contexts = data_loader.load_data("assets/data.json")

    print("Generating embeddings...")
    q_embeddings = embedding_handler.generate_embeddings(questions)
    c_embeddings = embedding_handler.generate_embeddings(contexts)

    similarity_scores = embedding_handler.compute_similarity(q_embeddings, c_embeddings)
    train_data = cross_encoder_trainer.prepare_training_data(questions, contexts, similarity_scores)

    print("Training Cross-Encoder model...")
    cross_encoder_trainer.train(train_data)
    cross_encoder_trainer.save_model("./output/fine-tuned-cross-encoder")

    fine_tuned_model = CrossEncoderTrainer.load_fine_tuned_model()

    print("Loading test data...")
    test_questions, test_contexts = load_test_data("assets/test_data.json")

    print("Generating predictions with original and fine-tuned models...")
    original_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    original_scores = predictor.make_predictions(original_model, test_questions, test_contexts)
    fine_tuned_scores = predictor.make_predictions(fine_tuned_model, test_questions, test_contexts)

    results = predictor.prepare_results_for_json(test_questions, test_contexts, original_scores, fine_tuned_scores)
    predictor.save_results_to_json(results)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
