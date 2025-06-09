import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RobertaTokenizer
from text_dataset import TextSentimentDataset
from torch.utils.data import DataLoader
from text_improved_train import EnhancedSentimentModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def plot_roc_curves(all_labels, all_probs, class_names, save_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        # Convert to one-vs-all problem
        binary_labels = np.array([1 if l == i else 0 for l in all_labels])
        class_probs = np.array(all_probs)[:, i]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Log AUC scores
    for i, class_name in enumerate(class_names):
        binary_labels = np.array([1 if l == i else 0 for l in all_labels])
        class_probs = np.array(all_probs)[:, i]
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        logger.info(f"AUC for {class_name}: {roc_auc:.3f}")


def plot_probability_distribution(all_probs, all_labels, class_names, save_path):
    """Plot probability distributions for each class"""
    plt.figure(figsize=(15, 5))

    for i, class_name in enumerate(class_names):
        plt.subplot(1, 3, i + 1)

        for j, target_class in enumerate(class_names):
            # Get probabilities for class i where true label is j
            mask = np.array(all_labels) == j
            if np.any(mask):
                class_probs = np.array(all_probs)[mask, i]
                plt.hist(class_probs, alpha=0.5, bins=20, label=f"True {target_class}")

        plt.title(f"{class_name} Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Probability distribution plots saved to {save_path}")


def load_best_model(checkpoint_dir="./classification_checkpoints"):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and f.startswith('model_epoch_')]

    if not checkpoints:
        logger.warning("No checkpoints found! Using final model if available.")
        if os.path.exists(os.path.join(checkpoint_dir, "final_model.pt")):
            return os.path.join(checkpoint_dir, "final_model.pt")
        else:
            raise FileNotFoundError("No model checkpoints or final model found!")

    f1_scores = []
    for checkpoint in checkpoints:
        try:
            f1_str = checkpoint.split('_f1_')[1].split('.pt')[0]
            f1_scores.append((checkpoint, float(f1_str)))
        except (IndexError, ValueError):
            logger.warning(f"Could not extract F1 score from checkpoint name: {checkpoint}")

    if not f1_scores:
        logger.warning("Could not extract F1 scores from checkpoint names. Using last checkpoint.")
        return os.path.join(checkpoint_dir, checkpoints[-1])

    f1_scores.sort(key=lambda x: x[1], reverse=True)
    best_checkpoint = f1_scores[0][0]
    logger.info(f"Selected best checkpoint: {best_checkpoint} with F1: {f1_scores[0][1]}")
    return os.path.join(checkpoint_dir, best_checkpoint)


def test_model(model_path, test_file="data/test.csv", output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    test_df = pd.read_csv(test_file, sep=";")
    has_sentiment_labels = "Sentiment" in test_df.columns
    has_audio_id = "Audio_ID" in test_df.columns

    if has_audio_id:
        logger.info("Audio ID column found in test data.")
    else:
        # Check for other possible Audio ID column names
        possible_id_columns = ["AudioID", "Audio ID", "id", "ID", "Id"]
        for col in possible_id_columns:
            if col in test_df.columns:
                logger.info(f"Found ID column: {col}")
                test_df.rename(columns={col: "Audio_ID"}, inplace=True)
                has_audio_id = True
                break

        if not has_audio_id:
            logger.warning("No Audio ID column found in test data.")

    sentiment_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    if has_sentiment_labels:
        test_df["Score"] = test_df["Sentiment"].map(sentiment_map)
        logger.info("Test data sentiment distribution:")
        logger.info(test_df["Sentiment"].value_counts())

    tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        logger.info("Using default roberta-large tokenizer")

    max_length = 128
    test_encodings = tokenizer(
        test_df["Utterance"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    test_dataset = TextSentimentDataset(
        {k: v.numpy() for k, v in test_encodings.items()},
        test_df["Score"].tolist() if has_sentiment_labels else [0] * len(test_df)
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if model_path.endswith('.pt'):
        model = EnhancedSentimentModel(model_name="roberta-large")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            logger.info(f"Loaded model from checkpoint: {model_path}")
        else:
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model state dict: {model_path}")
    else:
        model = EnhancedSentimentModel(model_name="roberta-large")
        checkpoint_path = load_best_model(model_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

    model.to(device)
    model.eval()

    all_probabilities = []
    all_true_classes = []

    class_names = ["negative", "neutral", "positive"]

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = outputs["probabilities"]

            all_probabilities.extend(probabilities.cpu().numpy())
            if has_sentiment_labels:
                true_classes = (labels + 1).long().cpu().numpy()  # Convert from [-1,0,1] to [0,1,2]
                all_true_classes.extend(true_classes)

    # Convert probabilities to class predictions
    pred_classes = np.argmax(all_probabilities, axis=1)

    # Convert to percentages for better readability (e.g., [0.1, 0.2, 0.7] -> [10, 20, 70])
    probability_percentages = np.round(np.array(all_probabilities) * 100, 1)

    # Make sure we preserve any existing columns, especially the Audio ID if present
    # Create a new dataframe to hold all the results
    results_df = test_df.copy()

    # Add all probabilities as separate columns
    results_df["Negative_Prob"] = probability_percentages[:, 0]
    results_df["Neutral_Prob"] = probability_percentages[:, 1]
    results_df["Positive_Prob"] = probability_percentages[:, 2]

    # Add the predicted class
    results_df["Predicted_Sentiment"] = [class_names[idx] for idx in pred_classes]

    output_file = os.path.join(output_dir, "classification_test_predictions.csv")
    results_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

    if has_sentiment_labels:
        true_sentiments = [class_names[idx] for idx in all_true_classes]
        pred_sentiments = [class_names[idx] for idx in pred_classes]

        accuracy = accuracy_score(true_sentiments, pred_sentiments)
        f1 = f1_score(true_sentiments, pred_sentiments, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(true_sentiments, pred_sentiments))

        # Create and plot confusion matrix
        cm = confusion_matrix(true_sentiments, pred_sentiments, labels=class_names)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        logger.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

        # Plot ROC curves
        plot_roc_curves(all_true_classes, all_probabilities, class_names, os.path.join(output_dir, "roc_curves.png"))
        logger.info(f"ROC curves saved to {output_dir}")

        # Plot probability distributions
        plot_probability_distribution(all_probabilities, all_true_classes, class_names,
                                      os.path.join(output_dir, "probability_distributions.png"))
        logger.info(f"Probability distributions saved to {output_dir}")

        return accuracy, f1
    else:
        logger.info("No sentiment labels in test data. Evaluation metrics cannot be calculated.")
        return None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the classification sentiment model")
    parser.add_argument("--model_path", type=str, default="./classification_checkpoints",
                        help="Path to model checkpoint or directory with checkpoints")
    parser.add_argument("--test_file", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")

    args = parser.parse_args()

    test_model(args.model_path, args.test_file, args.output_dir)