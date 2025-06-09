import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_curve, \
    roc_curve, auc
from transformers import Wav2Vec2Processor

# Import from our modules
from audio_preprocess import (AudioSentimentDataset, AudioTransform, set_seed,
                        AudioSentimentDatasetWav2Vec2, DataCollatorWithPadding)
from audio_model import ImprovedAudioModel, Wav2Vec2ForSentimentClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_RESULTS_DIR = "./test_results"

# Class names for reporting
CLASS_NAMES = ['negative', 'neutral', 'positive']


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_probability_distributions(probs, labels, save_path):
    """Plot probability distributions for each class"""
    plt.figure(figsize=(15, 5))

    for i, class_name in enumerate(CLASS_NAMES):
        plt.subplot(1, 3, i + 1)

        for j, target_class in enumerate(CLASS_NAMES):
            # Get probabilities for class i where true label is j
            mask = np.array(labels) == j
            if np.any(mask):
                class_probs = np.array(probs)[mask, i]
                plt.hist(class_probs, alpha=0.5, bins=20, label=f"True {target_class}")

        plt.title(f"{class_name} Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(probs, labels, save_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        # Convert to one-vs-all problem
        binary_labels = np.array([1 if l == i else 0 for l in labels])
        class_probs = np.array(probs)[:, i]

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


def test_wav2vec2_model(model_path, test_csv="data/test.csv", test_audio_dir="data/test"):
    """Test wav2vec2 model"""
    set_seed(SEED)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Load processor and model
    # Try to detect the wav2vec2 model from the path
    if "facebook/wav2vec2" in model_path or "wav2vec2" in model_path:
        wav2vec2_model_name = "facebook/wav2vec2-base"  # default
    else:
        wav2vec2_model_name = "facebook/wav2vec2-base"

    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)

    # Load model
    model = Wav2Vec2ForSentimentClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    # Dataset
    test_dataset = AudioSentimentDatasetWav2Vec2(test_csv, test_audio_dir, processor=processor)
    data_collator = DataCollatorWithPadding(processor=processor, padding=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=data_collator)

    all_probs, all_preds, all_labels, audio_ids = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            input_values = batch['input_values'].to(DEVICE)
            labels = batch['labels']

            outputs = model(input_values=input_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Get corresponding Audio IDs
            batch_size = len(input_values)
            batch_ids = test_dataset.df.iloc[i * BATCH_SIZE: i * BATCH_SIZE + batch_size]["Audio ID"].tolist()
            audio_ids.extend(batch_ids)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Plot and save results
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(TEST_RESULTS_DIR, "confusion_matrix_wav2vec2.png"))
    plot_probability_distributions(all_probs, all_labels,
                                   os.path.join(TEST_RESULTS_DIR, "prob_distributions_wav2vec2.png"))
    plot_roc_curves(all_probs, all_labels, os.path.join(TEST_RESULTS_DIR, "roc_curves_wav2vec2.png"))

    # Save predictions
    df = pd.DataFrame({
        "Audio ID": audio_ids,
        "True Sentiment": [CLASS_NAMES[i] for i in all_labels],
        "Predicted Sentiment": [CLASS_NAMES[i] for i in all_preds],
        "negative_Prob": [p[0] for p in all_probs],
        "neutral_Prob": [p[1] for p in all_probs],
        "positive_Prob": [p[2] for p in all_probs]
    })

    df.to_csv(os.path.join(TEST_RESULTS_DIR, "test_predictions_wav2vec2.csv"), index=False)
    logger.info(f"Saved predictions to {os.path.join(TEST_RESULTS_DIR, 'test_predictions_wav2vec2.csv')}")


def test_cnn_model(model_path, test_csv="data/test.csv", test_audio_dir="data/test"):
    """Test original CNN model"""
    set_seed(SEED)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Load model
    model = ImprovedAudioModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Dataset
    test_dataset = AudioSentimentDataset(test_csv, test_audio_dir)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs, all_preds, all_labels, audio_ids = [], [], [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Get corresponding Audio IDs
            batch_ids = test_dataset.df.iloc[i * BATCH_SIZE: i * BATCH_SIZE + len(inputs)]["Audio ID"].tolist()
            audio_ids.extend(batch_ids)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Plot and save results
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(TEST_RESULTS_DIR, "confusion_matrix_cnn.png"))
    plot_probability_distributions(all_probs, all_labels, os.path.join(TEST_RESULTS_DIR, "prob_distributions_cnn.png"))
    plot_roc_curves(all_probs, all_labels, os.path.join(TEST_RESULTS_DIR, "roc_curves_cnn.png"))

    # Save predictions
    df = pd.DataFrame({
        "Audio ID": audio_ids,
        "True Sentiment": [CLASS_NAMES[i] for i in all_labels],
        "Predicted Sentiment": [CLASS_NAMES[i] for i in all_preds],
        "negative_Prob": [p[0] for p in all_probs],
        "neutral_Prob": [p[1] for p in all_probs],
        "positive_Prob": [p[2] for p in all_probs]
    })

    df.to_csv(os.path.join(TEST_RESULTS_DIR, "test_predictions_cnn.csv"), index=False)
    logger.info(f"Saved predictions to {os.path.join(TEST_RESULTS_DIR, 'test_predictions_cnn.csv')}")


def find_best_models():
    """Find all available trained models"""
    models = {'wav2vec2': [], 'cnn': []}

    if os.path.exists("./models"):
        for run in os.listdir("./models"):
            run_path = os.path.join("./models", run)

            # Check for wav2vec2 model
            wav2vec2_path = os.path.join(run_path, "best_model")
            if os.path.exists(wav2vec2_path) and "wav2vec2" in run:
                models['wav2vec2'].append(wav2vec2_path)

            # Check for CNN model
            cnn_path = os.path.join(run_path, "best_model.pt")
            if os.path.exists(cnn_path) and "cnn" in run:
                models['cnn'].append(cnn_path)
            elif os.path.exists(cnn_path) and "wav2vec2" not in run:
                models['cnn'].append(cnn_path)

    return models


def main():
    logger.info("Starting model evaluation...")

    # Find all available models
    models = find_best_models()

    # Test wav2vec2 models
    if models['wav2vec2']:
        logger.info(f"Found {len(models['wav2vec2'])} wav2vec2 model(s)")
        for i, model_path in enumerate(models['wav2vec2']):
            logger.info(f"Testing wav2vec2 model {i + 1}: {model_path}")
            try:
                test_wav2vec2_model(model_path)
            except Exception as e:
                logger.error(f"Error testing wav2vec2 model {model_path}: {e}")
    else:
        logger.info("No wav2vec2 models found")

    # Test CNN models
    if models['cnn']:
        logger.info(f"Found {len(models['cnn'])} CNN model(s)")
        for i, model_path in enumerate(models['cnn']):
            logger.info(f"Testing CNN model {i + 1}: {model_path}")
            try:
                test_cnn_model(model_path)
            except Exception as e:
                logger.error(f"Error testing CNN model {model_path}: {e}")
    else:
        logger.info("No CNN models found")

    if not models['wav2vec2'] and not models['cnn']:
        logger.error("No trained models found. Please train a model first.")
    else:
        logger.info("Model evaluation complete. Check the test_results directory for detailed results.")


if __name__ == "__main__":
    main()