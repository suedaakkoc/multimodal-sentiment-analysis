import os
import argparse
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
from transformers import BertTokenizer

# Import from our modules
from multimodal_dataset import MultimodalSentimentDataset
from multimodal_model import MultimodalSentimentModel
from preprocess import AudioTransform, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['negative', 'neutral', 'positive']


def parse_args():
    parser = argparse.ArgumentParser(description="Test multimodal sentiment analysis model")

    parser.add_argument("--modality", type=str, default="both", choices=["audio", "text", "both"],
                        help="Which modality to use: 'audio', 'text', or 'both'")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--test_csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--audio_test_dir", type=str, default="data/test",
                        help="Directory containing test audio files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for text")
    parser.add_argument("--output_dir", type=str, default="./multimodal_test_results",
                        help="Directory to save test results")

    return parser.parse_args()


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


def load_model(model_path, modality=None, device=DEVICE):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determine modality from checkpoint if not specified
    if modality is None:
        if 'modality' in checkpoint:
            modality = checkpoint['modality']
        else:
            # Try to infer modality from directory name
            dir_name = os.path.basename(os.path.dirname(model_path))
            if 'audio' in dir_name and 'text' not in dir_name:
                modality = 'audio'
            elif 'text' in dir_name and 'audio' not in dir_name:
                modality = 'text'
            else:
                modality = 'both'

        logger.info(f"Inferred modality: {modality}")

    # Extract model config from checkpoint
    model_config = {}
    if 'config' in checkpoint:
        model_config = checkpoint['config']

    # Create model
    model = MultimodalSentimentModel(
        modality=modality,
        num_classes=3,
    ).to(device)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Model loaded successfully with modality: {modality}")
    return model, modality


def test_model(args):
    """Test the model on the test set"""
    set_seed(SEED)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, modality = load_model(args.model_path, args.modality)
    model.eval()

    logger.info(f"Testing with modality: {modality}")

    # Create tokenizer for text modality
    tokenizer = None
    if modality in ['text', 'both']:
        # Try to load tokenizer from model directory
        model_dir = os.path.dirname(args.model_path)
        tokenizer_dir = os.path.join(model_dir, "tokenizer")

        if os.path.exists(tokenizer_dir):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
            logger.info(f"Loaded tokenizer from {tokenizer_dir}")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Using default BERT tokenizer")

    # Create dataset and data loader
    test_transform = AudioTransform(augment=False) if modality in ['audio', 'both'] else None

    test_dataset = MultimodalSentimentDataset(
        csv_path=args.test_csv,
        audio_dir=args.audio_test_dir if modality in ['audio', 'both'] else None,
        modality=modality,
        transform=test_transform,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: Keep shuffle=False for proper indexing
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []
    audio_ids = []

    # Keep track of the current index in the dataset
    current_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                # Move everything to device
                labels = batch['label'].to(DEVICE)
                batch_size = labels.size(0)

                # Prepare inputs based on modality
                if modality == 'audio':
                    audio = batch['audio'].to(DEVICE)
                    inputs = {'audio': audio}
                elif modality == 'text':
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                else:  # 'both'
                    audio = batch['audio'].to(DEVICE)
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    inputs = {
                        'audio': audio,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }

                # Forward pass
                outputs = model(**inputs)
                probabilities = outputs['probabilities']

                # Get predictions
                _, preds = torch.max(outputs['logits'], 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

                # Get corresponding Audio IDs for this batch
                batch_audio_ids = []
                for i in range(batch_size):
                    dataset_idx = current_idx + i
                    try:
                        if dataset_idx < len(test_dataset.df):
                            audio_id = test_dataset.df.iloc[dataset_idx]["Audio_ID"]
                            batch_audio_ids.append(str(audio_id))
                        else:
                            batch_audio_ids.append(f"sample_{dataset_idx}")
                    except (KeyError, IndexError):
                        batch_audio_ids.append(f"sample_{dataset_idx}")

                audio_ids.extend(batch_audio_ids)

                # Update current index
                current_idx += batch_size

            except Exception as e:
                logger.error(f"Error in testing batch: {e}")
                # Still need to update the index even if batch fails
                current_idx += labels.size(0) if 'labels' in locals() else args.batch_size
                continue

    # Calculate metrics
    if len(all_preds) > 0 and len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        logger.info(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(args.output_dir, f"{modality}_confusion_matrix.png"))

        # Save predictions to CSV
        results_df = pd.DataFrame({
            "Audio_ID": audio_ids,
            "True_Sentiment": [CLASS_NAMES[i] for i in all_labels],
            "Predicted_Sentiment": [CLASS_NAMES[i] for i in all_preds],
            "Negative_Prob": [p[0] for p in all_probs],
            "Neutral_Prob": [p[1] for p in all_probs],
            "Positive_Prob": [p[2] for p in all_probs]
        })

        # Save predictions
        results_path = os.path.join(args.output_dir, f"{modality}_test_predictions.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved predictions to {results_path}")

        # Additional visualizations
        plot_probability_distributions(all_probs, all_labels,
                                       os.path.join(args.output_dir, f"{modality}_probability_distributions.png"))
        plot_roc_curves(all_probs, all_labels,
                        os.path.join(args.output_dir, f"{modality}_roc_curves.png"))

        return accuracy, f1
    else:
        logger.warning("No predictions were made. Check the test dataset and model configuration.")
        return 0.0, 0.0


def main():
    """Main function for testing"""
    args = parse_args()
    logger.info(f"Testing with modality: {args.modality}")
    logger.info(f"Using device: {DEVICE}")

    accuracy, f1 = test_model(args)

    logger.info(f"Testing complete. Final metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()