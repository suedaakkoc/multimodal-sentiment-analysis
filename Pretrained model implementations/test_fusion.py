import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from transformers import Wav2Vec2Processor, RobertaTokenizer
from torch.utils.data import DataLoader
import json

# Import classes from early_fusion.py (updated imports)
from early_fusion import (
    MultimodalSentimentDataset,
    HybridDataCollator,  # Updated from MultimodalDataCollator
    HybridMultimodalModel,  # Updated from EarlyFusionModel
    set_seed,
    clear_memory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_fusion_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 8  # Reduced for memory efficiency (matching training script)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['negative', 'neutral', 'positive']
SEED = 42

# Set seed for reproducibility
set_seed(SEED)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Hybrid Fusion Model - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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


def plot_attention_analysis(attention_weights, save_path):
    """Plot attention weights analysis"""
    if attention_weights is None or len(attention_weights) == 0:
        return

    # Convert to numpy if it's a tensor
    if isinstance(attention_weights[0], torch.Tensor):
        attention_weights = [w.cpu().numpy() for w in attention_weights]

    attention_array = np.array(attention_weights)

    plt.figure(figsize=(12, 5))

    # Plot 1: Distribution of attention weights
    plt.subplot(1, 2, 1)
    text_weights = attention_array[:, 0]
    audio_weights = attention_array[:, 1]

    plt.hist(text_weights, alpha=0.7, label='Text Attention', bins=30, color='blue')
    plt.hist(audio_weights, alpha=0.7, label='Audio Attention', bins=30, color='red')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Attention Weight Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Average attention weights
    plt.subplot(1, 2, 2)
    avg_text = np.mean(text_weights)
    avg_audio = np.mean(audio_weights)

    plt.bar(['Text', 'Audio'], [avg_text, avg_audio],
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Average Attention Weight')
    plt.title('Average Attention Weights')
    plt.ylim(0, 1)

    # Add value labels on bars
    plt.text(0, avg_text + 0.01, f'{avg_text:.3f}', ha='center', va='bottom')
    plt.text(1, avg_audio + 0.01, f'{avg_audio:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Average attention - Text: {avg_text:.3f}, Audio: {avg_audio:.3f}")


def find_best_model(model_dir):
    """Find the best model in the directory based on F1 score"""
    best_model = None
    best_f1 = -1

    for file in os.listdir(model_dir):
        if file.startswith("best_hybrid_model_") and file.endswith(".pt"):
            # Extract F1 score from filename
            try:
                f1_score = float(file.split("_f1_")[1].split(".pt")[0])
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = os.path.join(model_dir, file)
            except:
                continue

    if best_model is None:
        # Look for any .pt file in the directory
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if pt_files:
            best_model = os.path.join(model_dir, pt_files[0])

    return best_model


def test_hybrid_fusion_model(model_path, test_csv, test_audio_dir,
                             text_model_path=None, audio_model_path=None,
                             output_dir="./test_results"):
    """Test the hybrid fusion model"""
    logger.info("=" * 80)
    logger.info("HYBRID FUSION MODEL TESTING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {test_csv}")
    logger.info(f"Device: {DEVICE}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        logger.info("Model checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        return None, None

    # Get model arguments from checkpoint if available
    model_args = checkpoint.get('args', None)

    # Set default model names
    text_model_name = "roberta-large"
    audio_model_name = "facebook/wav2vec2-base"
    fusion_dim = 512

    if model_args:
        text_model_name = getattr(model_args, 'text_model_name', text_model_name)
        audio_model_name = getattr(model_args, 'audio_model_name', audio_model_name)
        fusion_dim = getattr(model_args, 'fusion_dim', fusion_dim)
        text_model_path = getattr(model_args, 'text_model_path', text_model_path)
        audio_model_path = getattr(model_args, 'audio_model_path', audio_model_path)

    # Initialize processors
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(text_model_name)

    logger.info(f"Text model: {text_model_name}")
    logger.info(f"Audio model: {audio_model_name}")

    # Initialize model
    model = HybridMultimodalModel(
        text_model_path=text_model_path,
        audio_model_path=audio_model_path,
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        fusion_dim=fusion_dim
    )

    # Load model weights
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Model loaded with validation F1: {checkpoint.get('val_f1', 'N/A')}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return None, None

    model = model.to(DEVICE)
    model.eval()

    # Create test dataset
    try:
        test_dataset = MultimodalSentimentDataset(
            csv_path=test_csv,
            audio_dir=test_audio_dir,
            audio_processor=audio_processor,
            text_tokenizer=text_tokenizer
        )
        logger.info(f"Test dataset created with {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating test dataset: {e}")
        return None, None

    # Create data collator
    data_collator = HybridDataCollator(audio_processor=audio_processor)

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=1,  # Reduced for stability
        pin_memory=False
    )

    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    all_attention_weights = []
    audio_ids = []
    test_loss = 0.0

    logger.info("Starting evaluation...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # Clear memory periodically
                if i % 10 == 0:
                    clear_memory()

                # Move to device
                labels = batch['labels'].to(DEVICE)
                text_input_ids = batch['text_input_ids'].to(DEVICE)
                text_attention_mask = batch['text_attention_mask'].to(DEVICE)
                audio_input_values = batch['audio_input_values'].to(DEVICE)
                audio_attention_mask = batch.get('audio_attention_mask', None)
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(DEVICE)

                # Forward pass
                outputs = model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    audio_input_values=audio_input_values,
                    audio_attention_mask=audio_attention_mask,
                    labels=labels
                )

                logits = outputs["logits"]
                probs = outputs["probabilities"]

                if outputs["loss"] is not None:
                    test_loss += outputs["loss"].item() * labels.size(0)

                # Get predictions
                _, preds = torch.max(logits, 1)

                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Collect attention weights if available
                if "attention_weights" in outputs and outputs["attention_weights"] is not None:
                    all_attention_weights.extend(outputs["attention_weights"].cpu().numpy())

                # Get corresponding Audio IDs
                batch_size = len(labels)
                batch_start = i * BATCH_SIZE
                batch_end = min(batch_start + batch_size, len(test_dataset))
                batch_ids = test_dataset.df.iloc[batch_start:batch_end]["Audio_ID"].tolist()
                audio_ids.extend(batch_ids)

                # Clear batch data
                del text_input_ids, text_attention_mask, audio_input_values, labels, outputs, logits, probs

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue

    # Calculate metrics
    if len(all_preds) == 0:
        logger.error("No predictions generated!")
        return None, None

    avg_test_loss = test_loss / len(test_dataset) if test_loss > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Log results
    logger.info("=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 (Weighted): {f1:.4f}")
    logger.info(f"Test F1 (Macro): {f1_macro:.4f}")
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(output_dir, "hybrid_fusion_confusion_matrix.png"))
    logger.info(f"Confusion matrix saved to {output_dir}")

    # Plot ROC curves
    plot_roc_curves(all_labels, all_probs, CLASS_NAMES, os.path.join(output_dir, "hybrid_fusion_roc_curves.png"))
    logger.info(f"ROC curves saved to {output_dir}")

    # Plot probability distributions
    plot_probability_distribution(all_probs, all_labels, CLASS_NAMES,
                                  os.path.join(output_dir, "probability_distributions.png"))
    logger.info(f"Probability distributions saved to {output_dir}")

    # Plot attention analysis if available
    if all_attention_weights:
        plot_attention_analysis(all_attention_weights, os.path.join(output_dir, "attention_analysis.png"))
        logger.info(f"Attention analysis saved to {output_dir}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "Audio_ID": audio_ids[:len(all_preds)],  # Ensure same length
        "True_Sentiment": [CLASS_NAMES[i] for i in all_labels],
        "Predicted_Sentiment": [CLASS_NAMES[i] for i in all_preds],
        "Negative_Prob": [p[0] for p in all_probs],
        "Neutral_Prob": [p[1] for p in all_probs],
        "Positive_Prob": [p[2] for p in all_probs]
    })

    predictions_csv_path = os.path.join(output_dir, "hybrid_fusion_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.info(f"Predictions saved to {predictions_csv_path}")

    # Save test metrics
    test_metrics = {
        'test_loss': avg_test_loss,
        'test_accuracy': accuracy,
        'test_f1_weighted': f1,
        'test_f1_macro': f1_macro,
        'num_samples': len(all_preds),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                                       output_dict=True)
    }

    if all_attention_weights:
        attention_array = np.array(all_attention_weights)
        test_metrics['attention_stats'] = {
            'avg_text_attention': float(np.mean(attention_array[:, 0])),
            'avg_audio_attention': float(np.mean(attention_array[:, 1])),
            'std_text_attention': float(np.std(attention_array[:, 0])),
            'std_audio_attention': float(np.std(attention_array[:, 1]))
        }

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")

    # Clean up memory
    clear_memory()

    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Test hybrid fusion model")

    # Data arguments
    parser.add_argument("--test_csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--test_audio_dir", type=str, default="data/test",
                        help="Directory containing test audio files")

    # Model arguments
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing model checkpoints")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to specific model checkpoint")
    parser.add_argument("--text_model_path", type=str, default=None,
                        help="Path to the original text model (if needed)")
    parser.add_argument("--audio_model_path", type=str, default=None,
                        help="Path to the original audio model (if needed)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Directory to save test results")

    args = parser.parse_args()

    # Find model to test
    model_path = args.model_path
    if model_path is None and args.model_dir is not None:
        model_path = find_best_model(args.model_dir)
        if model_path is None:
            logger.error(f"No model found in {args.model_dir}")
            return
    elif model_path is None:
        # Find most recent hybrid model
        model_dirs = [os.path.join("hybrid_models_MELD", d) for d in os.listdir("hybrid_models_MELD")
                      if d.startswith("hybrid_fusion_") and os.path.isdir(os.path.join("hybrid_models_MELD", d))]
        if not model_dirs:
            logger.error("No hybrid fusion models found in ./hybrid_models")
            return

        # Sort by creation time
        model_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        model_path = find_best_model(model_dirs[0])
        if model_path is None:
            logger.error(f"No model found in {model_dirs[0]}")
            return

    logger.info(f"Using model: {model_path}")

    # Test model
    accuracy, f1 = test_hybrid_fusion_model(
        model_path=model_path,
        test_csv=args.test_csv,
        test_audio_dir=args.test_audio_dir,
        text_model_path=args.text_model_path,
        audio_model_path=args.audio_model_path,
        output_dir=args.output_dir
    )

    if accuracy is not None and f1 is not None:
        logger.info("=" * 80)
        logger.info("TESTING COMPLETED SUCCESSFULLY!")
        logger.info(f"Final Test Accuracy: {accuracy:.4f}")
        logger.info(f"Final Test F1 Score: {f1:.4f}")
        logger.info("=" * 80)
    else:
        logger.error("Testing failed!")


if __name__ == "__main__":
    main()