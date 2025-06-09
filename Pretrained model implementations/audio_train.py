import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
from transformers import Wav2Vec2Processor

# Import custom modules
from audio_preprocess import (AudioSentimentDataset, AudioTransform, calculate_class_weights, set_seed,
                        AudioSentimentDatasetWav2Vec2, DataCollatorWithPadding)
from audio_model import (ImprovedAudioModel, Wav2Vec2ForSentimentClassification)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 16  # Reduced batch size for wav2vec2
EPOCHS = 50  # Epochs for wav2vec2
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
WEIGHT_DECAY = 1e-4
PATIENCE = 5
EARLY_STOPPING_PATIENCE = 10
SEED = 42
MODEL_DIR = "./models"
RESULTS_DIR = "./results"

# Model configuration
USE_WAV2VEC2 = True  # Set to false for original CNN models
WAV2VEC2_MODEL = "facebook/wav2vec2-base"  # Can also use "facebook/wav2vec2-large"
POOLING_MODE = "mean"  # Options: "mean", "sum", "max"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
set_seed(SEED)


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


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, save_dir):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot F1 scores
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()


# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def train_one_epoch_wav2vec2(model, train_loader, criterion, optimizer):
    """Train one epoch for wav2vec2 model"""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        try:
            input_values = batch['input_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_values.size(0)
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            continue

    # Check if we processed any batches
    if len(train_loader.dataset) > 0:
        avg_train_loss = train_loss / len(train_loader.dataset)
    else:
        avg_train_loss = 0

    # Check if we have any predictions
    if len(train_preds) > 0 and len(train_labels) > 0:
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
    else:
        train_accuracy = 0
        train_f1 = 0

    return avg_train_loss, train_accuracy, train_f1, train_preds, train_labels


def train_one_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch (original CNN models)"""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for inputs, labels in tqdm(train_loader, desc="Training"):
        try:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            continue

    # Check if we processed any batches
    if len(train_loader.dataset) > 0:
        avg_train_loss = train_loss / len(train_loader.dataset)
    else:
        avg_train_loss = 0

    # Check if we have any predictions
    if len(train_preds) > 0 and len(train_labels) > 0:
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
    else:
        train_accuracy = 0
        train_f1 = 0

    return avg_train_loss, train_accuracy, train_f1, train_preds, train_labels


def validate_wav2vec2(model, val_loader, criterion, class_names=None):
    """Validate the wav2vec2 model"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                input_values = batch['input_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                probs = torch.softmax(logits, dim=1)

                val_loss += loss.item() * input_values.size(0)
                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue

    # Check if we processed any batches
    if len(val_loader.dataset) > 0:
        avg_val_loss = val_loss / len(val_loader.dataset)
    else:
        avg_val_loss = 0

    # Check if we have any predictions
    if len(val_preds) > 0 and len(val_labels) > 0:
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Confusion matrix
        if class_names:
            cm = confusion_matrix(val_labels, val_preds)
            logger.info("\nConfusion Matrix:")
            cm_str = "\n"
            for i, row in enumerate(cm):
                cm_str += f"{class_names[i]}: {row}\n"
            logger.info(cm_str)
    else:
        val_accuracy = 0
        val_f1 = 0

    return avg_val_loss, val_accuracy, val_f1, val_preds, val_labels, val_probs


def validate(model, val_loader, criterion, class_names=None):
    """Validate the model (original CNN models)"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            try:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue

    # Check if we processed any batches
    if len(val_loader.dataset) > 0:
        avg_val_loss = val_loss / len(val_loader.dataset)
    else:
        avg_val_loss = 0

    # Check if we have any predictions
    if len(val_preds) > 0 and len(val_labels) > 0:
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Confusion matrix
        if class_names:
            cm = confusion_matrix(val_labels, val_preds)
            logger.info("\nConfusion Matrix:")
            cm_str = "\n"
            for i, row in enumerate(cm):
                cm_str += f"{class_names[i]}: {row}\n"
            logger.info(cm_str)
    else:
        val_accuracy = 0
        val_f1 = 0

    return avg_val_loss, val_accuracy, val_f1, val_preds, val_labels, val_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs,
                patience, early_stopping_patience, model_dir=MODEL_DIR, use_wav2vec2=False):

    best_val_f1 = 0
    early_stop_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "wav2vec2" if use_wav2vec2 else "cnn"
    run_dir = os.path.join(model_dir, f"run_{model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    train_f1s, val_f1s = [], []

    for epoch in range(epochs):
        if use_wav2vec2:
            train_loss, train_acc, train_f1, _, _ = train_one_epoch_wav2vec2(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1, val_preds, val_labels, _ = validate_wav2vec2(model, val_loader, criterion)
        else:
            train_loss, train_acc, train_f1, _, _ = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1, val_preds, val_labels, _ = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, F1: {train_f1:.4f} | Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            if use_wav2vec2:
                # Save the model properly
                model.save_pretrained(os.path.join(run_dir, "best_model"))
                # Also save the processor
                if hasattr(model, 'processor'):
                    model.processor.save_pretrained(os.path.join(run_dir, "best_model"))
            else:
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            plot_confusion_matrix(confusion_matrix(val_labels, val_preds), ['negative', 'neutral', 'positive'],
                                  os.path.join(run_dir, f"confusion_matrix_epoch_{epoch+1}.png"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, run_dir)

    logger.info(f"Best Validation F1: {best_val_f1:.4f}")
    return best_val_f1, run_dir


# In train.py, update the wav2vec2 section in main():

def main():
    set_seed(SEED)

    if USE_WAV2VEC2:
        logger.info(f"Using Wav2Vec2 model: {WAV2VEC2_MODEL}")

        # Initialize processor
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)

        # Create datasets
        train_dataset = AudioSentimentDatasetWav2Vec2("data/train.csv", "data/train", processor=processor, augment=True)
        val_dataset = AudioSentimentDatasetWav2Vec2("data/validation.csv", "data/validation", processor=processor)

        # Create data collator
        data_collator = DataCollatorWithPadding(processor=processor, padding=True)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=data_collator, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=data_collator, num_workers=2)

        # Initialize model properly
        model = Wav2Vec2ForSentimentClassification.from_pretrained_with_sentiment(
            WAV2VEC2_MODEL,
            num_labels=3,
            pooling_mode=POOLING_MODE
        ).to(DEVICE)

        # Freeze feature extractor for more stable training
        model.freeze_feature_extractor()

        # Initialize criterion (wav2vec2 model handles loss internally, but we can use this for validation)
        class_weights = calculate_class_weights("data/train.csv").to(DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)

        # Initialize optimizer with different learning rates for different parts
        optimizer = optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': LEARNING_RATE / 10},  # Lower LR for pretrained layers
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},  # Higher LR for classifier
        ], weight_decay=WEIGHT_DECAY)

        # Initialize scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=PATIENCE // 2)

        # Train model
        best_val_f1, model_path = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            EPOCHS, PATIENCE, EARLY_STOPPING_PATIENCE, use_wav2vec2=True
        )

    else:
        logger.info("Using original CNN models")

        # Create datasets with original preprocessing
        train_dataset = AudioSentimentDataset("data/train.csv", "data/train",
                                              transform=AudioTransform(augment=True), augment=True)
        val_dataset = AudioSentimentDataset("data/validation.csv", "data/validation")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # Initialize model
        class_weights = calculate_class_weights("data/train.csv").to(DEVICE)
        model = ImprovedAudioModel(num_classes=3).to(DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2)

        # Train model
        best_val_f1, model_path = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            EPOCHS, PATIENCE, EARLY_STOPPING_PATIENCE, use_wav2vec2=False
        )

    logger.info(f"Training complete. Best model saved to {model_path}")


if __name__ == "__main__":
    main()