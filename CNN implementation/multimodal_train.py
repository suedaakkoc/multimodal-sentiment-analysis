import os
import argparse
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
import json
from transformers import BertTokenizer, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

# Import custom modules
from preprocess import AudioTransform, calculate_class_weights, set_seed
from multimodal_model import MultimodalSentimentModel, FocalLoss
from multimodal_dataset import MultimodalSentimentDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multimodal sentiment analysis model")

    # General training parameters
    parser.add_argument("--modality", type=str, default="both", choices=["audio", "text", "both"],
                        help="Which modality to use: 'audio', 'text', or 'both'")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for learning rate reduction")
    parser.add_argument("--early_stopping", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision")

    # Data paths
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="data/validation.csv",
                        help="Path to validation CSV file")
    parser.add_argument("--audio_train_dir", type=str, default="data/train",
                        help="Directory containing training audio files")
    parser.add_argument("--audio_val_dir", type=str, default="data/validation",
                        help="Directory containing validation audio files")

    # Model specific parameters
    parser.add_argument("--audio_dropout", type=float, default=0.5,
                        help="Dropout rate for audio model")
    parser.add_argument("--text_dropout", type=float, default=0.3,
                        help="Dropout rate for text model")
    parser.add_argument("--fusion_dropout", type=float, default=0.6,
                        help="Dropout rate for fusion model")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for text")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of Cross Entropy Loss")
    parser.add_argument("--audio_modality_dropout", type=float, default=0.3,
                        help="Probability to drop audio modality during training")
    parser.add_argument("--text_modality_dropout", type=float, default=0.3,
                        help="Probability to drop text modality during training")

    # Output directories
    parser.add_argument("--model_dir", type=str, default="./multimodal_models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="./multimodal_results",
                        help="Directory to save results")

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


def train_one_epoch(model, train_loader, criterion, optimizer, modality, use_amp=False):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    scaler = GradScaler() if use_amp else None

    for batch in tqdm(train_loader, desc="Training"):
        try:
            # Move everything to device
            labels = batch['label'].to(DEVICE)

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

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(**inputs, labels=labels)
                    loss = outputs['loss']

                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs, labels=labels)
                loss = outputs['loss']

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Get predictions
            _, preds = torch.max(outputs['logits'], 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            train_loss += loss.item() * labels.size(0)
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            continue

    # Calculate metrics
    avg_train_loss = train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0

    if len(train_preds) > 0 and len(train_labels) > 0:
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
    else:
        train_accuracy = 0
        train_f1 = 0

    return avg_train_loss, train_accuracy, train_f1, train_preds, train_labels


def validate(model, val_loader, criterion, modality, class_names=None):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                # Move everything to device
                labels = batch['label'].to(DEVICE)

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

                outputs = model(**inputs, labels=labels)
                loss = outputs['loss']
                probabilities = outputs['probabilities']

                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs['logits'], 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probabilities.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue

    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0

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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, modality,
                epochs, patience, early_stopping_patience, model_dir, use_amp=False):
    """Train the model with early stopping and checkpointing"""
    best_val_f1 = 0
    early_stop_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_dir, f"{modality}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save tokenizer if text modality is used
    if modality in ['text', 'both']:
        tokenizer_dir = os.path.join(run_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Saved tokenizer to {tokenizer_dir}")

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(epochs):
        train_loss, train_acc, train_f1, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, modality, use_amp)

        val_loss, val_acc, val_f1, val_preds, val_labels, _ = validate(
            model, val_loader, criterion, modality, ['negative', 'neutral', 'positive'])

        # Track learning rate before scheduler step
        old_lr = [group['lr'] for group in optimizer.param_groups][0]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Check if learning rate changed
        new_lr = [group['lr'] for group in optimizer.param_groups][0]
        if new_lr != old_lr:
            logger.info(f"Learning rate changed from {old_lr} to {new_lr}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        logger.info(f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0

            # Save best model
            checkpoint_path = os.path.join(run_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'modality': modality,
                'config': {
                    'audio_feature_dim': model.audio_feature_dim if hasattr(model, 'audio_feature_dim') else 0,
                    'text_feature_dim': model.text_feature_dim if hasattr(model, 'text_feature_dim') else 0,
                    'fusion_dim': model.fusion_dim if hasattr(model, 'fusion_dim') else 0,
                }
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

            # Save confusion matrix
            cm_path = os.path.join(run_dir, f"confusion_matrix_epoch_{epoch + 1}.png")
            plot_confusion_matrix(confusion_matrix(val_labels, val_preds),
                                  ['negative', 'neutral', 'positive'], cm_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Plot metrics after each epoch
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                     train_f1s, val_f1s, run_dir)

    logger.info(f"Best Validation F1: {best_val_f1:.4f}")

    # Save model architecture details
    model_info = {
        'modality': modality,
        'audio_feature_dim': model.audio_feature_dim if hasattr(model, 'audio_feature_dim') else 0,
        'text_feature_dim': model.text_feature_dim if hasattr(model, 'text_feature_dim') else 0,
        'fusion_dim': model.fusion_dim if hasattr(model, 'fusion_dim') else 0,
        'best_val_f1': float(best_val_f1),
    }

    with open(os.path.join(run_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

    return best_val_f1, os.path.join(run_dir, "best_model.pt")


def main():
    """Main function for training"""
    args = parse_args()
    set_seed(SEED)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    logger.info(f"Starting training with modality: {args.modality}")
    logger.info(f"Using device: {DEVICE}")

    # Create tokenizer for text modality
    tokenizer = None
    if args.modality in ['text', 'both']:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Loaded BERT tokenizer")

    # Create datasets and data loaders
    logger.info("Creating datasets and data loaders...")

    train_transform = AudioTransform(augment=True) if args.modality in ['audio', 'both'] else None
    val_transform = AudioTransform(augment=False) if args.modality in ['audio', 'both'] else None

    train_dataset = MultimodalSentimentDataset(
        csv_path=args.train_csv,
        audio_dir=args.audio_train_dir if args.modality in ['audio', 'both'] else None,
        modality=args.modality,
        transform=train_transform,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=True
    )

    val_dataset = MultimodalSentimentDataset(
        csv_path=args.val_csv,
        audio_dir=args.audio_val_dir if args.modality in ['audio', 'both'] else None,
        modality=args.modality,
        transform=val_transform,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Create model
    logger.info(f"Creating {args.modality} model...")
    model = MultimodalSentimentModel(
        modality=args.modality,
        num_classes=3,
        audio_dropout=args.audio_dropout,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        vocab_size=tokenizer.vocab_size if tokenizer else 30522,
        embedding_dim=256,
        filter_sizes=[3, 4, 5],
        num_filters=128,
        max_length=args.max_length
    ).to(DEVICE)

    # Calculate class weights if needed
    class_weights = None
    if args.use_focal_loss:
        from preprocess import calculate_class_weights
        class_weights = calculate_class_weights(args.train_csv).to(DEVICE)
        logger.info(f"Using class weights for Focal Loss: {class_weights}")
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    if args.modality == 'text':
        # Use cosine scheduler with warmup for text modality (common for transformer-based models)
        num_training_steps = len(train_loader) * args.epochs
        warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        # Use ReduceLROnPlateau for audio and multimodal
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience // 2
        )
        # Log whenever learning rate changes
        old_lr = [group['lr'] for group in optimizer.param_groups][0]
        logger.info(f"Initial learning rate: {old_lr}")

    # Train model
    best_val_f1, model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        modality=args.modality,
        epochs=args.epochs,
        patience=args.patience,
        early_stopping_patience=args.early_stopping,
        model_dir=args.model_dir,
        use_amp=args.use_amp
    )

    logger.info(f"Training complete. Best model saved to {model_path}")
    logger.info(f"Best validation F1 score: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()