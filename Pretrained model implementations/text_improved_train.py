import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer, RobertaModel,
    get_cosine_schedule_with_warmup
)
import os
import random
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from text_dataset import TextSentimentDataset
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("classification_training.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Random Seed for Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


set_seed()


# Classification model
class EnhancedSentimentModel(nn.Module):
    def __init__(self, model_name="roberta-large", dropout_rate=0.3, use_attention_pooling=True):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        if use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )

        # Changed the final layer to output 3 values (one for each class)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)  # Changed from 1 to 3 for three sentiment classes
        )

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Changed to Cross-Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_attention_pooling:
            hidden_states = outputs.last_hidden_state
            attention_weights = self.attention(hidden_states)
            pooled_output = torch.sum(attention_weights * hidden_states, dim=1)
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)
        probabilities = F.softmax(logits, dim=1)

        loss = None
        if labels is not None:
            # Convert regression-style labels to class indices
            # -1.0 (negative) -> 0, 0.0 (neutral) -> 1, 1.0 (positive) -> 2
            class_labels = (labels + 1).long()  # Convert from [-1,0,1] to [0,1,2]
            loss = self.loss_fn(logits, class_labels)

        return {"loss": loss, "logits": logits, "probabilities": probabilities}


# Training function
def train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        num_epochs=5,
        grad_accum_steps=2,
        early_stopping_patience=3,
        checkpoint_dir="./classification_checkpoints",
        use_amp=True
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    global_step = 0
    scaler = GradScaler() if use_amp else None

    class_names = ["negative", "neutral", "positive"]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"]
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                loss = loss / grad_accum_steps

                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            train_loss += loss.item() * grad_accum_steps
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        all_probabilities = []
        all_true_classes = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                probabilities = outputs["probabilities"]

                val_loss += loss.item()
                all_probabilities.extend(probabilities.cpu().numpy())

                # Convert regression labels to class indices for evaluation
                true_classes = (labels + 1).long().cpu().numpy()  # Convert from [-1,0,1] to [0,1,2]
                all_true_classes.extend(true_classes)

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Convert probabilities to class predictions
        pred_classes = np.argmax(all_probabilities, axis=1)

        # Convert class indices back to string labels for reporting
        pred_sentiments = [class_names[idx] for idx in pred_classes]
        true_sentiments = [class_names[idx] for idx in all_true_classes]

        accuracy = accuracy_score(all_true_classes, pred_classes)
        f1 = f1_score(all_true_classes, pred_classes, average='weighted')

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(true_sentiments, pred_sentiments))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_f1 = f1
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f"model_epoch_{epoch + 1}_val_loss_{avg_val_loss:.4f}_f1_{f1:.4f}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': avg_val_loss,
                'f1': f1,
            }, checkpoint_path)
            logger.info(f"Best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))

    return best_val_loss, best_f1


def main():
    train_df = pd.read_csv("data/train.csv", sep=";")
    val_df = pd.read_csv("data/validation.csv", sep=";")

    # Still using the same mapping, but the model will convert to class indices
    sentiment_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    train_df["Score"] = train_df["Sentiment"].map(sentiment_map)
    val_df["Score"] = val_df["Sentiment"].map(sentiment_map)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    max_length = 128

    train_encodings = tokenizer(train_df["Utterance"].tolist(), truncation=True, padding=True, max_length=max_length,
                                return_tensors="pt")
    val_encodings = tokenizer(val_df["Utterance"].tolist(), truncation=True, padding=True, max_length=max_length,
                              return_tensors="pt")

    train_dataset = TextSentimentDataset({k: v.numpy() for k, v in train_encodings.items()}, train_df["Score"].tolist())
    val_dataset = TextSentimentDataset({k: v.numpy() for k, v in val_encodings.items()}, val_df["Score"].tolist())

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = EnhancedSentimentModel(model_name="roberta-large", dropout_rate=0.3, use_attention_pooling=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.01)

    num_training_steps = len(train_dataloader) * 5
    warmup_steps = int(0.06 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    best_val_loss, best_f1 = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=5,
        grad_accum_steps=2,
        early_stopping_patience=3,
        checkpoint_dir="./classification_checkpoints",
        use_amp=True
    )

    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}, Best F1: {best_f1:.4f}")

    tokenizer.save_pretrained("./classification_checkpoints/tokenizer")
    torch.save(model.state_dict(), "./classification_checkpoints/final_model.pt")
    logger.info("Final model saved.")


if __name__ == "__main__":
    main()