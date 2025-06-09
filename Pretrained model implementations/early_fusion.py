import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import time
from datetime import datetime
import json
import pandas as pd
import random
from transformers import Wav2Vec2Processor, RobertaTokenizer, get_cosine_schedule_with_warmup
import torchaudio
import librosa
from safetensors.torch import load_file

# Import your existing models
from text_improved_train import EnhancedSentimentModel
from audio_model import Wav2Vec2ForSentimentClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_multimodal_fusion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants - optimized for your needs
BATCH_SIZE = 8  # Increased as requested
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.005
PATIENCE = 4
EARLY_STOPPING_PATIENCE = 8
GRAD_ACCUM_STEPS = 2  # Adjusted for batch_size=8
SEED = 42
MODEL_DIR = "./hybrid_models"
RESULTS_DIR = "./hybrid_results"
USE_AMP = False  # Disabled for memory efficiency

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Seed setting
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Memory management utilities
import gc


def clear_memory():
    """Clear GPU memory between operations"""
    gc.collect()
    torch.cuda.empty_cache()


class AttentionFusionLayer(nn.Module):
    """
    Attention-based fusion for your superior text model + audio model
    """

    def __init__(self, text_dim, audio_dim, fusion_dim=512, dropout_rate=0.3):
        super().__init__()

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim

        # Project both modalities to same dimension for fair comparison
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.audio_projection = nn.Linear(audio_dim, fusion_dim)

        # Layer normalization
        self.text_norm = nn.LayerNorm(fusion_dim)
        self.audio_norm = nn.LayerNorm(fusion_dim)

        # Attention mechanism - learns to weight text vs audio
        self.attention_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 1)
        )

        # Gate for final fusion
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, text_features, audio_features):
        """
        Args:
            text_features: [batch_size, text_dim] - from your trained text model
            audio_features: [batch_size, audio_dim] - from your trained audio model
        Returns:
            fused_features: [batch_size, fusion_dim]
            attention_weights: [batch_size, 2] - [text_weight, audio_weight]
        """
        # Project to common space
        text_proj = self.text_norm(self.text_projection(text_features))
        audio_proj = self.audio_norm(self.audio_projection(audio_features))

        # Stack for attention computation
        modalities = torch.stack([text_proj, audio_proj], dim=1)  # [batch, 2, fusion_dim]

        # Compute attention weights
        attention_scores = self.attention_layer(modalities)  # [batch, 2, 1]
        attention_weights = torch.softmax(attention_scores.squeeze(-1), dim=1)  # [batch, 2]

        # Apply attention weights
        weighted_text = text_proj * attention_weights[:, 0:1]
        weighted_audio = audio_proj * attention_weights[:, 1:2]

        # Gate mechanism for final fusion
        combined = torch.cat([weighted_text, weighted_audio], dim=1)
        gate = self.gate(combined)

        # Final fusion
        fused_features = weighted_text * gate + weighted_audio * (1 - gate)

        return fused_features, attention_weights


class HybridMultimodalModel(nn.Module):
    """
    Hybrid model that uses your existing trained unimodal models
    """

    def __init__(self,
                 text_model_path=None,
                 audio_model_path=None,
                 text_model_name="roberta-large",
                 audio_model_name="facebook/wav2vec2-base",
                 num_classes=3,
                 dropout_rate=0.3,
                 fusion_dim=512):
        super().__init__()

        # Load your existing trained text model
        self.text_model = EnhancedSentimentModel(model_name=text_model_name, dropout_rate=dropout_rate)

        # Remove final classifier to get features
        self.text_feature_dim = 128  # Based on your EnhancedSentimentModel architecture
        self.text_model.classifier = nn.Sequential(*list(self.text_model.classifier.children())[:-1])

        # Load your trained text weights
        if text_model_path:
            self._load_text_weights(text_model_path)
            logger.info(f"Loaded trained text model from {text_model_path}")

        # Load your existing trained audio model
        self.audio_model = Wav2Vec2ForSentimentClassification.from_pretrained_with_sentiment(
            audio_model_name, num_labels=3
        )

        # Remove classifier to get features
        self.audio_feature_dim = self.audio_model.config.hidden_size
        if hasattr(self.audio_model, 'classifier'):
            delattr(self.audio_model, 'classifier')

        # Load your trained audio weights
        if audio_model_path:
            self._load_audio_weights(audio_model_path)
            logger.info(f"Loaded trained audio model from {audio_model_path}")

        # Attention-based fusion (the key innovation)
        self.fusion_layer = AttentionFusionLayer(
            text_dim=self.text_feature_dim,
            audio_dim=self.audio_feature_dim,
            fusion_dim=fusion_dim,
            dropout_rate=dropout_rate
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize new layers
        self._init_fusion_weights()

    def _load_text_weights(self, text_model_path):
        """Load your trained text model weights"""
        try:
            if text_model_path.endswith('.safetensors'):
                state_dict = load_file(text_model_path)
            else:
                checkpoint = torch.load(text_model_path, map_location='cpu')
                state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Remove final classifier weights since we're using features
            classifier_keys = [k for k in state_dict.keys() if "classifier.8" in k]
            for key in classifier_keys:
                state_dict.pop(key, None)

            self.text_model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            logger.warning(f"Could not load text weights from {text_model_path}: {e}")

    def _load_audio_weights(self, audio_model_path):
        """Load your trained audio model weights"""
        try:
            if audio_model_path.endswith('.safetensors'):
                state_dict = load_file(audio_model_path)
            elif os.path.isdir(audio_model_path):
                self.audio_model = Wav2Vec2ForSentimentClassification.from_pretrained(audio_model_path)
                self.audio_feature_dim = self.audio_model.config.hidden_size
                if hasattr(self.audio_model, 'classifier'):
                    delattr(self.audio_model, 'classifier')
                return
            else:
                checkpoint = torch.load(audio_model_path, map_location='cpu')
                state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Map keys and remove classifier
            audio_state_dict = {}
            for key, value in state_dict.items():
                if "classifier" in key:
                    continue
                if key.startswith("wav2vec2."):
                    audio_state_dict[key] = value
                else:
                    audio_state_dict[f"wav2vec2.{key}"] = value

            self.audio_model.load_state_dict(audio_state_dict, strict=False)

        except Exception as e:
            logger.warning(f"Could not load audio weights from {audio_model_path}: {e}")

    def _init_fusion_weights(self):
        """Initialize fusion and classifier weights"""
        for module in [self.fusion_layer, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, text_input_ids=None, text_attention_mask=None,
                audio_input_values=None, audio_attention_mask=None, labels=None):
        """
        Forward pass using your existing trained models + attention fusion
        """
        # Get text features from your trained text model
        text_outputs = self.text_model.roberta(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )

        # Apply your text model's attention pooling
        if self.text_model.use_attention_pooling:
            text_hidden_states = text_outputs.last_hidden_state
            text_attention_weights = self.text_model.attention(text_hidden_states)
            text_pooled = torch.sum(text_attention_weights * text_hidden_states, dim=1)
        else:
            text_pooled = text_outputs.last_hidden_state[:, 0, :]

        # Pass through text model layers (except final classifier)
        text_features = text_pooled
        for layer in self.text_model.classifier:
            text_features = layer(text_features)

        # Get audio features from your trained audio model
        if isinstance(audio_input_values, dict) and "input_values" in audio_input_values:
            audio_outputs = self.audio_model.wav2vec2(
                input_values=audio_input_values["input_values"],
                attention_mask=audio_input_values.get("attention_mask", None),
                mask_time_indices=None  # Disable masking to prevent sequence length errors
            )
        else:
            audio_outputs = self.audio_model.wav2vec2(
                input_values=audio_input_values,
                attention_mask=audio_attention_mask,
                mask_time_indices=None  # Disable masking to prevent sequence length errors
            )

        # Apply audio model's pooling
        audio_hidden_states = audio_outputs[0]
        audio_features = self.audio_model.merged_strategy(
            audio_hidden_states, mode=self.audio_model.pooling_mode
        )

        # Attention-based fusion (the key innovation)
        fused_features, attention_weights = self.fusion_layer(text_features, audio_features)

        # Final classification
        logits = self.classifier(fused_features)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Calculate loss
        loss = None
        if labels is not None:
            # Handle label conversion
            if labels.dim() == 1 and not labels.dtype == torch.long:
                class_labels = (labels + 1).long()
            else:
                class_labels = labels
            loss = self.loss_fn(logits, class_labels)

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attention_weights,  # For analysis
            "text_features": text_features,  # For debugging
            "audio_features": audio_features  # For debugging
        }


# Use Project 1's robust dataset and training infrastructure
class MultimodalSentimentDataset(Dataset):
    """
    Robust dataset class with comprehensive error handling
    """

    def __init__(self, csv_path, audio_dir, audio_processor=None, text_tokenizer=None, augment=False):
        self.augment = augment

        try:
            self.df = pd.read_csv(csv_path, sep=";")
            self._standardize_column_names()

            # Robust label mapping
            self.label_map = {
                'negative': 0, -1: 0, '-1': 0, -1.0: 0,
                'neutral': 1, 0: 1, '0': 1, 0.0: 1,
                'positive': 2, 1: 2, '1': 2, 1.0: 2
            }

            self.audio_dir = audio_dir
            self.audio_processor = audio_processor or Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.text_tokenizer = text_tokenizer or RobertaTokenizer.from_pretrained("roberta-large")

            logger.info(f"Dataset loaded from {csv_path}")
            logger.info(f"Dataset size: {len(self.df)} samples")
            self._count_labels()

        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            self.df = pd.DataFrame(columns=["Audio_ID", "Sentiment", "Utterance"])
            self.audio_dir = audio_dir
            self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def _standardize_column_names(self):
        """Standardize column names"""
        audio_id_columns = ["Audio ID", "AudioID", "Audio_ID", "id", "ID", "Id"]
        for col in audio_id_columns:
            if col in self.df.columns:
                self.df.rename(columns={col: "Audio_ID"}, inplace=True)
                break

        sentiment_columns = ["Sentiment", "sentiment", "label", "Label", "class", "Class"]
        for col in sentiment_columns:
            if col in self.df.columns:
                self.df.rename(columns={col: "Sentiment"}, inplace=True)
                break

        text_columns = ["Utterance", "utterance", "text", "Text", "transcript", "Transcript"]
        for col in text_columns:
            if col in self.df.columns:
                self.df.rename(columns={col: "Utterance"}, inplace=True)
                break

        required_columns = ["Audio_ID", "Sentiment", "Utterance"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} not found in CSV file")

    def _count_labels(self):
        """Count label distribution"""
        label_counts = {0: 0, 1: 0, 2: 0}
        for idx in range(len(self.df)):
            try:
                label = self.df.iloc[idx]['Sentiment']
                label_idx = self.label_map.get(label, 1)
                label_counts[label_idx] += 1
            except Exception as e:
                logger.warning(f"Error counting label at index {idx}: {e}")

        logger.info(
            f"Label distribution: Negative: {label_counts[0]}, Neutral: {label_counts[1]}, Positive: {label_counts[2]}")

    def speech_file_to_array_fn(self, path):
        """Load and preprocess audio"""
        try:
            TARGET_SAMPLE_RATE = 16000
            MAX_AUDIO_LENGTH = 2.5  # Reduced for memory efficiency

            speech_array, sampling_rate = torchaudio.load(path)

            if speech_array.shape[0] > 1:
                speech_array = speech_array.mean(dim=0, keepdim=True)

            speech_array = speech_array.squeeze().numpy()

            if sampling_rate != TARGET_SAMPLE_RATE:
                speech_array = librosa.resample(
                    speech_array, orig_sr=sampling_rate, target_sr=TARGET_SAMPLE_RATE
                )

            if self.augment:
                try:
                    import noisereduce as nr
                    speech_array = nr.reduce_noise(y=speech_array, sr=TARGET_SAMPLE_RATE)
                except:
                    pass

            max_length = int(TARGET_SAMPLE_RATE * MAX_AUDIO_LENGTH)
            if len(speech_array) > max_length:
                speech_array = speech_array[:max_length]

            return speech_array

        except Exception as e:
            logger.warning(f"Error loading audio {path}: {e}")
            return np.zeros(int(16000 * 2.5))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get dataset item with robust error handling"""
        try:
            label_text = self.df.iloc[idx]['Sentiment']
            label = self.label_map.get(label_text, 1)

            audio_id = str(self.df.iloc[idx]['Audio_ID'])
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")

            speech_array = self.speech_file_to_array_fn(audio_path)

            audio_inputs = self.audio_processor(
                speech_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            utterance = self.df.iloc[idx]['Utterance']

            text_inputs = self.text_tokenizer(
                utterance,
                truncation=True,
                max_length=96,  # Reduced for memory efficiency
                padding="max_length",
                return_tensors="pt"
            )

            return {
                "text_input_ids": text_inputs.input_ids.squeeze(),
                "text_attention_mask": text_inputs.attention_mask.squeeze(),
                "audio_input_values": audio_inputs.input_values.squeeze(),
                "labels": label
            }

        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            return {
                "text_input_ids": torch.zeros(96, dtype=torch.long),
                "text_attention_mask": torch.zeros(96, dtype=torch.long),
                "audio_input_values": torch.zeros(16000 * 2),
                "labels": 1
            }


class HybridDataCollator:
    """Data collator with memory optimization """

    def __init__(self, audio_processor=None):
        self.audio_processor = audio_processor

    def __call__(self, features):
        batch = {}

        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        # Text inputs
        batch["text_input_ids"] = torch.stack([f["text_input_ids"] for f in features])
        batch["text_attention_mask"] = torch.stack([f["text_attention_mask"] for f in features])

        # Audio inputs with proper padding
        audio_inputs = [{"input_values": f["audio_input_values"]} for f in features]

        if self.audio_processor:
            audio_batch = self.audio_processor.pad(audio_inputs, padding=True, return_tensors="pt")
            batch["audio_input_values"] = audio_batch["input_values"]
            if "attention_mask" in audio_batch:
                batch["audio_attention_mask"] = audio_batch["attention_mask"]
        else:
            max_length = max(len(f["audio_input_values"]) for f in features)
            padded_inputs = []
            for f in features:
                input_values = f["audio_input_values"]
                padding_length = max_length - len(input_values)
                if padding_length > 0:
                    padded_input = torch.nn.functional.pad(input_values, (0, padding_length))
                else:
                    padded_input = input_values
                padded_inputs.append(padded_input)
            batch["audio_input_values"] = torch.stack(padded_inputs)

        return batch


def main():
    parser = argparse.ArgumentParser(description="Hybrid multimodal fusion using existing trained models")

    # Paths to your existing trained models
    parser.add_argument("--text_model_path", type=str, required=True,
                        help="Path to your trained text model (.pt or .safetensors)")
    parser.add_argument("--audio_model_path", type=str, required=True,
                        help="Path to your trained audio model (.pt, .safetensors, or directory)")

    # Data arguments
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/validation.csv")
    parser.add_argument("--audio_dir", type=str, default="data/train")
    parser.add_argument("--val_audio_dir", type=str, default="data/validation")

    # Model arguments
    parser.add_argument("--text_model_name", type=str, default="roberta-large")
    parser.add_argument("--audio_model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--fusion_dim", type=int, default=512)

    # Training arguments optimized for 4GB VRAM
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)

    args = parser.parse_args()

    # Set up logging
    logger.info("=" * 80)
    logger.info("HYBRID MULTIMODAL FUSION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Text model path: {args.text_model_path}")
    logger.info(f"Audio model path: {args.audio_model_path}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Memory-optimized for 4GB VRAM")

    # Initialize processors
    audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(args.text_model_name)

    # Create datasets
    train_dataset = MultimodalSentimentDataset(
        csv_path=args.train_csv,
        audio_dir=args.audio_dir,
        audio_processor=audio_processor,
        text_tokenizer=text_tokenizer
    )

    val_dataset = MultimodalSentimentDataset(
        csv_path=args.val_csv,
        audio_dir=args.val_audio_dir,
        audio_processor=audio_processor,
        text_tokenizer=text_tokenizer
    )

    # Create data loaders (memory optimized)
    data_collator = HybridDataCollator(audio_processor=audio_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=False
    )

    # Initialize hybrid model
    model = HybridMultimodalModel(
        text_model_path=args.text_model_path,
        audio_model_path=args.audio_model_path,
        text_model_name=args.text_model_name,
        audio_model_name=args.audio_model_name,
        fusion_dim=args.fusion_dim
    ).to(DEVICE)

    # Optimizer (only train fusion + classifier, keep your trained models frozen)
    optimizer = optim.AdamW([
        {'params': model.fusion_layer.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info("Starting hybrid multimodal training...")
    logger.info(f"Only training fusion layer + classifier (keeping your trained models frozen)")

    # Freeze the pretrained models
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.audio_model.parameters():
        param.requires_grad = False

    logger.info("Frozen pretrained text and audio models")

    # Training variables
    best_val_f1 = 0
    early_stop_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MODEL_DIR, f"hybrid_fusion_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    attention_weights_history = []

    class_names = ['negative', 'neutral', 'positive']

    logger.info(f"Starting training for {args.epochs} epochs...")

    # Training loop (memory-efficient)
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for step, batch in progress_bar:
            try:
                # Clear memory periodically
                if step % 10 == 0:
                    clear_memory()

                # Move to device
                text_input_ids = batch["text_input_ids"].to(DEVICE)
                text_attention_mask = batch["text_attention_mask"].to(DEVICE)
                audio_input_values = batch["audio_input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # Zero gradients at the start of accumulation
                if step % args.grad_accum_steps == 0:
                    optimizer.zero_grad()

                # Forward pass
                outputs = model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    audio_input_values=audio_input_values,
                    labels=labels
                )

                loss = outputs["loss"] / args.grad_accum_steps
                loss.backward()

                # Track metrics
                train_loss += loss.item() * args.grad_accum_steps
                logits = outputs["logits"]
                _, preds = torch.max(logits, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                # Optimizer step
                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        list(model.fusion_layer.parameters()) + list(model.classifier.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    clear_memory()

                # Update progress
                progress_bar.set_description(f"Training (loss: {loss.item():.4f})")

                # Clear batch data
                del text_input_ids, text_attention_mask, audio_input_values, labels, outputs, logits

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at step {step}, clearing cache...")
                    clear_memory()
                    continue
                else:
                    raise e

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds) if train_preds else 0
        train_f1 = f1_score(train_labels, train_preds, average='weighted') if train_preds else 0

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_attention_weights = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    text_input_ids = batch["text_input_ids"].to(DEVICE)
                    text_attention_mask = batch["text_attention_mask"].to(DEVICE)
                    audio_input_values = batch["audio_input_values"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)

                    outputs = model(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        audio_input_values=audio_input_values,
                        labels=labels
                    )

                    val_loss += outputs["loss"].item() * labels.size(0)
                    _, preds = torch.max(outputs["logits"], 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    if "attention_weights" in outputs:
                        val_attention_weights.append(outputs["attention_weights"])

                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_accuracy = accuracy_score(val_labels, val_preds) if val_preds else 0
        val_f1 = f1_score(val_labels, val_preds, average='weighted') if val_preds else 0

        # Attention analysis
        if val_attention_weights:
            avg_attention = torch.cat(val_attention_weights, dim=0).mean(dim=0)
            logger.info(f"Attention weights - Text: {avg_attention[0]:.3f}, Audio: {avg_attention[1]:.3f}")
            attention_weights_history.extend(val_attention_weights)

        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        logger.info(f"Epoch time: {epoch_time:.2f}s")

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': avg_val_loss,
                'args': args,
                'attention_weights_history': attention_weights_history
            }, os.path.join(run_dir, f"best_hybrid_model_f1_{val_f1:.4f}.pt"))

            logger.info(f"New best model saved! Val F1: {val_f1:.4f}")

            # Plot confusion matrix
            if val_preds and val_labels:
                cm = confusion_matrix(val_labels, val_preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
                plt.close()
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered!")
                break

        # Plot training metrics
        if epoch % 3 == 0:  # Every 3 epochs to save time
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(train_accuracies, label='Train Acc')
            plt.plot(val_accuracies, label='Val Acc')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Progress')
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(train_f1s, label='Train F1')
            plt.plot(val_f1s, label='Val F1')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.title('F1 Score Progress')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'training_progress.png'))
            plt.close()

        clear_memory()  # Clear memory after each epoch

    # Final results
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Validation F1: {best_val_f1:.4f}")
    logger.info(f"Model saved to: {run_dir}")

    if attention_weights_history:
        final_attention = torch.cat(attention_weights_history[-10:], dim=0).mean(dim=0)
        logger.info(f"Final attention weights - Text: {final_attention[0]:.3f}, Audio: {final_attention[1]:.3f}")

        # Plot final attention analysis
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        text_weights = [w.mean(dim=0)[0].item() for w in attention_weights_history[::5]]
        audio_weights = [w.mean(dim=0)[1].item() for w in attention_weights_history[::5]]
        plt.plot(text_weights, label='Text Attention', marker='o')
        plt.plot(audio_weights, label='Audio Attention', marker='s')
        plt.xlabel('Validation Steps')
        plt.ylabel('Attention Weight')
        plt.title('Attention Evolution During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        final_weights = torch.cat(attention_weights_history[-5:], dim=0)
        text_dist = final_weights[:, 0].cpu().numpy()
        audio_dist = final_weights[:, 1].cpu().numpy()
        plt.hist(text_dist, alpha=0.7, label='Text', bins=20, color='blue')
        plt.hist(audio_dist, alpha=0.7, label='Audio', bins=20, color='red')
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        plt.title('Final Attention Distribution')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'attention_analysis.png'))
        plt.close()

    # Save final metrics
    final_metrics = {
        'best_val_f1': best_val_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'final_attention_weights': final_attention.tolist() if attention_weights_history else None
    }

    with open(os.path.join(run_dir, 'training_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Training metrics saved to: {os.path.join(run_dir, 'training_metrics.json')}")
    logger.info("Ready to compare with your unimodal baselines!")


if __name__ == "__main__":
    main()