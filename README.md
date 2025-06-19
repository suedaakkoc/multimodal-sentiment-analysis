# Multimodal Sentiment Analysis System

A comprehensive sentiment analysis system that supports text, audio, and multimodal fusion approaches using two different architectures: **Pretrained Models** (RoBERTa + Wav2Vec2) and **CNN-based Models**.

## Overview

This system implements multimodal sentiment analysis with three main modalities:
- **Text-only**: Sentiment analysis from text transcripts
- **Audio-only**: Sentiment analysis from audio features
- **Multimodal Fusion**: Combined text and audio analysis

### Supported Datasets
- IEMOCAP
- MELD
- Custom datasets following the specified format

## Installation

```bash
pip install torch torchaudio transformers scikit-learn pandas numpy matplotlib seaborn librosa noisereduce tqdm
```

## Data Structure

Your data should be organized as follows:

```
data/
├── train/
│   ├── 1001.flac
│   ├── 1002.flac
│   └── ...
├── validation/
│   ├── 2001.flac
│   └── ...
├── test/
│   ├── 3001.flac
│   └── ...
├── train.csv
├── validation.csv
└── test.csv
```

**CSV Format** (semicolon-separated):
```
Audio_ID;Sentiment;Utterance
1001;positive;This is a great day!
1002;negative;I'm feeling sad today
1003;neutral;The weather is okay
```

**Audio Requirements**:
- Format: FLAC files
- Naming: `{Audio_ID}.flac`

---

## Architecture 1: Pretrained Models

Uses state-of-the-art pretrained models for each modality:
- **Text**: RoBERTa-large with attention pooling
- **Audio**: Wav2Vec2-base/large with CNN layers
- **Fusion**: Attention-based feature-level fusion

### Training Pretrained Models

#### 1. Train Text Model
```bash
python text_improved_train.py
```

**Configuration** (modify in code):
- `batch_size = 16`
- `num_epochs = 5`
- `learning_rate = 1.5e-5`
- `max_length = 128`

**Output**: Models saved to `./classification_checkpoints/`

#### 2. Train Audio Model
```bash
python audio_train.py
```

**Configuration** (modify in code):
```python
USE_WAV2VEC2 = True  # Set to False for CNN
WAV2VEC2_MODEL = "facebook/wav2vec2-base"  # or wav2vec2-large
POOLING_MODE = "mean"  # "mean", "sum", or "max"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
```

**Output**: Models saved to `./models/run_[model_type]_[timestamp]/`

#### 3. Train Multimodal Fusion
```bash
python early_fusion.py \
    --text_model_path ./classification_checkpoints/model_epoch_3_val_loss_0.2543_f1_0.8234.pt \
    --audio_model_path ./models/run_wav2vec2_20240315_143022/best_model \
    --train_csv data/train.csv \
    --val_csv data/validation.csv \
    --audio_dir data/train \
    --val_audio_dir data/validation \
    --batch_size 8 \
    --epochs 20
```

**Key Arguments**:
- `--text_model_path`: Path to trained text model
- `--audio_model_path`: Path to trained audio model
- `--fusion_dim`: Fusion layer dimension (default: 512)
- `--grad_accum_steps`: Gradient accumulation steps (default: 2)

**Output**: Models saved to `./hybrid_models/hybrid_fusion_[timestamp]/`

### Testing Pretrained Models

#### Test Text Model
```bash
python text_improved_test.py \
    --model_path ./classification_checkpoints/best_model.pt \
    --test_file data/test.csv \
    --output_dir ./text_results
```

#### Test Audio Model
```bash
python audio_test.py
```
Automatically finds and tests all trained audio models.

#### Test Fusion Model
```bash
python test_fusion.py \
    --model_path ./hybrid_models/hybrid_fusion_*/best_model.pt \
    --test_csv data/test.csv \
    --test_audio_dir data/test \
    --output_dir ./fusion_results
```

---

## Architecture 2: CNN-Based Models

Custom CNN architectures for all modalities with unified training pipeline.

### Training CNN Models

#### Basic Training Command
```bash
python multimodal_train.py --modality [audio|text|both] [OPTIONS]
```

#### 1. Audio-Only Model
```bash
python multimodal_train.py \
    --modality audio \
    --train_csv data/train.csv \
    --val_csv data/validation.csv \
    --audio_train_dir data/train \
    --audio_val_dir data/validation \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --audio_dropout 0.5
```

#### 2. Text-Only Model
```bash
python multimodal_train.py \
    --modality text \
    --train_csv data/train.csv \
    --val_csv data/validation.csv \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --text_dropout 0.3 \
    --max_length 128
```

#### 3. Multimodal Fusion Model
```bash
python multimodal_train.py \
    --modality both \
    --train_csv data/train.csv \
    --val_csv data/validation.csv \
    --audio_train_dir data/train \
    --audio_val_dir data/validation \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --audio_dropout 0.5 \
    --text_dropout 0.3 \
    --fusion_dropout 0.6
```

### Testing CNN Models

```bash
python multimodal_test.py \
    --modality [audio|text|both] \
    --model_path ./multimodal_models/[model_run]/best_model.pt \
    --test_csv data/test.csv \
    --audio_test_dir data/test \
    --batch_size 32 \
    --output_dir ./test_results
```

### Dataset-Specific CNN Configurations

#### IEMOCAP Dataset
```bash
# Audio-only
python multimodal_train.py --modality audio --weight_decay 1e-4 --audio_dropout 0.3 --epochs 50

# Text-only  
python multimodal_train.py --modality text --weight_decay 1e-4 --text_dropout 0.3 --epochs 70

# Fusion
python multimodal_train.py --modality both --weight_decay 1e-3 --audio_dropout 0.3 --text_dropout 0.3 --epochs 50
```

#### MELD Dataset
```bash
# All models use dropout 0.0
python multimodal_train.py --modality [audio|text|both] --weight_decay 1e-4 --audio_dropout 0.0 --text_dropout 0.0 --epochs 50
```

---

## Output Files

### Model Checkpoints
```
./classification_checkpoints/     # Pretrained text models
./models/run_[type]_[timestamp]/  # Pretrained audio models
./hybrid_models/                  # Pretrained fusion models
./multimodal_models/              # CNN-based models
```

### Results
Each model produces:
- `*_predictions.csv`: Detailed predictions with probabilities
- `confusion_matrix*.png`: Confusion matrix visualization
- `roc_curves*.png`: ROC curves for each class
- `probability_distributions*.png`: Probability distribution analysis
- `*_metrics.json`: Comprehensive performance metrics
- `attention_analysis.png`: Attention weights (fusion models only)

### Training Outputs
- `training_metrics.png`: Training curves
- `model_info.json`: Model configuration and metrics
- `tokenizer/`: Saved tokenizer (for text models)

---

### Performance Comparison
![results](https://github.com/user-attachments/assets/d614f18c-7530-4d42-ba29-244d14c1cd98)

