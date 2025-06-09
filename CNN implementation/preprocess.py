import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
import logging
import noisereduce as nr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
MAX_AUDIO_LENGTH = 3  # 3 seconds


# Audio Transformation Pipeline
class AudioTransform:
    def __init__(self, augment=False):
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.augment = augment

    def time_shift(self, audio, shift_limit=0.1):
        """Apply time shifting augmentation"""
        if random.random() < 0.5:
            return audio

        shift = int(random.random() * shift_limit * audio.shape[1])
        if random.random() > 0.5:
            # Shift right
            audio_shift = torch.zeros_like(audio)
            audio_shift[:, shift:] = audio[:, :-shift]
            return audio_shift
        else:
            # Shift left
            audio_shift = torch.zeros_like(audio)
            audio_shift[:, :-shift] = audio[:, shift:]
            return audio_shift

    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise"""
        if random.random() < 0.5:
            return audio

        noise = torch.randn_like(audio) * noise_factor
        return audio + noise

    def pitch_shift(self, audio, sr):
        """Simple pitch shifting approximation"""
        if random.random() < 0.5:
            return audio

        # Stretch or compress the audio
        factor = random.uniform(0.9, 1.1)
        indices = torch.arange(0, audio.shape[1], factor)
        indices = indices.clamp(0, audio.shape[1] - 1).long()
        return audio[:, indices]

    def spectral_mask(self, spec, max_mask_freq=8, max_mask_time=8):
        """Apply frequency and time masking"""
        if random.random() < 0.5:
            return spec

        # Frequency masking
        if random.random() < 0.5:
            freq_mask_size = random.randint(1, max_mask_freq)
            start_freq = random.randint(0, spec.shape[1] - freq_mask_size)
            spec[:, start_freq:start_freq + freq_mask_size, :] = 0

        # Time masking
        if random.random() < 0.5:
            time_mask_size = random.randint(1, max_mask_time)
            start_time = random.randint(0, spec.shape[2] - time_mask_size)
            spec[:, :, start_time:start_time + time_mask_size] = 0

        return spec

    def normalize(self, spec):
        """Normalize spectrogram with robust statistics"""
        # Mean and std across time and frequency, keeping batch dim
        mean = spec.mean([1, 2], keepdim=True)
        std = spec.std([1, 2], keepdim=True)

        # Add small epsilon to avoid division by zero
        normalized = (spec - mean) / (std + 1e-8)

        # Clip values to avoid outliers
        normalized = torch.clamp(normalized, min=-6, max=6)

        return normalized

    def __call__(self, audio, sr=SAMPLE_RATE):
        try:
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != SAMPLE_RATE:
                resampler = Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                audio = resampler(audio)

            # Apply data augmentation if enabled
            if self.augment:
                audio = self.time_shift(audio)
                audio = self.add_noise(audio)
                audio = self.pitch_shift(audio, SAMPLE_RATE)

            # 🎯 Reduce noise
            try:
                audio_np = audio.squeeze(0).cpu().numpy()  # shape: (samples,)
                reduced_np = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
                audio = torch.tensor(reduced_np).unsqueeze(0)
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")

            # Adjust length
            target_length = int(SAMPLE_RATE * MAX_AUDIO_LENGTH)
            if audio.shape[1] < target_length:
                padding = target_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            elif audio.shape[1] > target_length:
                audio = audio[:, :target_length]

            # Generate spectrogram
            spec = self.mel_spectrogram(audio)
            spec = self.amplitude_to_db(spec)

            if self.augment:
                spec = self.spectral_mask(spec)

            spec = self.normalize(spec)

            if torch.isnan(spec).any() or torch.isinf(spec).any():
                spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

            return spec

        except Exception as e:
            logger.error(f"Error in audio transformation: {e}")
            return torch.zeros((1, N_MELS, int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH) + 1))


# Dataset Class
class AudioSentimentDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None, augment=False):
        try:
            self.df = pd.read_csv(csv_path, sep=";")
            self.audio_dir = audio_dir
            self.transform = transform or AudioTransform(augment=augment)
            self.augment = augment

            # Create a more robust label mapping
            self.label_map = {
                'negative': 0, -1: 0, '-1': 0, -1.0: 0,
                'neutral': 1, 0: 1, '0': 1, 0.0: 1,
                'positive': 2, 1: 2, '1': 2, 1.0: 2
            }

            # Print dataset size
            logger.info(f"Dataset loaded from {csv_path}")
            logger.info(f"Dataset size: {len(self.df)} samples")

            # Count and print label distribution
            self._count_labels()
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            # Create an empty dataframe if loading fails
            self.df = pd.DataFrame(columns=["Audio ID", "Sentiment"])
            self.audio_dir = audio_dir
            self.transform = transform or AudioTransform(augment=False)
            self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def _count_labels(self):
        """Count and print the distribution of labels"""
        label_counts = {0: 0, 1: 0, 2: 0}  # negative, neutral, positive
        for idx in range(len(self.df)):
            try:
                label = self.df.iloc[idx]['Sentiment']
                label_idx = self.label_map.get(label, 1)  # Default to neutral if unknown
                label_counts[label_idx] += 1
            except Exception as e:
                logger.warning(f"Error counting label at index {idx}: {e}")

        logger.info(
            f"Label distribution: Negative: {label_counts[0]}, Neutral: {label_counts[1]}, Positive: {label_counts[2]}")
        return label_counts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            audio_id = str(self.df.iloc[idx]['Audio ID'])  # Ensure audio_id is string

            # Get label and convert to numerical
            label_text = self.df.iloc[idx]['Sentiment']
            label = self.label_map.get(label_text, 1)  # Default to neutral if unknown

            audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")

            try:
                audio, sr = torchaudio.load(audio_path)
            except FileNotFoundError:
                # Handle missing audio files by returning zeros
                logger.warning(f"Audio file {audio_path} not found, using silent audio")
                audio = torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH)
                sr = SAMPLE_RATE
            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}")
                audio = torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH)
                sr = SAMPLE_RATE

            # Apply transformation
            spectrogram = self.transform(audio, sr)

            return spectrogram, label

        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            # Return dummy data
            return torch.zeros((1, N_MELS, int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH) + 1)), 1


# Function to calculate class weights
def calculate_class_weights(csv_path):
    """Calculate class weights to handle imbalanced data"""
    try:
        df = pd.read_csv(csv_path, sep=";")

        # Create label mapping
        label_map = {
            'negative': 0, -1: 0, '-1': 0, -1.0: 0,
            'neutral': 1, 0: 1, '0': 1, 0.0: 1,
            'positive': 2, 1: 2, '1': 2, 1.0: 2
        }

        # Convert labels to numerical values
        y = []
        for label in df['Sentiment']:
            try:
                y.append(label_map.get(label, 1))  # Default to neutral if unknown
            except:
                y.append(1)  # Default to neutral

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )

        logger.info(f"Class weights: {class_weights}")
        return torch.tensor(class_weights, dtype=torch.float)

    except Exception as e:
        logger.error(f"Error calculating class weights: {e}")
        # Default weights if calculation fails
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)