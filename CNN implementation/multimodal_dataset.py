import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import logging
from transformers import BertTokenizer

from preprocess import AudioTransform


class MultimodalSentimentDataset(Dataset):
    """
    Dataset for multimodal sentiment analysis that can handle audio, text, or both modalities.
    """

    def __init__(self, csv_path, audio_dir=None, modality='both', transform=None, tokenizer=None,
                 max_length=128, augment=False):
        """
        Initialize the multimodal dataset.

        Args:
            csv_path: Path to CSV file with data
            audio_dir: Directory containing audio files (required for audio or both modalities)
            modality: 'audio', 'text', or 'both'
            transform: Audio transformation pipeline (required for audio or both modalities)
            tokenizer: BERT tokenizer (required for text or both modalities)
            max_length: Maximum sequence length for text tokenization
            augment: Whether to apply data augmentation
        """
        self.modality = modality
        self.max_length = max_length

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("multimodal_dataset.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        try:
            self.df = pd.read_csv(csv_path, sep = ";")

            # Check and rename columns if needed
            if 'Audio ID' in self.df.columns:
                self.df.rename(columns={'Audio ID': 'Audio_ID'}, inplace=True)

            if 'Audio_ID' not in self.df.columns and modality in ['audio', 'both']:
                possible_id_cols = ['AudioID', 'id', 'ID', 'Id']
                for col in possible_id_cols:
                    if col in self.df.columns:
                        self.df.rename(columns={col: 'Audio_ID'}, inplace=True)
                        break

                if 'Audio_ID' not in self.df.columns:
                    self.logger.warning("No Audio ID column found. Audio modality may not work correctly.")

            # Create a more robust label mapping
            self.label_map = {
                'negative': 0, -1: 0, '-1': 0, -1.0: 0,
                'neutral': 1, 0: 1, '0': 1, 0.0: 1,
                'positive': 2, 1: 2, '1': 2, 1.0: 2
            }

            # Audio modality setup
            if modality in ['audio', 'both']:
                if audio_dir is None:
                    raise ValueError("audio_dir is required for audio or both modalities")
                self.audio_dir = audio_dir
                self.transform = transform or AudioTransform(augment=augment)

            # Text modality setup
            if modality in ['text', 'both']:
                if tokenizer is None:
                    self.logger.info("No tokenizer provided, using bert-base-uncased by default")
                    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                else:
                    self.tokenizer = tokenizer

                # Check if the text column exists
                if "Utterance" not in self.df.columns and "Text" not in self.df.columns:
                    text_col = None
                    possible_text_cols = ['utterance', 'text', 'sentence', 'content']
                    for col in possible_text_cols:
                        if col in self.df.columns.str.lower():
                            text_col = self.df.columns[self.df.columns.str.lower() == col][0]
                            break

                    if text_col:
                        self.df.rename(columns={text_col: 'Utterance'}, inplace=True)
                    else:
                        self.logger.error("No text column found in the dataset")
                elif "Text" in self.df.columns and "Utterance" not in self.df.columns:
                    self.df.rename(columns={"Text": "Utterance"}, inplace=True)

            # Print dataset size
            self.logger.info(f"Dataset loaded from {csv_path} with modality: {modality}")
            self.logger.info(f"Dataset size: {len(self.df)} samples")

            # Count and print label distribution
            self._count_labels()

        except Exception as e:
            self.logger.error(f"Error initializing dataset: {e}")
            # Create an empty dataframe if loading fails
            self.df = pd.DataFrame(columns=["Audio_ID", "Utterance", "Sentiment"])
            self.modality = modality

            if modality in ['audio', 'both']:
                self.audio_dir = audio_dir or ""
                self.transform = transform or AudioTransform(augment=False)

            if modality in ['text', 'both']:
                self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")

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
                self.logger.warning(f"Error counting label at index {idx}: {e}")

        self.logger.info(
            f"Label distribution: Negative: {label_counts[0]}, Neutral: {label_counts[1]}, Positive: {label_counts[2]}")
        return label_counts

    def __len__(self):
        return len(self.df)

    def _process_audio(self, idx):
        """Process audio data for a given index"""
        try:
            index_in_df = idx % len(self.df)  # Handle possible indexing issues
            audio_id = str(self.df.iloc[index_in_df]['Audio_ID'])
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")

            try:
                audio, sr = torchaudio.load(audio_path)
            except FileNotFoundError:
                # Try alternative file extensions
                for ext in ['.wav', '.mp3', '.ogg']:
                    try:
                        alt_path = os.path.join(self.audio_dir, f"{audio_id}{ext}")
                        if os.path.exists(alt_path):
                            audio, sr = torchaudio.load(alt_path)
                            self.logger.info(f"Found audio file with alternative extension: {alt_path}")
                            break
                    except FileNotFoundError:
                        continue
                else:
                    # Handle missing audio files by returning zeros
                    self.logger.warning(
                        f"Audio file for ID {audio_id} not found (tried multiple extensions), using silent audio")
                    audio = torch.zeros(1, 22050 * 3)  # 3 seconds of silence at 22050Hz
                    sr = 22050
            except Exception as e:
                self.logger.warning(f"Error loading audio {audio_path}: {e}")
                audio = torch.zeros(1, 22050 * 3)
                sr = 22050

            # Apply transformation
            spectrogram = self.transform(audio, sr)
            return spectrogram

        except Exception as e:
            self.logger.error(f"Error processing audio at index {idx}: {e}")
            # Return dummy spectrogram with expected dimensions
            return torch.zeros((1, 64, 129))  # [channels, n_mels, time]

    def _process_text(self, idx):
        """Process text data for a given index"""
        try:
            utterance = str(self.df.iloc[idx]['Utterance'])

            # Tokenize the text
            encoding = self.tokenizer(
                utterance,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Remove batch dimension
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            return input_ids, attention_mask

        except Exception as e:
            self.logger.error(f"Error processing text at index {idx}: {e}")
            # Return dummy token IDs and attention mask
            return torch.zeros(self.max_length, dtype=torch.long), torch.zeros(self.max_length, dtype=torch.long)

    def _get_label(self, idx):
        """Get the sentiment label for a given index"""
        try:
            label_text = self.df.iloc[idx]['Sentiment']

            # Convert label to integer index (0, 1, 2)
            if isinstance(label_text, (int, float)) or (
                    isinstance(label_text, str) and label_text.lstrip('-').isdigit()):
                # For numeric labels like -1, 0, 1 or their string equivalents
                if float(label_text) < 0:
                    label = 0  # negative
                elif float(label_text) > 0:
                    label = 2  # positive
                else:
                    label = 1  # neutral
            else:
                # For string labels like 'negative', 'neutral', 'positive'
                label = self.label_map.get(label_text, 1)  # Default to neutral if unknown

            return torch.tensor(label, dtype=torch.long)
        except Exception as e:
            self.logger.warning(f"Error getting label at index {idx}: {e}, using neutral")
            return torch.tensor(1, dtype=torch.long)  # Default to neutral

    def __getitem__(self, idx):
        """
        Get a sample from the dataset based on the modality.

        Returns:
            dict: Sample data with keys depending on modality:
                - 'audio': spectrogram
                - 'input_ids', 'attention_mask': text inputs
                - 'label': sentiment label
        """
        try:
            result = {}

            # Get label for all modalities
            result['label'] = self._get_label(idx)

            # Process based on modality
            if self.modality == 'audio':
                result['audio'] = self._process_audio(idx)

            elif self.modality == 'text':
                input_ids, attention_mask = self._process_text(idx)
                result['input_ids'] = input_ids
                result['attention_mask'] = attention_mask

            else:  # 'both'
                result['audio'] = self._process_audio(idx)
                input_ids, attention_mask = self._process_text(idx)
                result['input_ids'] = input_ids
                result['attention_mask'] = attention_mask

            return result

        except Exception as e:
            self.logger.error(f"Error in __getitem__ at index {idx}: {e}")
            # Return dummy data based on modality
            dummy_result = {'label': torch.tensor(1, dtype=torch.long)}  # Default to neutral

            if self.modality == 'audio':
                dummy_result['audio'] = torch.zeros((1, 64, 129))

            elif self.modality == 'text':
                dummy_result['input_ids'] = torch.zeros(self.max_length, dtype=torch.long)
                dummy_result['attention_mask'] = torch.zeros(self.max_length, dtype=torch.long)

            else:  # 'both'
                dummy_result['audio'] = torch.zeros((1, 64, 129))
                dummy_result['input_ids'] = torch.zeros(self.max_length, dtype=torch.long)
                dummy_result['attention_mask'] = torch.zeros(self.max_length, dtype=torch.long)

            return dummy_result