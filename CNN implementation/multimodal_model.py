import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Squeeze-and-Excitation for better feature recalibration
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE block
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AudioModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(AudioModel, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Residual blocks with increasing channels
        self.res_block1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature dimension after pooling
        self.feature_dim = 256

        # Initialize weights for better training
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.pool1(self.res_block1(x))
        x = self.pool2(self.res_block2(x))
        x = self.pool3(self.res_block3(x))

        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, feature_dim]

        return x


class TextModel(nn.Module):
    def __init__(self,
                 vocab_size=30522,  # BERT vocab size
                 embedding_dim=256,
                 filter_sizes=[3, 4, 5],
                 num_filters=128,
                 dropout_rate=0.3,
                 max_length=128):
        """
        CNN-based text model using BERT tokenizer.
        """
        super(TextModel, self).__init__()

        self.max_length = max_length
        self.embedding_dim = embedding_dim

        # Embedding layer (not pre-trained)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])

        # Calculate the total number of features after convolution and pooling
        self.feature_dim = len(filter_sizes) * num_filters

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

        # Initialize conv layers
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the text model.
        Returns features for fusion.
        """
        # Apply embeddings
        embedded = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)

        # Transpose for Conv1d: (batch_size, embedding_dim, seq_length)
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_output_length)

            # Max pooling over time
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)

            conv_outputs.append(pooled)

        # Concatenate outputs from all convolutions
        features = torch.cat(conv_outputs, dim=1)  # (batch_size, feature_dim)

        return features


class MultifactorFeatureNormalizer(nn.Module):
    """
    Normalizes features from different modalities to balance their contributions
    """

    def __init__(self, audio_dim, text_dim):
        super(MultifactorFeatureNormalizer, self).__init__()
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        # Learnable scaling parameters
        self.audio_scale = nn.Parameter(torch.ones(1))
        self.text_scale = nn.Parameter(torch.ones(1))

    def forward(self, audio_features, text_features):
        # Apply layer normalization
        audio_normalized = self.audio_norm(audio_features)
        text_normalized = self.text_norm(text_features)

        # Apply learnable scaling
        audio_normalized = audio_normalized * self.audio_scale
        text_normalized = text_normalized * self.text_scale

        return audio_normalized, text_normalized


class ModalityFusion(nn.Module):
    """
    Implements a more sophisticated fusion strategy for multimodal learning
    """

    def __init__(self, audio_dim, text_dim, hidden_dim=256, dropout_rate=0.5, audio_dropout_prob=0.3, text_dropout_prob=0.3):
        super(ModalityFusion, self).__init__()

        # Project individual modalities to hidden dimension
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Attention for cross-modal interaction
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=1)
        )

        # Final fusion output dimension
        self.output_dim = hidden_dim

        self.audio_dropout_prob = audio_dropout_prob
        self.text_dropout_prob = text_dropout_prob

    def forward(self, audio_features, text_features):
        if self.training:
            if torch.rand(1).item() < self.audio_dropout_prob:
                audio_features = torch.zeros_like(audio_features)
            if torch.rand(1).item() < self.text_dropout_prob:
                text_features = torch.zeros_like(text_features)

        # Project features
        audio_projected = self.audio_projector(audio_features)
        text_projected = self.text_projector(text_features)

        # Compute cross-modal attention weights
        combined = torch.cat([audio_projected, text_projected], dim=1)
        attention_weights = self.attention(combined)

        # Apply weights to features
        audio_weight = attention_weights[:, 0].unsqueeze(1).expand_as(audio_projected)
        text_weight = attention_weights[:, 1].unsqueeze(1).expand_as(text_projected)

        # Weighted sum
        fused_features = (audio_projected * audio_weight) + (text_projected * text_weight)

        return fused_features


class MultimodalSentimentModel(nn.Module):
    def __init__(self, modality='both',
                 num_classes=3,
                 audio_dropout=0.5,
                 text_dropout=0.3,
                 fusion_dropout=0.5,
                 vocab_size=30522,
                 embedding_dim=256,
                 filter_sizes=[3, 4, 5],
                 num_filters=128,
                 max_length=128):
        """
        Improved multimodal sentiment analysis model supporting audio, text, or both.

        Args:
            modality: 'audio', 'text', or 'both'
            num_classes: Number of output classes (3 for negative, neutral, positive)
        """
        super(MultimodalSentimentModel, self).__init__()

        self.modality = modality

        # Initialize audio model if needed
        if modality in ['audio', 'both']:
            self.audio_model = AudioModel(num_classes=num_classes, dropout_rate=audio_dropout)
            self.audio_feature_dim = self.audio_model.feature_dim
        else:
            self.audio_model = None
            self.audio_feature_dim = 0

        # Initialize text model if needed
        if modality in ['text', 'both']:
            self.text_model = TextModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                dropout_rate=text_dropout,
                max_length=max_length
            )
            self.text_feature_dim = self.text_model.feature_dim
        else:
            self.text_model = None
            self.text_feature_dim = 0

        # Combined feature dimension after fusion
        if modality == 'both':
            # Feature normalization layer
            self.feature_normalizer = MultifactorFeatureNormalizer(
                audio_dim=self.audio_feature_dim,
                text_dim=self.text_feature_dim
            )

            # Fusion layer
            self.fusion_layer = ModalityFusion(
                audio_dim=self.audio_feature_dim,
                text_dim=self.text_feature_dim,
                hidden_dim=256,
                dropout_rate=fusion_dropout
            )

            # Set fusion dimension to fusion layer output dimension
            self.fusion_dim = self.fusion_layer.output_dim
        else:
            self.fusion_dim = self.audio_feature_dim + self.text_feature_dim

        # Classifier layers
        if modality == 'audio':
            self.classifier = self._create_audio_classifier(num_classes, audio_dropout)
        elif modality == 'text':
            self.classifier = self._create_text_classifier(num_classes, text_dropout)
        else:  # 'both'
            self.classifier = self._create_fusion_classifier(num_classes, fusion_dropout)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def _create_audio_classifier(self, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.audio_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes)
        )

    def _create_text_classifier(self, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.text_feature_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def _create_fusion_classifier(self, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio=None, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the model based on modality.

        Args:
            audio: Audio input for audio modality (batch_size, channels, height, width)
            input_ids: Token IDs for text modality (batch_size, seq_length)
            attention_mask: Attention mask for text modality
            labels: True labels (optional)

        Returns:
            Dictionary containing loss, logits, and probabilities
        """
        if self.modality == 'audio':
            if audio is None:
                raise ValueError("Audio input is required for audio modality")

            # Process audio input
            features = self.audio_model(audio)

        elif self.modality == 'text':
            if input_ids is None:
                raise ValueError("Text input is required for text modality")

            # Process text input
            features = self.text_model(input_ids, attention_mask)

        else:  # 'both'
            if audio is None or input_ids is None:
                raise ValueError("Both audio and text inputs are required for multimodal fusion")

            # Process both modalities
            audio_features = self.audio_model(audio)
            text_features = self.text_model(input_ids, attention_mask)

            # Normalize features
            audio_normalized, text_normalized = self.feature_normalizer(audio_features, text_features)

            # Apply fusion
            features = self.fusion_layer(audio_normalized, text_normalized)

        # Apply classifier based on modality
        logits = self.classifier(features)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

            # Add L2 regularization for 'both' modality to prevent overfitting
            if self.modality == 'both':
                l2_lambda = 0.001
                l2_reg = 0
                for param in self.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg

        return {"loss": loss, "logits": logits, "probabilities": probabilities}


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