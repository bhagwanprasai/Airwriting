"""
HTR Model Architecture - Inference Only
========================================
Contains ONLY the model classes needed for prediction.
No training code, no data loading, no optimization.

Usage in Flask:
    from htr_model import Seq2SeqAttention, LabelEncoder, InferenceConfig
    
    # Load model
    encoder = LabelEncoder()
    encoder.build_vocab(DEFAULT_VOCAB)
    
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=256,
        decoder_hidden=256,
        attention_hidden=128,
        dropout=0.3
    )
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION (Inference)
# ══════════════════════════════════════════════════════════════════════════

class InferenceConfig:
    """Minimal config needed for inference"""
    IMG_HEIGHT = 64
    MIN_WIDTH = 32
    MAX_WIDTH = 512
    ENCODER_HIDDEN = 256
    DECODER_HIDDEN = 256
    ATTENTION_HIDDEN = 128
    DROPOUT = 0.3
    MAX_OUTPUT_LENGTH = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════
# SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════════════

class SpecialTokens:
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

# Default vocabulary (must match training)
DEFAULT_VOCAB = sorted(set(
    " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ\\]abcdefghijklmnopqrstuvwxyz"
))

# ══════════════════════════════════════════════════════════════════════════
# LABEL ENCODER
# ══════════════════════════════════════════════════════════════════════════

class LabelEncoder:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        """Build vocabulary from text list"""
        chars = sorted(set(''.join(texts)))
        
        self.char2idx = {
            SpecialTokens.PAD: 0,
            SpecialTokens.SOS: 1,
            SpecialTokens.EOS: 2,
            SpecialTokens.UNK: 3
        }
        
        for i, char in enumerate(chars, start=4):
            self.char2idx[char] = i
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
    def encode(self, text):
        """Encode text to indices"""
        indices = [self.char2idx[SpecialTokens.SOS]]
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[SpecialTokens.UNK]))
        indices.append(self.char2idx[SpecialTokens.EOS])
        return indices
    
    def decode(self, indices):
        """Decode indices to text"""
        chars = []
        for idx in indices:
            if idx == self.char2idx[SpecialTokens.EOS]:
                break
            if idx not in [self.char2idx[SpecialTokens.PAD], 
                          self.char2idx[SpecialTokens.SOS]]:
                chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)
    
    def num_classes(self):
        return len(self.char2idx)

# ══════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE CLASSES
# ══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CNNEncoder(nn.Module):
    """CNN feature extractor"""
    def __init__(self, dropout=0.3):
        super(CNNEncoder, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_pool(out)
        out = self.dropout(out)
        out = out.squeeze(2)
        out = out.permute(0, 2, 1)
        return out

class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder"""
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
    def forward(self, x):
        outputs, hidden = self.lstm(x)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    def __init__(self, encoder_hidden, decoder_hidden, attention_hidden):
        super(BahdanauAttention, self).__init__()
        
        self.attention_hidden = attention_hidden
        self.encoder_projection = nn.Linear(encoder_hidden * 2, attention_hidden, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden, attention_hidden, bias=False)
        self.energy = nn.Linear(attention_hidden, 1, bias=False)
        self.layer_norm = nn.LayerNorm(attention_hidden)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        
        encoder_proj = self.encoder_projection(encoder_outputs)
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)
        combined = encoder_proj + decoder_proj
        combined = self.layer_norm(combined)
        energy = self.energy(torch.tanh(combined))
        energy = energy.squeeze(2)
        
        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(energy, dim=1)
        attention_weights = self.dropout(attention_weights)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights

class LSTMDecoder(nn.Module):
    """LSTM decoder with attention"""
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.decoder_hidden = decoder_hidden
        
        self.embedding = nn.Embedding(num_classes, decoder_hidden, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(encoder_hidden, decoder_hidden, attention_hidden)
        
        self.lstm = nn.LSTMCell(
            input_size=decoder_hidden + encoder_hidden * 2,
            hidden_size=decoder_hidden
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(decoder_hidden + encoder_hidden * 2, decoder_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        embedded = self.embedding(input_token)
        embedded = self.embedding_dropout(embedded)
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        lstm_input = torch.cat([embedded, context], dim=1)
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        output_input = torch.cat([hidden, context], dim=1)
        output = self.fc_out(output_input)
        
        return output, hidden, cell, attention_weights

class Seq2SeqAttention(nn.Module):
    """Complete Seq2Seq model with attention"""
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.3, drop_path=0.0):
        super(Seq2SeqAttention, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_encoder = CNNEncoder(dropout=dropout)
        self.lstm_encoder = LSTMEncoder(512, encoder_hidden, dropout=dropout)
        self.decoder = LSTMDecoder(num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=dropout)
        
        bridge_input_dim = encoder_hidden * 2
        self.encoder_to_decoder_h = nn.Linear(bridge_input_dim, decoder_hidden)
        self.encoder_to_decoder_c = nn.Linear(bridge_input_dim, decoder_hidden)
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.0, attention_mask=None):
        """Forward pass - inference mode if targets=None"""
        batch_size = images.size(0)
        device = images.device
        
        # Encode
        cnn_features = self.cnn_encoder(images)
        encoder_outputs, (h_n, c_n) = self.lstm_encoder(cnn_features)
        
        # Bridge
        top_h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        top_c = torch.cat([c_n[-2], c_n[-1]], dim=1)
        decoder_hidden = torch.tanh(self.encoder_to_decoder_h(top_h))
        decoder_cell = torch.tanh(self.encoder_to_decoder_c(top_c))
        
        seq_len = encoder_outputs.size(1)
        
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            mask = attention_mask.to(device)
            if mask.size(1) != seq_len:
                mask = mask[:, :seq_len] if mask.size(1) > seq_len else F.pad(mask, (0, seq_len - mask.size(1)), value=False)
        
        # INFERENCE MODE (targets=None)
        max_len = 32  # InferenceConfig.MAX_OUTPUT_LENGTH
        outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
        
        # Start with SOS token
        input_token = torch.full((batch_size,), 1, dtype=torch.long, device=device)  # SOS = 1
        
        for t in range(max_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
            )
            
            outputs[:, t] = output
            input_token = output.argmax(dim=1)
            
            # Stop if all sequences predict EOS
            if (input_token == 2).all():  # EOS = 2
                break
        
        return outputs, None
    
    def predict(self, image_tensor, encoder):
        """
        Convenience method for prediction.
        
        Args:
            image_tensor: [1, 1, H, W] tensor
            encoder: LabelEncoder instance
            
        Returns:
            predicted_text: str
            confidence: float
        """
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(image_tensor)
            
            # Get predicted indices
            predicted_indices = outputs[0].argmax(dim=1).cpu().numpy()
            
            # Find EOS position (token ID = 2)
            eos_token = 2
            try:
                eos_pos = list(predicted_indices).index(eos_token)
            except ValueError:
                # If no EOS found, use all timesteps (shouldn't happen in practice)
                eos_pos = len(predicted_indices) - 1
            
            # Decode to text
            predicted_text = encoder.decode(predicted_indices)
            
            # Calculate confidence ONLY up to EOS (not including padding timesteps)
            # This fixes the low confidence issue where padding timesteps dilute the score
            probs = torch.softmax(outputs[0], dim=1)
            max_probs = probs[:eos_pos+1].max(dim=1)[0]  # Only up to and including EOS
            confidence = max_probs.mean().item()
            
            return predicted_text, confidence


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION - Load Model
# ══════════════════════════════════════════════════════════════════════════

def load_model(model_path, device='cpu', vocab=None):
    """
    Load trained model for inference.
    
    Args:
        model_path: Path to .pth file
        device: 'cuda' or 'cpu'
        vocab: List of characters (uses DEFAULT_VOCAB if None)
        
    Returns:
        model: Loaded Seq2SeqAttention model
        encoder: LabelEncoder instance
    """
    # Setup encoder
    encoder = LabelEncoder()
    if vocab is None:
        vocab = DEFAULT_VOCAB
    encoder.build_vocab(vocab)
    
    # Create model
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=256,
        decoder_hidden=256,
        attention_hidden=128,
        dropout=0.3
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"  Device: {device}")
    print(f"  Vocabulary size: {encoder.num_classes()}")
    
    return model, encoder


# ══════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("HTR Model Architecture - Inference Only")
    print("="*50)
    
    # Test encoder
    encoder = LabelEncoder()
    encoder.build_vocab(DEFAULT_VOCAB)
    print(f"Vocabulary size: {encoder.num_classes()}")
    
    # Test model creation
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=256,
        decoder_hidden=256,
        attention_hidden=128,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 64, 128)
    output, _ = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print("\n Model architecture loaded successfully!")