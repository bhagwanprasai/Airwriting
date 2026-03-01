"""
Model architecture components for Seq2Seq Attention model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Training.config import Config


# ============================================================================
# DropPath (Stochastic Depth)
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# Residual Block
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and optional DropPath"""
    def __init__(self, in_channels, out_channels, drop_path=0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.drop_path(out)  # Apply stochastic depth
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


# ============================================================================
# Bahdanau Attention
# ============================================================================

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism with layer normalization"""
    def __init__(self, encoder_hidden, decoder_hidden, attention_hidden):
        super(BahdanauAttention, self).__init__()
        
        self.attention_hidden = attention_hidden
        self.encoder_projection = nn.Linear(encoder_hidden * 2, attention_hidden, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden, attention_hidden, bias=False)
        self.energy = nn.Linear(attention_hidden, 1, bias=False)
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(attention_hidden)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        
        # Project encoder outputs: [batch, seq_len, attention_hidden]
        encoder_proj = self.encoder_projection(encoder_outputs)
        
        # Project decoder hidden: [batch, 1, attention_hidden]
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)
        
        # Add and apply layer norm
        combined = encoder_proj + decoder_proj
        combined = self.layer_norm(combined)
        
        # Calculate energy: [batch, seq_len]
        energy = self.energy(torch.tanh(combined))
        energy = energy.squeeze(2)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))
        
        # Calculate attention weights: [batch, seq_len]
        attention_weights = F.softmax(energy, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Calculate context vector: [batch, encoder_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


# ============================================================================
# CNN Encoder
# ============================================================================

class CNNEncoder(nn.Module):
    """CNN encoder with clear pooling strategy for seq2seq"""
    def __init__(self, drop_path=0.1):
        super(CNNEncoder, self).__init__()
        
        # Block 1: 64 -> H/2, W/2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # H/2, W/2
        )
        
        # Block 2: 128 -> H/4, W/4
        self.block2 = nn.Sequential(
            ResidualBlock(64, 128, drop_path=drop_path),
            nn.MaxPool2d(2, 2)  # H/4, W/4
        )
        
        # Block 3: 256 -> H/8, W/4 (only pool height)
        self.block3 = nn.Sequential(
            ResidualBlock(128, 256, drop_path=drop_path),
            ResidualBlock(256, 256, drop_path=drop_path),
            nn.MaxPool2d((2, 1), (2, 1))  # H/8, same W
        )
        
        # Block 4: 512 -> H/16, W/4
        self.block4 = nn.Sequential(
            ResidualBlock(256, 512, drop_path=drop_path),
            nn.MaxPool2d((2, 1), (2, 1))  # H/16, same W
        )
        
        # For height 64: after block4 = 64/16 = 4
        # Adaptive pool to height 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.dropout = nn.Dropout(Config.DROPOUT)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: [batch, 1, 64, width]
        out = self.block1(x)   # [batch, 64, 32, W/2]
        out = self.block2(out)  # [batch, 128, 16, W/4]
        out = self.block3(out)  # [batch, 256, 8, W/4]
        out = self.block4(out)  # [batch, 512, 4, W/4]
        
        # Adaptive pool to get height = 1
        batch, channels, height, width = out.size()
        out = self.adaptive_pool(out)  # [batch, 512, 1, W/4]
        out = self.dropout(out)
        
        # Reshape for RNN: [batch, seq_len, features]
        out = out.squeeze(2)  # [batch, 512, W/4]
        out = out.permute(0, 2, 1)  # [batch, W/4, 512]
        
        return out


# ============================================================================
# LSTM Encoder
# ============================================================================

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
        
        # Initialize LSTM weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
    def forward(self, x):
        outputs, hidden = self.lstm(x)
        return outputs, hidden


# ============================================================================
# LSTM Decoder
# ============================================================================

class LSTMDecoder(nn.Module):
    """LSTM decoder with attention mechanism"""
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.decoder_hidden = decoder_hidden
        self.encoder_hidden = encoder_hidden
        
        # Embedding with proper initialization
        self.embedding = nn.Embedding(num_classes, decoder_hidden, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden, decoder_hidden, attention_hidden)
        
        # LSTM cell - takes embedded + context
        self.lstm = nn.LSTMCell(
            input_size=decoder_hidden + encoder_hidden * 2,
            hidden_size=decoder_hidden
        )
        
        # Output projection: hidden + context -> num_classes
        self.fc_out = nn.Sequential(
            nn.Linear(decoder_hidden + encoder_hidden * 2, decoder_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # PAD token
        
        # Initialize LSTM cell
        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        # Set forget gate bias to 1
        self.lstm.bias_hh.data[self.decoder_hidden:2*self.decoder_hidden].fill_(1.0)
        
        # Initialize output layers
        for m in self.fc_out.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        # Embedding: [batch, decoder_hidden]
        embedded = self.embedding(input_token)
        embedded = self.embedding_dropout(embedded)
        
        # Attention: [batch, encoder_hidden * 2]
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        
        # LSTM input: concatenate embedded and context
        lstm_input = torch.cat([embedded, context], dim=1)
        
        # LSTM step
        new_hidden, new_cell = self.lstm(lstm_input, (hidden, cell))
        new_hidden = self.dropout(new_hidden)
        
        # Output: concatenate hidden and context
        output_input = torch.cat([new_hidden, context], dim=1)
        output = self.fc_out(output_input)
        
        return output, new_hidden, new_cell, attention_weights


# ============================================================================
# Seq2Seq Model
# ============================================================================

class Seq2SeqAttention(nn.Module):
    """Complete Seq2Seq model with attention"""
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.5, drop_path=0.1):
        super(Seq2SeqAttention, self).__init__()
        
        self.num_classes = num_classes
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        
        self.cnn_encoder = CNNEncoder(drop_path=drop_path)  # With stochastic depth
        self.lstm_encoder = LSTMEncoder(512, encoder_hidden, dropout)
        self.decoder = LSTMDecoder(num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout)
        
        # Bridge layers: LSTM encoder is bidirectional with 2 layers
        # h_n shape: [num_layers * 2, batch, hidden] = [4, batch, hidden]
        # We'll use only the top layer's hidden states (last 2 directions)
        bridge_input_dim = encoder_hidden * 2  # Just top layer, both directions
        
        self.encoder_to_decoder_h = nn.Linear(bridge_input_dim, decoder_hidden)
        self.encoder_to_decoder_c = nn.Linear(bridge_input_dim, decoder_hidden)
        
        # Initialize bridge
        nn.init.xavier_uniform_(self.encoder_to_decoder_h.weight)
        nn.init.xavier_uniform_(self.encoder_to_decoder_c.weight)
        nn.init.zeros_(self.encoder_to_decoder_h.bias)
        nn.init.zeros_(self.encoder_to_decoder_c.bias)
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, attention_mask=None):
        """Forward pass with optional attention mask for padding"""
        batch_size = images.size(0)
        device = images.device
        
        # CNN encoding: [batch, seq_len, 512]
        cnn_features = self.cnn_encoder(images)
        
        # LSTM encoding: encoder_outputs [batch, seq_len, encoder_hidden*2]
        # h_n, c_n: [num_layers*2, batch, encoder_hidden] = [4, batch, hidden]
        encoder_outputs, (h_n, c_n) = self.lstm_encoder(cnn_features)
        
        # Use only top layer hidden states (last 2 are from top layer: forward and backward)
        # h_n[-2] = top layer forward, h_n[-1] = top layer backward
        top_h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, encoder_hidden * 2]
        top_c = torch.cat([c_n[-2], c_n[-1]], dim=1)  # [batch, encoder_hidden * 2]
        
        # Bridge to decoder initial states: [batch, decoder_hidden]
        decoder_hidden = torch.tanh(self.encoder_to_decoder_h(top_h))
        decoder_cell = torch.tanh(self.encoder_to_decoder_c(top_c))
        
        seq_len = encoder_outputs.size(1)
        
        # Use provided mask or create default (all True)
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Ensure mask matches encoder output sequence length
            mask = attention_mask.to(device)
            if mask.size(1) != seq_len:
                # Adjust mask size if CNN output differs
                mask = mask[:, :seq_len] if mask.size(1) > seq_len else F.pad(mask, (0, seq_len - mask.size(1)), value=False)
        
        if targets is not None:
            max_len = targets.size(1)
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            input_token = targets[:, 0]
            
            for t in range(1, max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = targets[:, t]
                else:
                    input_token = output.argmax(dim=1)
            
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            max_len = Config.MAX_OUTPUT_LENGTH
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            input_token = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            
            for t in range(max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                input_token = output.argmax(dim=1)
                
                if (input_token == 2).all():
                    break
            
            attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None
        
        return outputs, attention_weights


# ============================================================================
# Beam Search Decoder
# ============================================================================

class BeamSearchDecoder:
    """Beam search decoder for inference"""
    def __init__(self, model, beam_width=5):
        self.model = model
        self.beam_width = beam_width
        
    def decode(self, images, attention_mask=None):
        """Beam search with optional attention mask"""
        self.model.eval()
        batch_size = images.size(0)
        device = images.device
        
        predictions = []
        
        with torch.no_grad():
            cnn_features = self.model.cnn_encoder(images)
            encoder_outputs, (h_n, c_n) = self.model.lstm_encoder(cnn_features)
            
            seq_len = encoder_outputs.size(1)
            
            for b in range(batch_size):
                # Extract top layer hidden states for this sample
                # h_n[-2] = top layer forward, h_n[-1] = top layer backward
                top_h = torch.cat([h_n[-2, b:b+1], h_n[-1, b:b+1]], dim=1)  # [1, encoder_hidden * 2]
                top_c = torch.cat([c_n[-2, b:b+1], c_n[-1, b:b+1]], dim=1)  # [1, encoder_hidden * 2]
                
                decoder_hidden = torch.tanh(self.model.encoder_to_decoder_h(top_h))
                decoder_cell = torch.tanh(self.model.encoder_to_decoder_c(top_c))
                
                encoder_outputs_b = encoder_outputs[b:b+1]
                
                # Use attention mask if provided
                if attention_mask is not None:
                    mask_b = attention_mask[b:b+1].to(device)
                    # Adjust mask size if needed
                    if mask_b.size(1) != seq_len:
                        mask_b = mask_b[:, :seq_len] if mask_b.size(1) > seq_len else F.pad(mask_b, (0, seq_len - mask_b.size(1)), value=False)
                else:
                    mask_b = torch.ones(1, seq_len, dtype=torch.bool, device=device)
                
                # beams: (log_score, sequence, hidden, cell)
                # Start with log(1) = 0
                beams = [(0.0, [1], decoder_hidden, decoder_cell)]
                
                for _ in range(Config.MAX_OUTPUT_LENGTH):
                    candidates = []
                    
                    for log_score, sequence, hidden, cell in beams:
                        if sequence[-1] == 2:  # EOS token
                            candidates.append((log_score, sequence, hidden, cell))
                            continue
                        
                        input_token = torch.tensor([sequence[-1]], device=device)
                        output, new_hidden, new_cell, _ = self.model.decoder(
                            input_token, hidden, cell, encoder_outputs_b, mask_b
                        )
                        
                        log_probs = F.log_softmax(output, dim=1)
                        top_log_probs, top_indices = log_probs.topk(self.beam_width)
                        
                        for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                            # ADD log probabilities (not multiply)
                            new_log_score = log_score + log_prob.item()
                            new_sequence = sequence + [idx.item()]
                            candidates.append((new_log_score, new_sequence, new_hidden, new_cell))
                    
                    # Sort by log score (higher is better)
                    beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.beam_width]
                    
                    if all(seq[-1] == 2 for _, seq, _, _ in beams):
                        break
                
                best_sequence = beams[0][1]
                predictions.append(best_sequence)
        
        return predictions
