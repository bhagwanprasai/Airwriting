"""
Model Architecture Module
Contains the CRNN model and related components
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ResidualBlock(nn.Module):
    """Residual block with DropPath"""
    def __init__(self, in_c, out_c, stride=1, drop_path=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), 
                nn.BatchNorm2d(out_c)
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CRNN(nn.Module):
    """CRNN with Stochastic Depth"""
    def __init__(self, num_chars, hidden_size=256, dropout=0.5, drop_path=0.1):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, drop_path=drop_path), 
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, drop_path=drop_path), 
            ResidualBlock(256, 256, drop_path=drop_path), 
            nn.MaxPool2d((2, 1))
        )
        
        self.rnn = nn.LSTM(
            input_size=256 * 8, 
            hidden_size=hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_chars)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)


class LabelSmoothingCTCLoss(nn.Module):
    """CTC Loss with Label Smoothing"""
    def __init__(self, blank=0, smoothing=0.1, zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        if self.smoothing > 0:
            kl_loss = -log_probs.mean()
            loss = (1 - self.smoothing) * loss + self.smoothing * kl_loss
        
        return loss.mean()
