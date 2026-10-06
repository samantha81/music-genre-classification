import torch
import torch.nn as nn

class FrozenCNNLSTM(nn.Module):
    def __init__(self, cnn_backbone, num_classes=10, lstm_hidden=48, dropout=0.6):
        super().__init__()
        self.features = cnn_backbone.features
        self.gap = cnn_backbone.gap

        # Freeze the pre-trained weights
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.gap.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0.0
        )
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, segments):
        B, N, C, H, W = segments.shape
        segments = segments.view(B * N, C, H, W)
        
        with torch.no_grad():
            features = self.features(segments)  # (B*N, 256, 8, 8)
            features = self.gap(features)        # (B*N, 256, 1, 1)
            
        features = features.squeeze(-1).squeeze(-1)  # (B*N, 256)
        features = features.view(B, N, -1)           # (B, N, 256)
        lstm_out, _ = self.lstm(features)            # (B, N, hidden*2)
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, N, 1)
        attended = (lstm_out * attn_weights).sum(dim=1)                # (B, hidden*2)
        return self.classifier(attended)
