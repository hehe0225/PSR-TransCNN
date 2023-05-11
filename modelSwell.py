import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_classes=2, init_weights=False):
        super(Model, self).__init__()
        self.features = nn.Sequential(   
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=36, nhead=6) 

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(6624, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes),
        )
        
        
        if init_weights:
            self._initialize_weights()
            
  
    def forward(self, x):
        x=self.encoder_layer(x)
        x=x.unsqueeze(1) 
        x = self.features(x) 
        x = torch.flatten(x, start_dim=1) 
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
