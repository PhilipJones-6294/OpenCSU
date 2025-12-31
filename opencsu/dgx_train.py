"""
dgx_train.py
Trains the Neural Surrogate on the generated data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- THE MODEL ARCHITECTURE ---
class CSUSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 parameters [mu_T, gamma_M, drug_eff]
        # Output: 64x64 Image
        
        self.fc = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * 4 * 128), # Reshape base
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output 0-1 (Image brightness)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder(x)
        return x.squeeze(1)

# --- TRAINING LOOP ---
def train_model():
    print("🚀 Loading Data...")
    data = torch.load('csu_training_data.pt')
    dataset = TensorDataset(data['X'], data['Y'])
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = CSUSurrogate().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Pixel-wise prediction error
    
    print("🧠 Training Neural Surrogate...")
    model.train()
    
    for epoch in range(50): # 50 Epochs should be enough
        total_loss = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.6f}")
        
    # Save the 'Brain'
    torch.save(model.state_dict(), 'csu_surrogate_weights.pt')
    print("✅ Model Trained. Saved to csu_surrogate_weights.pt")

if __name__ == "__main__":
    train_model()