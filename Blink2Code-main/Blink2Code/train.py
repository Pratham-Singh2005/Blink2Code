import torch
import torch.optim as optim
import torch.nn as nn
from preprocess import load_and_preprocess
from cnn_model import BlinkCNN

print("🚀 Starting training script...")

# Load preprocessed data
print("📂 Loading preprocessed data...")
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_dim = load_and_preprocess()
print(f"✅ Data loaded! Training samples: {X_train_tensor.shape[0]}, Features: {input_dim}")

# Initialize model
print("🧠 Initializing model...")
model = BlinkCNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
print("🚀 Starting training loop...")

for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"📊 Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("✅ Training complete!")

# Save model
torch.save(model.state_dict(), "models/blink_cnn.pth")
print("💾 Model saved as 'models/blink_cnn.pth'")
