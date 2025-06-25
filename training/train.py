# This script defines the training routine for Deep SVDD using a given dataset and center.
# It performs standard epoch-based training with Adam optimizer and saves the final model checkpoint.

import torch
import torch.optim as optim
from tqdm import tqdm
import os

def train_svdd(model, train_loader, device, center_c, num_epochs=20, lr=1e-4, weight_decay=1e-6, save_path='checkpoints/deep_svdd.pth'):
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Move model to computation device and set hypersphere center
    model.to(device)
    model.c = center_c
    model.train()  # Enable training mode

    # Initialize Adam optimizer with optional weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop over epochs
    for epoch in range(1, num_epochs + 1):
        total_loss = 0  # Accumulate loss to compute epoch average
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)  # Progress bar

        for inputs, _ in loop:
            inputs = inputs.to(device)
            optimizer.zero_grad()  # Clear previous gradients

            outputs = model(inputs)  # Forward pass
            loss, _ = model.compute_loss(outputs)  # Compute SVDD loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer update

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())  # Show current batch loss in progress bar

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    # Save the trained model and relevant metadata to disk
    torch.save({
        'model_state_dict': model.state_dict(),
        'center_c': model.c,
        'embedding_dim': model.embedding_dim
    }, save_path)

    print(f"Model saved to {save_path}")