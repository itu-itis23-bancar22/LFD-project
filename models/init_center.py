# This function initializes the center 'c' of the hypersphere in Deep SVDD.
# It computes the mean of the latent representations from the encoder over the training set.

import torch

@torch.no_grad()  # Disable gradient tracking for efficiency
def init_center_c(model, train_loader, device):
    n_samples = 0
    c = torch.zeros(model.embedding_dim, device=device)  # Initialize center vector

    model.eval()  # Set model to evaluation mode
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)  # Get latent representations
        n_samples += outputs.shape[0]  # Update sample count
        c += torch.sum(outputs, dim=0)  # Accumulate latent vectors

    c /= n_samples  # Compute average (center of hypersphere)

    # Prevent any center dimension from being too close to zero (to avoid division errors later)
    c[(abs(c) < 1e-6) & (c < 0)] = -1e-6
    c[(abs(c) < 1e-6) & (c > 0)] = 1e-6

    return c