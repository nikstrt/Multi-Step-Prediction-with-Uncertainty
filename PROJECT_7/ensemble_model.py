# =====================================================================
# Deep Ensemble Workspace Module
# =====================================================================
import os
import torch
import torch.nn as nn
from models import ProbabilisticPredictor

class EnsemblePredictor(nn.Module):
    """
    Polymorphic Deep Ensemble wrapper containing multiple independent 
    Probabilistic Neural Networks. Automatically aggregates distribution parameters 
    via the Law of Total Variance to capture both Aleatoric and Epistemic uncertainty.
    """
    def __init__(self, num_models=5, state_dim=4, action_dim=2,
                 embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.num_models = num_models
        # Instantiates separate probabilistic predictors with independent weights
        self.models = nn.ModuleList([
            ProbabilisticPredictor(state_dim, action_dim, embedding_dim, hidden_dim)
            for _ in range(num_models)
        ])

    def forward(self, states, actions):
        """
        Forward Pass. Evaluates all ensemble members and computes the mixture components.
        
        Returns:
            mu_ensemble (Tensor): Consensus mean path [Batch, State_Dim]
            total_var (Tensor): Total variance corridor [Batch, State_Dim]
        """
        mus, vars_ = [], []
        for model in self.models:
            mu_m, var_m = model(states, actions)
            mus.append(mu_m)
            vars_.append(var_m)

        # Shape formatting: [Num_Models, Batch_Size, State_Dim]
        mus   = torch.stack(mus, dim=0)      
        vars_ = torch.stack(vars_, dim=0)    

        # 1. Consensus Mean (Expectation of Means)
        mu_ensemble = mus.mean(dim=0)                          
        
        # 2. Total Variance = Expected Aleatoric Noise + Epistemic Trajectory Dispersion
        # mus.var captures model disagreement as inputs drift into out-of-distribution regions
        total_var = vars_.mean(dim=0) + mus.var(dim=0, unbiased=False) + 1e-6

        return mu_ensemble, total_var


def gaussian_nll_loss(y_true, mu, var):
    """Computes Gaussian Negative Log-Likelihood Loss."""
    return torch.mean(0.5 * torch.log(var) + 0.5 * (y_true - mu) ** 2 / var)




import torch.optim as optim
def train_ensemble(ensemble, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu", weights_path="ensemble_best.pt", patience=5):
    ensemble.to(device)
    
    # Same exact optimizer and Scheduler as the Single model for fair comparison
    optimizer = optim.Adam(ensemble.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[+] Starting ENSEMBLE FAIR training loop | device: {device}")
    print("--------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        ensemble.train()
        running_train_loss = 0.0
        
        for X, A, Y in train_loader:
            X, A, Y = X.to(device), A.to(device), Y.to(device)
            optimizer.zero_grad()
            
            # Forward pass for allt of the ensemble's models
            # Here it is supposed that the ensemble's forward returns the sum or the average of the losses
            # if it returns lists: 
            losses = []
            for model in ensemble.models:
                mu, var = model(X, A)
                losses.append(gaussian_nll_loss(Y, mu, var))
            
            # Ensemble's Total Loss
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation Loop
        ensemble.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, A, Y in val_loader:
                X, A, Y = X.to(device), A.to(device), Y.to(device)
                
                losses = []
                for model in ensemble.models:
                    mu, var = model(X, A)
                    losses.append(gaussian_nll_loss(Y, mu, var))
                loss = sum(losses) / len(losses)
                running_val_loss += loss.item() * X.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # LR Scheduler as in the Single model
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Early Stopping as in the Single model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(ensemble.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    # return the best weights
    ensemble.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("--------------------------------------------------------------------")
    print(f"[+] Restored best ENSEMBLE parameter configuration. Val Loss Floor: {best_val_loss:.6f}")
    
    return ensemble