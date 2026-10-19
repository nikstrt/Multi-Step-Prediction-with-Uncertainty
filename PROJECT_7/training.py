# =====================================================================
# Core Model Training & Loss Function Engine
# =====================================================================
import torch
import torch.nn as nn
import torch.optim as optim

def gaussian_nll_loss(y_true, mu, var):
    return torch.mean(0.5 * torch.log(var) + 0.5 * (y_true - mu) ** 2 / var)

def train_model(model, train_loader, val_loader, optimizer, epochs=30, mode="deterministic", device="cpu", weights_path="best_model.pt", patience=5):
    model.to(device)
    mse_criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5) 
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    print(f"\n[+] Starting {mode.upper()} training loop | device: {device}")
    print("--------------------------------------------------------------------")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for X, A, Y in train_loader:
            X, A, Y = X.to(device), A.to(device), Y.to(device)
            optimizer.zero_grad()
            
            if mode == "deterministic":
                loss = mse_criterion(model(X, A), Y)
            else:
                mu, var = model(X, A)
                loss = gaussian_nll_loss(Y, mu, var)
                
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, A, Y in val_loader:
                X, A, Y = X.to(device), A.to(device), Y.to(device)
                if mode == "deterministic":
                    loss = mse_criterion(model(X, A), Y)
                else:
                    mu, var = model(X, A)
                    loss = gaussian_nll_loss(Y, mu, var)
                running_val_loss += loss.item() * X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break
                
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("--------------------------------------------------------------------")
    print(f"[+] Restored best parameter configuration. Val Loss Floor: {best_val_loss:.6f}")
    return model