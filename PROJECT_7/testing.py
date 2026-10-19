# =====================================================================
# Independent Test Set Evaluation Engine
# =====================================================================
import torch
import torch.nn as nn
import numpy as np

def evaluate_model(model, test_loader, model_mode="probabilistic", device="cpu"):
    model.eval().to(device)
    mse_criterion = nn.MSELoss()
    running_test_loss = 0.0
    all_predictions, all_targets = [], []
    
    print("\n[+] Launching Independent One-Step Prediction Evaluation (Directive v)...")
    print("--------------------------------------------------------------------")
    
    with torch.no_grad():
        for X, A, Y in test_loader:
            X, A, Y = X.to(device), A.to(device), Y.to(device)
            pred = model(X, A)[0] if model_mode == "probabilistic" else model(X, A) 
            
            loss = mse_criterion(pred, Y)
            running_test_loss += loss.item() * X.size(0)
            all_predictions.append(pred.cpu())
            all_targets.append(Y.cpu())
            
    total_test_mse = running_test_loss / len(test_loader.dataset)
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    feature_mses = np.mean((all_predictions - all_targets) ** 2, axis=0)
    
    print(f"GLOBAL ONE-STEP TEST {model_mode.upper()} MSE: {total_test_mse:.6f}")
    print("--------------------------------------------------------------------")
    print("FEATURE-BY-FEATURE MSE ACCURACY BREAKDOWN:")
    for name, f_mse in zip(names, feature_mses):
        print(f"  -> {name:<22}: {f_mse:.6f}")
    print("====================================================================\n")
    return total_test_mse