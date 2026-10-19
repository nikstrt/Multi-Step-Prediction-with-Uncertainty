# =====================================================================
# Terminal Matrix & Telemetry Output Formatting Tools
# =====================================================================
import torch
import os

def print_diagnostic_header(model_mode, train_mode, weights_path):
    print("\n" + "="*80)
    print(f"!!! CRITICAL DIAGNOSTIC CHECK !!!")
    print(f"TERMINAL DISK PATH FOR WEIGHTS IS: {os.path.abspath(weights_path)}")
    print(f"TRAIN MODE STATUS                  : {train_mode}")
    print(f"MODEL TARGET SPECIFICATION         : {model_mode.upper()}")
    print("="*80 + "\n")

def print_deterministic_error_matrix(horizon, rollout_predictions, ground_truth_path):
    print("\n====================================================================")
    print("  QUESTION VII: COMPOUNDING DETERMINISTIC PREDICTION ERROR TRACKING  ")
    print("====================================================================")
    print(f" Look-Ahead Step (k)  |  Cumulative Rollout MSE Loss")
    print("--------------------------------------------------------------------")
    for k in range(horizon):
        step_mse = torch.mean((rollout_predictions[k] - ground_truth_path[k]) ** 2).item()
        print(f"     Step k = {k+1:02d}       |         {step_mse:.6f}")
    print("====================================================================\n")

def print_probabilistic_horizon_matrix(horizon, rollout_means, rollout_vars, ground_truth_path):
    print("\n====================================================================")
    print("  QUESTIONS IX, X, XI: CALIBRATED PROBABILISTIC HORIZON ROLLOUTS     ")
    print("====================================================================")
    print(f" Look-Ahead Step (k)  |  Empirical Error (MSE)  |  Predicted Uncertainty (Mean Var)")
    print("--------------------------------------------------------------------")
    for k in range(horizon):
        actual_error = torch.mean((rollout_means[k] - ground_truth_path[k]) ** 2).item()
        predicted_uncertainty = torch.mean(rollout_vars[k]).item()
        print(f"     Step k = {k+1:02d}       |        {actual_error:.6f}         |             {predicted_uncertainty:.6f}")
    print("====================================================================\n")

def print_part_c_comparison_matrix(horizon, det_means, det_vars, mc_means, mc_vars, mm_means, mm_vars, ground_truth_path):
    print("\n====================================================================================================")
    print("  PART C (xii/xiv/xv): SYSTEM IDENTIFICATION PERFORMANCE AND VARIANCE EXPANSION MATRIX")
    print("====================================================================================================")
    print(" Horizon |   Deterministic Mean Engine   |      Monte Carlo Ensemble     |    Analytical Moment Match   ")
    print(" Step (k)| Tracking MSE | Predicted Var  | Tracking MSE | Empirical Var  | Tracking MSE | Analytical Var")
    print("----------------------------------------------------------------------------------------------------")
    for k in range(horizon):
        gt = ground_truth_path[k]
        print(f"  k = {k+1:02d}  |   "
              f"{torch.mean((det_means[k]-gt)**2).item():.6f}   |    {torch.mean(det_vars[k]).item():.6f}    |   "
              f"{torch.mean((mc_means[k]-gt)**2).item():.6f}   |    {torch.mean(mc_vars[k]).item():.6f}    |   "
              f"{torch.mean((mm_means[k]-gt)**2).item():.6f}   |    {torch.mean(mm_vars[k]).item():.6f}")
    print("====================================================================================================\n")