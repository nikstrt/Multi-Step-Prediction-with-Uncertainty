
# # =====================================================================
# #  Universal Modular Orchestration & Planning Flow
# # =====================================================================
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch
# import torch.optim as optim
# import numpy as np

# from dataset_utils import CartPoleDatasetManager
# from models import DeterministicPredictor, ProbabilisticPredictor
# from propagation import (run_deterministic_rollout, 
#                                run_probabilistic_rollout,
#                                run_deterministic_mean_propagation, 
#                                run_monte_carlo_propagation, 
#                                run_moment_matching_propagation)

# from training import train_model
# from testing import evaluate_model

# from plotting import (plot_trajectory_comparison, 
#                             plot_monte_carlo_fan_chart, 
#                             plot_global_calibration,
#                             plot_global_error_totality,       
#                             plot_global_variance_totality,
#                             plot_pure_deterministic_totality)     

# from reporting import (print_diagnostic_header, print_deterministic_error_matrix, 
#                              print_probabilistic_horizon_matrix, print_part_c_comparison_matrix)
# from ensemble_model import EnsemblePredictor, train_ensemble

# # ── CENTRALIZED CONTROL KNOBS ─────────────────────────────────────────
# TRAIN_MODE           = True             # True = Optimize from scratch | False = Evaluation Mode
# MODEL_MODE           = "deterministic"   # Options: "deterministic" | "probabilistic"
# PROB_TYPE            = "ensemble"        # Options: "single" | "ensemble" (Ignored if deterministic)
# TARGET_TRAJECTORY_ID = 250               # Hardcoded target (will be dynamically corrected if unsafe)
# HORIZON_LIMIT        = 20
# SEED                 = 42

# BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def set_global_seed(seed: int) -> None:
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): 
#         torch.cuda.manual_seed_all(seed)

# def main():
#     # Make TARGET_TRAJECTORY_ID modifiable within the global scope if it fails safety checks
#     global TARGET_TRAJECTORY_ID 
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     set_global_seed(SEED)
    
#     # EVOLVED WEIGHT ISOLATION STRATEGY
#     if MODEL_MODE == "deterministic":
#         weights_name = "deterministic_best.pt"
#     else:
#         weights_name = f"probabilistic_{PROB_TYPE}_best.pt"
#     weights_path = os.path.join(OUTPUT_DIR, weights_name)
    
#     print_diagnostic_header(f"{MODEL_MODE.upper()} ({PROB_TYPE.upper() if MODEL_MODE=='probabilistic' else 'N/A'})", TRAIN_MODE, weights_path)

#     data_manager = CartPoleDatasetManager(file_path="cartpole_stochastic_transitions.csv")
#     train_loader, val_loader, test_loader = data_manager.preprocess_and_split(batch_size=64, random_seed=SEED)
    
#   # ── TARGET TRAJECTORY SELECTION  ───────────
#     test_ids = data_manager.test_ids
    
#     # Clean up the IDs (convert to int) and SORT them in ascending order
#     clean_test_ids = sorted([int(tid) for tid in test_ids])
    
#     print(f"\n[+] Available Test Set Trajectory IDs ({len(clean_test_ids)} total):")
#     print(f"    {clean_test_ids}")

#     if TARGET_TRAJECTORY_ID not in clean_test_ids:
#         print(f"\n[!] WARNING: Requested TARGET_TRAJECTORY_ID {TARGET_TRAJECTORY_ID} is NOT in the Test Set (it may be in Train/Val).")
#         TARGET_TRAJECTORY_ID = int(np.random.choice(clean_test_ids))
#         print(f"[+] Automatically randomly reassigned TARGET_TRAJECTORY_ID to {TARGET_TRAJECTORY_ID} to prevent data leakage.\n")
#     else:
#         print(f"\n[+] TARGET_TRAJECTORY_ID {TARGET_TRAJECTORY_ID} is valid and strictly isolated in the Test Set.\n")

#     # ── DYNAMIC INSTANTIATION LAYER ────────────────────────────────
#     model_kwargs = dict(state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64)
    
#     if MODEL_MODE == "deterministic": 
#         model = DeterministicPredictor(**model_kwargs)
#     elif MODEL_MODE == "probabilistic":
#         if PROB_TYPE == "single":
#             model = ProbabilisticPredictor(**model_kwargs)
#         elif PROB_TYPE == "ensemble":
#             ensemble_kwargs = dict(num_models=5, state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64)
#             model = EnsemblePredictor(**ensemble_kwargs)
#         else:
#             raise ValueError(f"[-] Unknown PROB_TYPE specification: '{PROB_TYPE}'")
#     else:
#         raise ValueError(f"[-] Unknown MODEL_MODE choice: '{MODEL_MODE}'")
        
#     # ── DYNAMIC OPTIMIZATION ROUTER ────────────────────────
#     if TRAIN_MODE:
#         if os.path.exists(weights_path): 
#             os.remove(weights_path)
            
#         if MODEL_MODE == "deterministic" or (MODEL_MODE == "probabilistic" and PROB_TYPE == "single"):
#             optimizer = optim.Adam(model.parameters(), lr=1e-3)
#             trained_model = train_model(
#                 model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
#                 epochs=50, mode=MODEL_MODE, device=str(device), weights_path=weights_path, patience=5
#             )
#         else:
#             trained_model = train_ensemble(
#                 ensemble=model, train_loader=train_loader, val_loader=val_loader, 
#                 epochs=50, lr=1e-3, device=str(device), weights_path=weights_path
#             )
            
#         evaluate_model(model=trained_model, test_loader=test_loader, model_mode=MODEL_MODE, device=device)
#         torch.save(trained_model.state_dict(), weights_path)
#     else:
#         if not os.path.exists(weights_path):
#             raise FileNotFoundError(f"[-] Weights missing at '{weights_path}'. Switch TRAIN_MODE=True to train first!")
#         model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
#         print(f"[+] Successfully loaded isolated weights from '{weights_path}'")
#         trained_model = model

   
#     suffix = "_det" if MODEL_MODE == "deterministic" else f"_{'en' if PROB_TYPE == 'ensemble' else 'single'}"
    
#     # ── SINGLE TRAJECTORY EVALUATOR ────────────────────────────────
#     print(f"[+] Extracting independent testing sequence for Trajectory {TARGET_TRAJECTORY_ID}...")
#     states, actions, targets = data_manager.get_unbroken_test_trajectory(TARGET_TRAJECTORY_ID)
#     horizon = min(HORIZON_LIMIT, len(actions))
#     initial_state, action_sequence, ground_truth = states[0].unsqueeze(0), actions[:horizon], targets[:horizon]
    
#     # =================================================================
#     # DETERMINISTIC EVALUATION FLOW
#     # =================================================================
#     if MODEL_MODE == "deterministic":
#         rollout_predictions = run_deterministic_rollout(trained_model, initial_state, action_sequence, device)
#         print_deterministic_error_matrix(horizon, rollout_predictions, ground_truth)
        
       
#         plot_trajectory_comparison(
#             ground_truth, 
#             rollout_predictions, 
#             data_manager.means, 
#             data_manager.stds, 
#             os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}{suffix}.jpg"), 
#             title_label="Deterministic Point Prediction"
#         )
#         print(f"\n[✓] Deterministic Trajectory Comparison Chart saved successfully to: {OUTPUT_DIR}")

#         # STANDALONE DETERMINISTIC TOTALITY LOOP
#         print(f"\n[+] Launching Deterministic Totality Evaluation across 50 Test trajectories...")
        
#         true_paths = []
#         pred_paths = []
        
#         # Take up to 50 trajectories strictly from the test set
#         totality_pool = clean_test_ids[:50] 
        
#         for t_id in totality_pool:
#             try:
#                 # Extract the trajectory
#                 t_states, t_actions, t_targets = data_manager.get_unbroken_test_trajectory(t_id)
#                 t_horizon = min(HORIZON_LIMIT, len(t_actions))
#                 t_init_state = t_states[0].unsqueeze(0)
#                 t_act_seq = t_actions[:t_horizon]
#                 t_gt_path = t_targets[:t_horizon]
                
#                 # Run the model
#                 t_det_preds = run_deterministic_rollout(trained_model, t_init_state, t_act_seq, device)
                
#                 # Store the results safely
#                 true_paths.append(t_gt_path)
#                 pred_paths.append(t_det_preds)
               
#             except Exception as e:
#                 # Silently skip trajectories that are too short or have issues
#                 continue
                
#         # Generate the Totality Plot
#         if len(true_paths) > 0:
#             output_file = os.path.join(OUTPUT_DIR, f"TOTALITY_global_error{suffix}.jpg")
#             plot_pure_deterministic_totality(true_paths, pred_paths, output_file)
#         else:
#             print("[-] Error: No valid trajectories found for Totality evaluation.")

#     # =================================================================
#     # PROBABILISTIC EVALUATION FLOW
#     # =================================================================
#     else:
#         rollout_means, rollout_vars = run_probabilistic_rollout(trained_model, initial_state, action_sequence, device)
#         print_probabilistic_horizon_matrix(horizon, rollout_means, rollout_vars, ground_truth)
        
    
#         plot_trajectory_comparison(ground_truth, rollout_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_prob_base{suffix}.jpg"), f"Probabilistic Baseline ({PROB_TYPE.upper()})")

#         det_means, det_vars = run_deterministic_mean_propagation(trained_model, initial_state, action_sequence, device)
#         mc_means, mc_vars, mc_particles = run_monte_carlo_propagation(trained_model, initial_state, action_sequence, 200, device)
#         mm_means, mm_vars = run_moment_matching_propagation(trained_model, initial_state, action_sequence, device)
        
#         print_part_c_comparison_matrix(horizon, det_means, det_vars, mc_means, mc_vars, mm_means, mm_vars, ground_truth)
        
        
#         plot_trajectory_comparison(ground_truth, det_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_det_prop{suffix}.jpg"), "Deterministic Mean Propagation", det_vars)
#         plot_trajectory_comparison(ground_truth, mc_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mc_prop{suffix}.jpg"), "Monte Carlo Propagation", mc_vars)
#         plot_trajectory_comparison(ground_truth, mm_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mm_prop{suffix}.jpg"), "Moment Matching Propagation", mm_vars)
#         plot_monte_carlo_fan_chart(ground_truth, mc_means, mc_particles, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mc_fan{suffix}.jpg"))
        
#         # ── PROJECT TOTALITY AUTOMATED EVALUATION LOOP ─────────────────
#         print(f"\n[+] Launching Project Totality Evaluation across the dataset pool...")
#         all_unique_trajectory_ids = sorted(data_manager.df["trajectory_id"].unique())
#         totality_pool = test_ids[:50] 
        
#         error_tracking_pool = []
#         variance_tracking_pool = []
        
#         for t_id in totality_pool:
#             try:
#                 t_states, t_actions, t_targets = data_manager.get_unbroken_test_trajectory(t_id)
#                 t_horizon = min(HORIZON_LIMIT, len(t_actions))
#                 t_init_state = t_states[0].unsqueeze(0)
#                 t_act_seq = t_actions[:t_horizon]
#                 t_gt_path = t_targets[:t_horizon]
                
#                 t_det_m, t_det_v = run_deterministic_mean_propagation(trained_model, t_init_state, t_act_seq, device)
#                 t_mc_m, t_mc_v, _ = run_monte_carlo_propagation(trained_model, t_init_state, t_act_seq, 100, device)
#                 t_mm_m, t_mm_v = run_moment_matching_propagation(trained_model, t_init_state, t_act_seq, device)
                
#                 error_tracking_pool.append({'true': t_gt_path, 'det': t_det_m, 'mc': t_mc_m, 'mm': t_mm_m})
#                 variance_tracking_pool.append({'det_var': t_det_v, 'mc_var': t_mc_v, 'mm_var': t_mm_v})
#             except Exception:
#                 continue 
                
#         print(f"[+] Compiling totality statistics over {len(error_tracking_pool)} valid trajectories...")
        
        
#         plot_global_error_totality(error_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_error_vs_horizon{suffix}.jpg"))
#         plot_global_variance_totality(variance_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_variance_vs_time{suffix}.jpg"))
#         plot_global_calibration(error_tracking_pool, variance_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_calibration{suffix}.jpg"))
        
#         print(f"\n[✓] All probabilistic global totality artifacts saved to: {OUTPUT_DIR}")

# if __name__ == "__main__":
#     main()




# =====================================================================
# Universal Modular Orchestration & Planning Flow
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
import numpy as np

from dataset_utils import CartPoleDatasetManager
from models import DeterministicPredictor, ProbabilisticPredictor
from propagation import (run_deterministic_rollout, 
                               run_probabilistic_rollout,
                               run_deterministic_mean_propagation, 
                               run_monte_carlo_propagation, 
                               run_moment_matching_propagation)

from training import train_model
from testing import evaluate_model

from plotting import (plot_trajectory_comparison, 
                            plot_monte_carlo_fan_chart, 
                            plot_global_calibration,
                            plot_global_error_totality,       
                            plot_global_variance_totality,
                            plot_pure_deterministic_totality)     

from reporting import (print_diagnostic_header, print_deterministic_error_matrix, 
                             print_probabilistic_horizon_matrix, print_part_c_comparison_matrix)
from ensemble_model import EnsemblePredictor, train_ensemble

# ── CENTRALIZED CONTROL KNOBS ─────────────────────────────────────────
TRAIN_MODE           = True             # True = Optimize from scratch | False = Evaluation Mode
MODEL_MODE           = "deterministic"   # Options: "deterministic" | "probabilistic"
DET_EVAL_MODE        = "one-step"        # Options: "one-step" | "multi-step" (Ignored if probabilistic)
PROB_TYPE            = "ensemble"        # Options: "single" | "ensemble" (Ignored if deterministic)
TARGET_TRAJECTORY_ID = 250               # Hardcoded target (will be dynamically corrected if unsafe)
HORIZON_LIMIT        = 20
SEED                 = 42

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def main():
    # Make TARGET_TRAJECTORY_ID modifiable within the global scope if it fails safety checks
    global TARGET_TRAJECTORY_ID 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    
    # Weight isolation strategy
    if MODEL_MODE == "deterministic":
        weights_name = "deterministic_best.pt"
    else:
        weights_name = f"probabilistic_{PROB_TYPE}_best.pt"
    weights_path = os.path.join(OUTPUT_DIR, weights_name)
    
    print_diagnostic_header(f"{MODEL_MODE.upper()} ({PROB_TYPE.upper() if MODEL_MODE=='probabilistic' else 'N/A'})", TRAIN_MODE, weights_path)

    data_manager = CartPoleDatasetManager(file_path="cartpole_stochastic_transitions.csv")
    train_loader, val_loader, test_loader = data_manager.preprocess_and_split(batch_size=64, random_seed=SEED)
    
    # ── TARGET TRAJECTORY SELECTION (LEAK-PROOF) ───────────
    test_ids = data_manager.test_ids
    clean_test_ids = sorted([int(tid) for tid in test_ids])
    
    print(f"\n[+] Available Test Set Trajectory IDs ({len(clean_test_ids)} total):")
    print(f"    {clean_test_ids}")

    if TARGET_TRAJECTORY_ID not in clean_test_ids:
        print(f"\n[!] WARNING: Requested TARGET_TRAJECTORY_ID {TARGET_TRAJECTORY_ID} is NOT in the Test Set (it may be in Train/Val).")
        TARGET_TRAJECTORY_ID = int(np.random.choice(clean_test_ids))
        print(f"[+] Automatically randomly reassigned TARGET_TRAJECTORY_ID to {TARGET_TRAJECTORY_ID} to prevent data leakage.\n")
    else:
        print(f"\n[+] TARGET_TRAJECTORY_ID {TARGET_TRAJECTORY_ID} is valid and strictly isolated in the Test Set.\n")

    # ── DYNAMIC INSTANTIATION LAYER ────────────────────────────────
    model_kwargs = dict(state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64)
    
    if MODEL_MODE == "deterministic": 
        model = DeterministicPredictor(**model_kwargs)
    elif MODEL_MODE == "probabilistic":
        if PROB_TYPE == "single":
            model = ProbabilisticPredictor(**model_kwargs)
        elif PROB_TYPE == "ensemble":
            ensemble_kwargs = dict(num_models=5, state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64)
            model = EnsemblePredictor(**ensemble_kwargs)
        else:
            raise ValueError(f"[-] Unknown PROB_TYPE specification: '{PROB_TYPE}'")
    else:
        raise ValueError(f"[-] Unknown MODEL_MODE choice: '{MODEL_MODE}'")
        
    # ── DYNAMIC OPTIMIZATION ROUTER ────────────────────────
    if TRAIN_MODE:
        if os.path.exists(weights_path): 
            os.remove(weights_path)
            
        if MODEL_MODE == "deterministic" or (MODEL_MODE == "probabilistic" and PROB_TYPE == "single"):
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            trained_model = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
                epochs=50, mode=MODEL_MODE, device=str(device), weights_path=weights_path, patience=5
            )
        else:
            trained_model = train_ensemble(
                ensemble=model, train_loader=train_loader, val_loader=val_loader, 
                epochs=50, lr=1e-3, device=str(device), weights_path=weights_path
            )
            
        evaluate_model(model=trained_model, test_loader=test_loader, model_mode=MODEL_MODE, device=device)
        torch.save(trained_model.state_dict(), weights_path)
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[-] Weights missing at '{weights_path}'. Switch TRAIN_MODE=True to train first!")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[+] Successfully loaded isolated weights from '{weights_path}'")
        trained_model = model

    # Set suffix dynamically based on model mode and evaluation settings
    if MODEL_MODE == "deterministic":
        suffix = "_det_onestep" if DET_EVAL_MODE == "one-step" else "_det_multistep"
    else:
        suffix = f"_{'en' if PROB_TYPE == 'ensemble' else 'single'}"
    
    # ── SINGLE TRAJECTORY EVALUATOR ────────────────────────────────
    print(f"[+] Extracting independent testing sequence for Trajectory {TARGET_TRAJECTORY_ID}...")
    states, actions, targets = data_manager.get_unbroken_test_trajectory(TARGET_TRAJECTORY_ID)
    horizon = min(HORIZON_LIMIT, len(actions))
    initial_state, action_sequence, ground_truth = states[0].unsqueeze(0), actions[:horizon], targets[:horizon]
    
    # =================================================================
    # DETERMINISTIC EVALUATION FLOW
    # =================================================================
    if MODEL_MODE == "deterministic":
        
        if DET_EVAL_MODE == "one-step":
            print(f"\n[+] Executing ONE-STEP Deterministic Evaluation for Trajectory {TARGET_TRAJECTORY_ID}...")
            
            # Extract true states to act as inputs (Teacher Forcing)
            input_states = states[:horizon].to(device)
            input_actions = action_sequence.to(device)
            
            trained_model.eval()
            with torch.no_grad():
                one_step_predictions = trained_model(input_states, input_actions).cpu()
            
            # Print One-Step Error Matrix
            print("\n====================================================================")
            print("  ONE-STEP DETERMINISTIC PREDICTION ERROR TRACKING (TEACHER FORCING) ")
            print("====================================================================")
            print(f" Look-Ahead Step (k)  |  One-Step MSE Loss")
            print("--------------------------------------------------------------------")
            for k in range(horizon):
                step_mse = torch.mean((one_step_predictions[k] - ground_truth[k]) ** 2).item()
                print(f"     Step k = {k+1:02d}       |         {step_mse:.6f}")
            print("====================================================================\n")
            
            # Plot One-Step Validation
            plot_trajectory_comparison(
                ground_truth, 
                one_step_predictions, 
                data_manager.means, 
                data_manager.stds, 
                os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}{suffix}.jpg"), 
                title_label="Deterministic One-Step Prediction"
            )
            print(f"\n[✓] Deterministic One-Step Trajectory Chart saved successfully to: {OUTPUT_DIR}")

        elif DET_EVAL_MODE == "multi-step":
            print(f"\n[+] Executing MULTI-STEP Autoregressive Evaluation for Trajectory {TARGET_TRAJECTORY_ID}...")
            
            rollout_predictions = run_deterministic_rollout(trained_model, initial_state, action_sequence, device)
            print_deterministic_error_matrix(horizon, rollout_predictions, ground_truth)
            
            plot_trajectory_comparison(
                ground_truth, 
                rollout_predictions, 
                data_manager.means, 
                data_manager.stds, 
                os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}{suffix}.jpg"), 
                title_label="Deterministic Multi-Step Prediction"
            )
            print(f"\n[✓] Deterministic Multi-Step Trajectory Chart saved successfully to: {OUTPUT_DIR}")

            # STANDALONE DETERMINISTIC TOTALITY LOOP
            print(f"\n[+] Launching Deterministic Totality Evaluation across 50 Test trajectories...")
            
            true_paths = []
            pred_paths = []
            totality_pool = clean_test_ids[:50] 
            
            for t_id in totality_pool:
                try:
                    t_states, t_actions, t_targets = data_manager.get_unbroken_test_trajectory(t_id)
                    t_horizon = min(HORIZON_LIMIT, len(t_actions))
                    t_init_state = t_states[0].unsqueeze(0)
                    t_act_seq = t_actions[:t_horizon]
                    t_gt_path = t_targets[:t_horizon]
                    
                    t_det_preds = run_deterministic_rollout(trained_model, t_init_state, t_act_seq, device)
                    
                    true_paths.append(t_gt_path)
                    pred_paths.append(t_det_preds)
                   
                except Exception as e:
                    continue
                    
            # Generate the Totality Plot
            if len(true_paths) > 0:
                output_file = os.path.join(OUTPUT_DIR, f"TOTALITY_global_error{suffix}.jpg")
                plot_pure_deterministic_totality(true_paths, pred_paths, output_file)
            else:
                print("[-] Error: No valid trajectories found for Totality evaluation.")
        
        else:
            raise ValueError(f"[-] Unknown DET_EVAL_MODE choice: '{DET_EVAL_MODE}'")

    # =================================================================
    # PROBABILISTIC EVALUATION FLOW
    # =================================================================
    else:
        rollout_means, rollout_vars = run_probabilistic_rollout(trained_model, initial_state, action_sequence, device)
        print_probabilistic_horizon_matrix(horizon, rollout_means, rollout_vars, ground_truth)
        
        plot_trajectory_comparison(ground_truth, rollout_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_prob_base{suffix}.jpg"), f"Probabilistic Baseline ({PROB_TYPE.upper()})")

        det_means, det_vars = run_deterministic_mean_propagation(trained_model, initial_state, action_sequence, device)
        mc_means, mc_vars, mc_particles = run_monte_carlo_propagation(trained_model, initial_state, action_sequence, 200, device)
        mm_means, mm_vars = run_moment_matching_propagation(trained_model, initial_state, action_sequence, device)
        
        print_part_c_comparison_matrix(horizon, det_means, det_vars, mc_means, mc_vars, mm_means, mm_vars, ground_truth)
        
        plot_trajectory_comparison(ground_truth, det_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_det_prop{suffix}.jpg"), "Deterministic Mean Propagation", det_vars)
        plot_trajectory_comparison(ground_truth, mc_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mc_prop{suffix}.jpg"), "Monte Carlo Propagation", mc_vars)
        plot_trajectory_comparison(ground_truth, mm_means, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mm_prop{suffix}.jpg"), "Moment Matching Propagation", mm_vars)
        plot_monte_carlo_fan_chart(ground_truth, mc_means, mc_particles, data_manager.means, data_manager.stds, os.path.join(OUTPUT_DIR, f"traj_{TARGET_TRAJECTORY_ID}_mc_fan{suffix}.jpg"))
        
        # ── PROJECT TOTALITY AUTOMATED EVALUATION LOOP ─────────────────
        print(f"\n[+] Launching Project Totality Evaluation across the dataset pool...")
        all_unique_trajectory_ids = sorted(data_manager.df["trajectory_id"].unique())
        totality_pool = test_ids[:50] 
        
        error_tracking_pool = []
        variance_tracking_pool = []
        
        for t_id in totality_pool:
            try:
                t_states, t_actions, t_targets = data_manager.get_unbroken_test_trajectory(t_id)
                t_horizon = min(HORIZON_LIMIT, len(t_actions))
                t_init_state = t_states[0].unsqueeze(0)
                t_act_seq = t_actions[:t_horizon]
                t_gt_path = t_targets[:t_horizon]
                
                t_det_m, t_det_v = run_deterministic_mean_propagation(trained_model, t_init_state, t_act_seq, device)
                t_mc_m, t_mc_v, _ = run_monte_carlo_propagation(trained_model, t_init_state, t_act_seq, 100, device)
                t_mm_m, t_mm_v = run_moment_matching_propagation(trained_model, t_init_state, t_act_seq, device)
                
                error_tracking_pool.append({'true': t_gt_path, 'det': t_det_m, 'mc': t_mc_m, 'mm': t_mm_m})
                variance_tracking_pool.append({'det_var': t_det_v, 'mc_var': t_mc_v, 'mm_var': t_mm_v})
            except Exception:
                continue 
                
        print(f"[+] Compiling totality statistics over {len(error_tracking_pool)} valid trajectories...")
        
        plot_global_error_totality(error_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_error_vs_horizon{suffix}.jpg"))
        plot_global_variance_totality(variance_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_variance_vs_time{suffix}.jpg"))
        plot_global_calibration(error_tracking_pool, variance_tracking_pool, os.path.join(OUTPUT_DIR, f"TOTALITY_calibration{suffix}.jpg"))
        
        print(f"\n[✓] All probabilistic global totality artifacts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()