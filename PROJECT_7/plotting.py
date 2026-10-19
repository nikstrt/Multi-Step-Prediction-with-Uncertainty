
# =====================================================================
# Complete Visualization Suite
# =====================================================================
import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_ragged_step_means(list_of_arrays, max_horizon=20):
    """Safely handles trajectories of varying lengths without inhomogeneous shape crashes."""
    step_means = []
    for k in range(max_horizon):
        valid_values = [arr[k] for arr in list_of_arrays if len(arr) > k]
        if valid_values: 
            step_means.append(np.mean(valid_values))
        else: 
            break
    return np.array(step_means)


def plot_trajectory_comparison(true_z, pred_z, means, stds, output_filename, title_label="Prediction", pred_vars=None):
    """True vs predicted trajectories for a single sequence with optional ±2σ shading."""
    true_np, pred_np = true_z.numpy(), pred_z.numpy()
    m, s = means.values, stds.values
    true_real, pred_real = true_np * s + m, pred_np * s + m
    steps = np.arange(1, len(true_real) + 1)
    
    names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    units = ["m", "m/s", "rad", "rad/s"]
    colors = ["teal", "purple", "darkorange", "crimson"]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Open-Loop Multi-Step Rollout Analysis ({title_label})", fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axs.flat):
        ax.plot(steps, true_real[:, idx], label="Ground Truth", color="black", lw=2.0)
        ax.plot(steps, pred_real[:, idx], label=f"{title_label}", color=colors[idx], lw=2.0, ls="--")
        
        if pred_vars is not None:
            shade = np.sqrt(pred_vars.numpy()[:, idx]) * s[idx]
            ax.fill_between(steps, pred_real[:, idx] - 2 * shade, pred_real[:, idx] + 2 * shade, color=colors[idx], alpha=0.15, label=r"$\pm 2\sigma$")
            
        ax.set_title(names[idx], fontweight='bold')
        ax.set_xlabel("Look-Ahead Step (k)"); ax.set_ylabel(f"({units[idx]})")
        ax.grid(True, ls=":", alpha=0.6); ax.legend(loc="best")
        ax.set_xticks(steps)
    plt.tight_layout(); plt.savefig(output_filename, dpi=300); plt.close()


def plot_monte_carlo_fan_chart(true_z, mc_means, mc_particles, means, stds, output_filename, num_samples=15):
    """Renders the stochastic particle clouds traveling alongside the ensemble mean path."""
    true_np, mean_np, parts_np = true_z.numpy(), mc_means.numpy(), mc_particles.numpy()
    m, s = means.values, stds.values
    true_real, mean_real = true_np * s + m, mean_np * s + m
    steps = np.arange(1, parts_np.shape[0] + 1)
    colors = ["teal", "purple", "darkorange", "crimson"]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monte Carlo Stochastic Fan Path Distribution Analysis", fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axs.flat):
        for j in range(min(num_samples, parts_np.shape[1])):
            p_real = parts_np[:, j, idx] * s[idx] + m[idx]
            ax.plot(steps, p_real, color=colors[idx], alpha=0.15, lw=0.8, label="Sampled Particles" if j==0 else None)
        ax.plot(steps, true_real[:, idx], color="black", lw=2.0, label="Ground Truth")
        ax.plot(steps, mean_real[:, idx], color=colors[idx], lw=2.0, ls="--", label="Ensemble Mean")
        ax.set_title(f"{idx}", fontweight='bold'); ax.grid(True, ls=":", alpha=0.6); ax.legend()
        ax.set_xticks(steps)
    plt.tight_layout(); plt.savefig(output_filename, dpi=300); plt.close()


def plot_global_error_totality(trajectory_data_list, output_filename):
    """
    Computes and renders the true global prediction error (MSE) across the 
    totality of all generated trajectories for each look-ahead horizon step (k).
    Includes faint background trajectory spreads for out-of-sample visual analysis.
    """
    plt.figure(figsize=(10, 6))
    all_det, all_mc, all_mm = [], [], []
    max_len = 0
    
    # First Pass: Compute MSE paths per trajectory and overlay faint background tracks
    for traj in trajectory_data_list:
        true = traj['true'].numpy() if torch.is_tensor(traj['true']) else np.array(traj['true'])
        det  = traj['det'].numpy()  if torch.is_tensor(traj['det'])  else np.array(traj['det'])
        mc   = traj['mc'].numpy()   if torch.is_tensor(traj['mc'])   else np.array(traj['mc'])
        mm   = traj['mm'].numpy()   if torch.is_tensor(traj['mm'])   else np.array(traj['mm'])
        
        err_det = np.mean((det - true)**2, axis=1)
        err_mc  = np.mean((mc - true)**2, axis=1)
        err_mm  = np.mean((mm - true)**2, axis=1)
        
        all_det.append(err_det)
        all_mc.append(err_mc)
        all_mm.append(err_mm)
        
        max_len = max(max_len, len(err_det))
        steps = np.arange(1, len(err_det) + 1)
        
        # RESTORED: Dynamic faint background line overlays
        plt.plot(steps, err_det, color="crimson", alpha=0.03, linewidth=0.4)
        plt.plot(steps, err_mc,  color="teal",    alpha=0.03, linewidth=0.4)
        plt.plot(steps, err_mm,  color="darkorange", alpha=0.03, linewidth=0.4)
        
    # Second Pass: Compute true ragged step means across the totality pool
    g_det = compute_ragged_step_means(all_det, max_len)
    g_mc  = compute_ragged_step_means(all_mc, max_len)
    g_mm  = compute_ragged_step_means(all_mm, max_len)
    global_steps = np.arange(1, len(g_det) + 1)
    
    # Overlay high-contrast bold consensus averages
    plt.plot(global_steps, g_det, label="Deterministic Mean Engine (Global Avg)", color="crimson", linestyle="-", marker="o", linewidth=2.5)
    plt.plot(global_steps, g_mc,  label="Monte Carlo Ensemble (Global Avg)",      color="teal",    linestyle="--", marker="s", linewidth=2.5)
    plt.plot(global_steps, g_mm,  label="Analytical Moment Match (Global Avg)",   color="darkorange", linestyle=":", marker="^", linewidth=2.5)
    
    plt.title("PROJECT TOTALITY: Global Prediction Error vs. Look-Ahead Horizon (k)", fontsize=12, fontweight="bold")
    plt.xlabel("Look-Ahead Horizon Step (k)", fontsize=10)
    plt.ylabel("Mean Squared Error (MSE) [Standardized Z-Space]", fontsize=10)
    plt.legend(loc="upper left", fontsize=10); plt.grid(True, ls=":", alpha=0.6); plt.xticks(global_steps)
    plt.tight_layout(); plt.savefig(output_filename, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[+] SUCCESS: Global totality error plot saved cleanly as '{output_filename}'")


def plot_global_variance_totality(trajectory_variance_list, output_filename):
    """
    Computes and renders the true global predicted variance across the
    totality of all generated trajectories for each look-ahead horizon step (k).
    Restores faint background predicted variance paths for full distribution coverage.
    """
    plt.figure(figsize=(10, 6))
    all_det, all_mc, all_mm = [], [], []
    max_len = 0
    
    # First Pass: Compute mean predicted variances per trajectory and overlay faint background tracks
    for traj in trajectory_variance_list:
        v_det = traj['det_var'].numpy() if torch.is_tensor(traj['det_var']) else np.array(traj['det_var'])
        v_mc  = traj['mc_var'].numpy()  if torch.is_tensor(traj['mc_var'])  else np.array(traj['mc_var'])
        v_mm  = traj['mm_var'].numpy()  if torch.is_tensor(traj['mm_var'])  else np.array(traj['mm_var'])
        
        mean_v_det = np.mean(v_det, axis=1)
        mean_v_mc  = np.mean(v_mc, axis=1)
        mean_v_mm  = np.mean(v_mm, axis=1)
        
        all_det.append(mean_v_det)
        all_mc.append(mean_v_mc)
        all_mm.append(mean_v_mm)
        
        max_len = max(max_len, len(mean_v_det))
        steps = np.arange(1, len(mean_v_det) + 1)
        
        # RESTORED: Dynamic faint background line overlays
        plt.plot(steps, mean_v_det, color="crimson", alpha=0.03, linewidth=0.4)
        plt.plot(steps, mean_v_mc,  color="teal",    alpha=0.03, linewidth=0.4)
        plt.plot(steps, mean_v_mm,  color="darkorange", alpha=0.03, linewidth=0.4)
        
    # Second Pass: Compute true ragged step means across the totality pool
    g_det = compute_ragged_step_means(all_det, max_len)
    g_mc  = compute_ragged_step_means(all_mc, max_len)
    g_mm  = compute_ragged_step_means(all_mm, max_len)
    global_steps = np.arange(1, len(g_det) + 1)
    
    # Overlay high-contrast bold consensus uncertainty growth paths
    plt.plot(global_steps, g_det, label="Deterministic Mean Engine (Global Flat)", color="crimson", linestyle="-", marker="o", linewidth=2.5)
    plt.plot(global_steps, g_mc,  label="Monte Carlo Ensemble (Global Expanding)", color="teal",    linestyle="--", marker="s", linewidth=2.5)
    plt.plot(global_steps, g_mm,  label="Analytical Moment Match (Global Expanding)", color="darkorange", linestyle=":", marker="^", linewidth=2.5)
    
    plt.title("PROJECT TOTALITY: Global Predicted Uncertainty (Variance) vs. Horizon (k)", fontsize=12, fontweight="bold")
    plt.xlabel("Look-Ahead Horizon Step (k)", fontsize=10)
    plt.ylabel(r"Mean Predicted Variance ($\sigma^2$) [Standardized Z-Space]", fontsize=10)
    plt.legend(loc="upper left", fontsize=10); plt.grid(True, ls=":", alpha=0.6); plt.xticks(global_steps)
    plt.tight_layout(); plt.savefig(output_filename, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[+] SUCCESS: Global totality variance plot saved cleanly as '{output_filename}'")



def plot_global_calibration(error_tracking_pool, variance_tracking_pool, output_filename):
 
    fig, ax = plt.subplots(figsize=(7, 7))
    
    markers = {"Det. Mean Propagation": "o", "Monte Carlo (N=100)": "s", "Moment Matching": "^"}
    colors  = ["steelblue", "darkorange", "green"]
    
    methods = [
        ("Det. Mean Propagation", 'det', 'det_var'),
        ("Monte Carlo (N=100)", 'mc', 'mc_var'),
        ("Moment Matching", 'mm', 'mm_var')
    ]
    
    # Find the maximum horizon length k
    max_len = max(len(t['true']) for t in error_tracking_pool)
    all_vals = []
    sc = None

    for (label, mean_key, var_key), color in zip(methods, colors):
        global_k_stds = []
        global_k_errs = []
        
        # For every time step k of the horizon
        for k in range(max_len):
            step_stds = []
            step_errs = []
            
            # Gather the error and the Std from all Totality pool trajectories
            for err_dict, var_dict in zip(error_tracking_pool, variance_tracking_pool):
                if k < len(err_dict['true']):
                    true_z = err_dict['true'][k]
                    pred_z = err_dict[mean_key][k]
                    pred_v = var_dict[var_key][k]
                    
                    step_stds.append(torch.mean(torch.sqrt(pred_v)).item())
                    step_errs.append(torch.mean(torch.abs(pred_z - true_z)).item())
            
            # Calculate global average at step k
            if step_stds and step_errs:
                global_k_stds.append(np.mean(step_stds))
                global_k_errs.append(np.mean(step_errs))
                
        marker_style = markers.get(label, "o")
        sc = ax.scatter(global_k_stds, global_k_errs, 
                        c=range(len(global_k_stds)), cmap="viridis", 
                        marker=marker_style, s=75, alpha=0.85, 
                        edgecolors="black", linewidths=0.5, label=label, zorder=3)
        all_vals.extend(global_k_stds + global_k_errs)

    max_val = max(all_vals) * 1.1 if all_vals else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="Perfect Calibration (y=x)")

    ax.set_xlabel(r"Global Predicted Std ($\sigma$) [Normalized Space]", fontsize=11)
    ax.set_ylabel("Global Actual Absolute Error [Normalized Space]", fontsize=11)
    ax.set_title("Global Uncertainty Calibration (50 Trajectories Pool)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Horizon Look-Ahead Step (k)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved Global Totality Calibration Chart: {output_filename}")



def compute_ragged_step_means(list_of_arrays, max_len):
    """Helper to compute step-wise means for arrays of different lengths."""
    sums = np.zeros(max_len)
    counts = np.zeros(max_len)
    for arr in list_of_arrays:
        length = len(arr)
        sums[:length] += arr
        counts[:length] += 1
    # Avoid division by zero
    counts[counts == 0] = 1 
    return sums / counts

def plot_pure_deterministic_totality(all_true_trajectories, all_pred_trajectories, output_filename):
    """
    Plots the global deterministic error.
    all_true_trajectories: List of numpy arrays [horizon, state_dim]
    all_pred_trajectories: List of numpy arrays [horizon, state_dim]
    """
    plt.figure(figsize=(10, 6))
    all_mse_curves = []
    max_len = 0
    
    # Calculate MSE for every trajectory
    for true_path, pred_path in zip(all_true_trajectories, all_pred_trajectories):
        # Convert to numpy if they are tensors
        if torch.is_tensor(true_path): true_path = true_path.cpu().numpy()
        if torch.is_tensor(pred_path): pred_path = pred_path.cpu().numpy()
            
        # Compute MSE across the 4 state dimensions for each step k
        # true_path shape: (H, 4), pred_path shape: (H, 4)
        err_curve = np.mean((pred_path - true_path)**2, axis=1)
        all_mse_curves.append(err_curve)
        
        max_len = max(max_len, len(err_curve))
        steps = np.arange(1, len(err_curve) + 1)
        
        # Plot the faint background line for this specific trajectory
        plt.plot(steps, err_curve, color="crimson", alpha=0.08, linewidth=0.8)
        
    # Calculate the global average across all trajectories
    global_avg_curve = compute_ragged_step_means(all_mse_curves, max_len)
    global_steps = np.arange(1, len(global_avg_curve) + 1)
    
    # Plot the bold global average line
    plt.plot(global_steps, global_avg_curve, label="Deterministic Global Avg MSE", 
             color="crimson", linestyle="-", marker="o", linewidth=2.5)
    
    plt.title("Deterministic Prediction Error vs. Look-Ahead Horizon (k)", fontsize=12, fontweight="bold")
    plt.xlabel("Look-Ahead Horizon Step (k)", fontsize=10)
    plt.ylabel("Mean Squared Error (MSE) [Standardized]", fontsize=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, ls=":", alpha=0.6)
    plt.xticks(global_steps)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] SUCCESS: Deterministic totality error plot saved to '{output_filename}'")