# =====================================================================
# Multi-Step Uncertainty Propagation Engines
# =====================================================================
import torch

def run_deterministic_rollout(model, initial_state, action_sequence, device="cpu"):
    model.eval().to(device)
    pred_states = []
    current_state_pred = initial_state.to(device)
    with torch.no_grad():
        for k in range(len(action_sequence)):
            action = action_sequence[k].view(1).to(device)
            next_state_pred = model(current_state_pred, action)
            pred_states.append(next_state_pred.cpu())
            current_state_pred = next_state_pred
    return torch.cat(pred_states, dim=0)

def run_probabilistic_rollout(model, initial_state, action_sequence, device="cpu"):
    model.eval().to(device)
    pred_means, pred_vars = [], []
    current_state_pred = initial_state.to(device)
    with torch.no_grad():
        for k in range(len(action_sequence)):
            action = action_sequence[k].view(1).to(device)
            mu, var = model(current_state_pred, action)
            pred_means.append(mu.cpu())
            pred_vars.append(var.cpu())
            current_state_pred = mu
    return torch.cat(pred_means, dim=0), torch.cat(pred_vars, dim=0)

def run_deterministic_mean_propagation(model, initial_state, action_sequence, device="cpu"):
    return run_probabilistic_rollout(model, initial_state, action_sequence, device)

def run_monte_carlo_propagation(model, initial_state, action_sequence, num_particles=100, device="cpu"):
    model.eval().to(device)
    current_particles = initial_state.repeat(num_particles, 1).to(device)
    trajectory_means, trajectory_vars, all_particle_paths = [], [], []
    
    with torch.no_grad():
        for k in range(len(action_sequence)):
            action = action_sequence[k].repeat(num_particles).to(device)
            mu, var = model(current_particles, action)
            
            step_mean = torch.mean(mu, dim=0, keepdim=True)
            step_var = torch.mean(var, dim=0, keepdim=True) + torch.var(mu, dim=0, keepdim=True)
            
            # Stochastic Handoff Update Loop
            noise = torch.randn_like(mu)
            current_particles = mu + torch.sqrt(var) * noise
            
            trajectory_means.append(step_mean.cpu())
            trajectory_vars.append(step_var.cpu())
            all_particle_paths.append(current_particles.unsqueeze(0).cpu()) # Aligned POST-update index
            
    return torch.cat(trajectory_means, dim=0), torch.cat(trajectory_vars, dim=0), torch.cat(all_particle_paths, dim=0)

def run_moment_matching_propagation(model, initial_state, action_sequence, device="cpu"):
    model.eval().to(device)  
    state_dim = initial_state.shape[-1]
    pred_means, pred_vars = [], []
    current_mean = initial_state.clone().to(device)
    current_cov = torch.zeros(state_dim, state_dim, device=device)
    
    for k in range(len(action_sequence)):
        
        x_wg = current_mean.detach().clone().requires_grad_(True)
        action = action_sequence[k].view(1).to(device)
        mu, var = model(x_wg, action)
        
        jacobian = torch.zeros(state_dim, state_dim, device=device)
        for i in range(state_dim):
            grad_i = torch.autograd.grad(mu[0, i], x_wg, retain_graph=True, create_graph=False)[0] 
            jacobian[i] = grad_i[0].detach()
            
        with torch.no_grad():
            next_cov = torch.mm(torch.mm(jacobian, current_cov), jacobian.t()) + torch.diag(var[0])
            next_var = torch.diag(next_cov).unsqueeze(0)
            
            pred_means.append(mu.detach().cpu())
            pred_vars.append(next_var.cpu())
            
            current_mean = mu.detach().clone()
            current_cov = next_cov.clone()
            
    return torch.cat(pred_means, dim=0), torch.cat(pred_vars, dim=0)