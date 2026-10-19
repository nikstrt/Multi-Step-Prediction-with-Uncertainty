# =====================================================================
# Neural Network Architectures
# =====================================================================
import torch
import torch.nn as nn

class DeterministicPredictor(nn.Module):
    """One-Step Deterministic Predictor Module."""
    def __init__(self, state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.action_embedding = nn.Embedding(num_embeddings=action_dim, embedding_dim=embedding_dim)
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, states, actions):
        act_embed = self.action_embedding(actions)
        x = torch.cat([states, act_embed], dim=1)
        return self.network(x)


class ProbabilisticPredictor(nn.Module):
    """One-Step Probabilistic Predictor with Parametric Variance Heads."""
    def __init__(self, state_dim=4, action_dim=2, embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.action_embedding = nn.Embedding(num_embeddings=action_dim, embedding_dim=embedding_dim)
        
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Integrated regularization check
        )
        
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.var_head = nn.Linear(hidden_dim, state_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for tracking stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, states, actions):
        act_embed = self.action_embedding(actions)
        h = self.shared_backbone(torch.cat([states, act_embed], dim=1))
        
        mu = self.mean_head(h)
        log_var = self.var_head(h)
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)
        var = torch.exp(log_var)
        
        return mu, var