# =====================================================================
# Trajectory-Level Splitting Utilities
# =====================================================================
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class CartPoleDatasetManager:
    """Handles data ingestion, Z-score standardization, and strict leak-proof trajectory splitting."""
    def __init__(self, file_path="cartpole_stochastic_transitions.csv"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[-] Missing '{self.file_path}'. Run data_pipeline script first!")
            
        self.df = pd.read_csv(self.file_path)
        self.obs_cols = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
        self.next_obs_cols = ["next_cart_position", "next_cart_velocity", "next_pole_angle", "next_pole_angular_velocity"]
        self.means = None
        self.stds = None
        
        # Trackers for isolated IDs
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []

    def print_raw_telemetry(self, count=5):
        traj0 = self.df[self.df["trajectory_id"] == 0]
        view = traj0[["step_id"] + self.obs_cols + ["action"] + self.next_obs_cols]
        print(f"\n--- FIRST {count} STEPS TRANSITION TELEMETRY VERIFICATION ---")
        print(view.head(count).to_string(index=False))
        print("-------------------------------------------------------\n")

    def preprocess_and_split(self, batch_size=64, random_seed=42, target_eval_id=250):
        
        # Extraction of unique trajectory IDs
        unique_ids = self.df["trajectory_id"].unique()
        
        if target_eval_id in unique_ids:
            unique_ids = np.delete(unique_ids, np.where(unique_ids == target_eval_id))
            
        # Split the Trajectory IDs (70% Train, 30% Test)
        train_ids, tv_ids = train_test_split(unique_ids, test_size=0.30, random_state=random_seed)
        # Split the temporary IDs (50% Val, 50% Test -> 15% και 15% over the totality)
        val_ids, test_ids = train_test_split(tv_ids, test_size=0.50, random_state=random_seed)
        
        # Re-enetering target trajectory inside the Test Set
        test_ids = np.append(test_ids, target_eval_id)
        
        # Save the attributes of the class for control
        self.train_ids = list(train_ids)
        self.val_ids = list(val_ids)
        self.test_ids = list(test_ids)
        
        # Filter the DataFrame on the basis of the leak-proff trajectory sets
        train_df = self.df[self.df["trajectory_id"].isin(self.train_ids)]
        val_df   = self.df[self.df["trajectory_id"].isin(self.val_ids)]
        test_df  = self.df[self.df["trajectory_id"].isin(self.test_ids)]
        
        
        X_train = train_df[self.obs_cols].values.astype(np.float32)
        A_train = train_df["action"].values.astype(np.int64)
        Y_train = train_df[self.next_obs_cols].values.astype(np.float32)
        
        X_val   = val_df[self.obs_cols].values.astype(np.float32)
        A_val   = val_df["action"].values.astype(np.int64)
        Y_val   = val_df[self.next_obs_cols].values.astype(np.float32)
        
        X_test  = test_df[self.obs_cols].values.astype(np.float32)
        A_test  = test_df["action"].values.astype(np.int64)
        Y_test  = test_df[self.next_obs_cols].values.astype(np.float32)
        
        # Calculate the data transformation exclusively on the Train Split
        self.means = pd.Series(X_train.mean(axis=0), index=self.obs_cols)
        self.stds = pd.Series(X_train.std(axis=0), index=self.obs_cols)
        
        # Zero variance division guard floor
        self.stds = self.stds.replace(0.0, 1e-8)
        self.stds[self.stds < 1e-8] = 1e-8
        
        m, s = self.means.values, self.stds.values
        def _normalize(data): return (data - m) / s
        
        # Create TensorDatasets
        train_ds = TensorDataset(torch.FloatTensor(_normalize(X_train)), torch.LongTensor(A_train), torch.FloatTensor(_normalize(Y_train)))
        val_ds   = TensorDataset(torch.FloatTensor(_normalize(X_val)), torch.LongTensor(A_val), torch.FloatTensor(_normalize(Y_val)))
        test_ds  = TensorDataset(torch.FloatTensor(_normalize(X_test)), torch.LongTensor(A_test), torch.FloatTensor(_normalize(Y_test)))
        
        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        self._print_summary(X_train, X_val, X_test, train_loader, val_loader, test_loader)
        return train_loader, val_loader, test_loader

    def _print_summary(self, X_tr, X_v, X_te, tl, vl, tel):
        print("\n" + "="*60 + "\n  LEAK-PROOF DATASET PIPELINE INITIALISED (TRAJECTORY-LEVEL)\n" + "="*60)
        print(f"  Total Raw Transitions : {len(self.df):,}")
        print(f"  Train Set Trajectories: {len(self.train_ids)} ({len(X_tr):,} samples, {len(tl)} batches)")
        print(f"  Val Set Trajectories  : {len(self.val_ids)} ({len(X_v):,} samples, {len(vl)} batches)")
        print(f"  Test Set Trajectories : {len(self.test_ids)} ({len(X_te):,} samples, {len(tel)} batches)")
        print("="*60 + "\n")

    def get_unbroken_test_trajectory(self, trajectory_id=250):
        if self.means is None or self.stds is None:
            raise RuntimeError("[-] Call preprocess_and_split() before attempting segment extraction!")
             
        # Check if target trajectory is in the Test Split
        if trajectory_id not in self.test_ids:
            print(f"[!] WARNING: Trajectory ID {trajectory_id} was not dynamically sampled into the Test set.")
            print(f"    Forcing trajectory validation check anyway to ensure script execution...")
            
        traj_df = self.df[self.df["trajectory_id"] == trajectory_id].sort_values("step_id")
        if traj_df.empty:
            raise ValueError(f"[-] Trajectory ID {trajectory_id} not found anywhere in the dataset.")
            
        m, s = self.means.values, self.stds.values
        states_raw = traj_df[self.obs_cols].to_numpy(dtype=np.float32)
        actions_raw = traj_df["action"].to_numpy(dtype=np.int64)
        targets_raw = traj_df[self.next_obs_cols].to_numpy(dtype=np.float32)
        
        return (torch.FloatTensor((states_raw - m) / s), 
                torch.LongTensor(actions_raw.copy()), 
                torch.FloatTensor((targets_raw - m) / s))