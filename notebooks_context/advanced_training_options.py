"""Simple reusable options for Advanced_Training.ipynb.

The goal of this file is not to hide the lesson.  It only removes repeated
definitions so the notebook can keep a linear structure:

1. choose a model,
2. choose optimizer/loss/scheduler,
3. run the same 5-step training loop,
4. evaluate regression metrics.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def assess_device():
    """Return the best available device: MPS, CUDA, then CPU."""

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------
# AbaloneNet variants
# ---------------------------------------------------------------------


class AbaloneNet(nn.Module):
    """Baseline dense feed-forward network."""

    def __init__(self, n_in: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=n_in, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, x):
        return self.layer_stack(x)


class AbaloneNetDropout(nn.Module):
    """Baseline dense network with dropout regularization."""

    def __init__(self, n_in: int, dropout_probability: float = 0.1):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=n_in, out_features=64),
            nn.Dropout(dropout_probability),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(dropout_probability),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.Dropout(dropout_probability),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, x):
        return self.layer_stack(x)

class AbaloneNetBatchNorm(nn.Module):
    """Baseline dense network with batchnormalization."""
    def __init__(self, n_in: int, dropout_probability: float = 0.1):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

class AbaloneNetBayesian(nn.Module):
    """Flexible network whose architecture is chosen by Bayesian optimization."""

    def __init__(self, n_in: int, hidden_sizes, dropout_probability: float = 0.0):
        super().__init__()
        layers = []
        current_features = n_in

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout_probability > 0:
                layers.append(nn.Dropout(dropout_probability))
            current_features = hidden_size

        layers.append(nn.Linear(current_features, 1))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer_stack(x)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def make_dataloader(X, y, batch_size=64, shuffle=True):
    """Create a TensorDataset and DataLoader from tensors."""

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def read_split_abalone(
    path="abalone.csv",
    device="cpu",
    batch_size=64,
    train_size=0.70,
    random_state=42,
):
    """Read Abalone, split train/validation/test, standardize, and tensorize."""

    df = pd.read_csv(path)

    # The only categorical feature is sex.  We turn it into dummy variables.
    df = pd.get_dummies(df, columns=["sex"], dtype=float)

    X_df = df.drop("rings", axis=1)
    y_np = df["rings"].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
        X_df.to_numpy(dtype=np.float32),
        y_np,
        train_size=train_size,
        random_state=random_state,
    )

    X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(
        X_temp_np,
        y_temp_np,
        test_size=0.50,
        random_state=random_state,
    )

    # Fit the scaler only on training data, then apply to validation and test.
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
    X_val_np = scaler.transform(X_val_np).astype(np.float32)
    X_test_np = scaler.transform(X_test_np).astype(np.float32)

    x_train = torch.tensor(X_train_np, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, device=device, dtype=torch.float32)
    x_val = torch.tensor(X_val_np, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, device=device, dtype=torch.float32)
    x_test = torch.tensor(X_test_np, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, device=device, dtype=torch.float32)

    train_loader = make_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size=len(y_val), shuffle=False)
    test_loader = make_dataloader(x_test, y_test, batch_size=len(y_test), shuffle=False)

    return {
        "df": df,
        "feature_columns": list(X_df.columns),
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "batch_size": batch_size,
        "num_feats": x_train.shape[1],
    }


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------


def create_scheduler(optimizer, step_size=50, gamma=0.7):
    """Create a StepLR scheduler."""

    return StepLR(optimizer, step_size=step_size, gamma=gamma)


# ---------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------


def update_early_stopping(
    model,
    val_loss,
    state=None,
    patience=30,
    min_delta=0.001,
    restore_best_weights=True,
):
    """Update early-stopping state and return (should_stop, state)."""

    val_loss = float(val_loss)

    if state is None:
        state = {
            "best_loss": val_loss,
            "best_model": copy.deepcopy(model.state_dict()),
            "counter": 0,
            "status": "First validation loss saved.",
        }
        return False, state

    if state["best_loss"] - val_loss >= min_delta:
        state["best_loss"] = val_loss
        state["best_model"] = copy.deepcopy(model.state_dict())
        state["counter"] = 0
        state["status"] = "Improvement found, counter reset to 0."
        return False, state

    state["counter"] += 1
    state["status"] = f"No improvement in the last {state['counter']} epochs."

    if state["counter"] >= patience:
        state["status"] = f"Early stopping triggered after {state['counter']} epochs."
        if restore_best_weights and state["best_model"] is not None:
            model.load_state_dict(state["best_model"])
        return True, state

    return False, state


# ---------------------------------------------------------------------
# l1 normalization
# ---------------------------------------------------------------------


def l1_penalty(model):
    """Return the L1 norm of all model parameters."""

    return sum(parameter.abs().sum() for parameter in model.parameters())

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def basic_training_loop(
    model,
    train_loader,
    loss_function,
    optimizer,
    epochs=500,
    val_loader=None,
    scheduler=None,
    l1_lambda=0.0,
    patience=None,
    min_delta=0.001,
    print_every=25,
):
    """Train any regression model with the same 5-step PyTorch loop."""

    history = {"train_loss": [], "valid_loss": [], "valid_mae": []}
    early_state = None
    best_epoch = epochs

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for X, y in train_loader:
            # 1. Forward pass
            preds = model(X)

            # 2. Calculate loss
            base_loss = loss_function(preds, y)
            loss = base_loss + l1_lambda * l1_penalty(model)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backwards
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            batch_losses.append(base_loss.item())

        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(float(np.mean(batch_losses)))

        if val_loader is not None:
            model.eval()
            valid_loss_total = 0.0
            valid_abs_error_total = 0.0
            valid_observations = 0

            with torch.inference_mode():
                for X_valid, y_valid in val_loader:
                    valid_preds = model(X_valid)
                    valid_loss = loss_function(valid_preds, y_valid)
                    batch_size = y_valid.shape[0]

                    valid_loss_total += valid_loss.item() * batch_size
                    valid_abs_error_total += torch.abs(valid_preds - y_valid).sum().item()
                    valid_observations += batch_size

            valid_loss = valid_loss_total / valid_observations
            valid_mae = valid_abs_error_total / valid_observations

            history["valid_loss"].append(valid_loss)
            history["valid_mae"].append(valid_mae)

            if valid_loss == min(history["valid_loss"]):
                best_epoch = epoch + 1

            if print_every and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {history['train_loss'][-1]:.4f}, "
                    f"Validation Loss: {valid_loss:.4f}, "
                    f"Validation MAE: {valid_mae:.4f}"
                )

            if patience is not None:
                stop, early_state = update_early_stopping(
                    model,
                    valid_loss,
                    state=early_state,
                    patience=patience,
                    min_delta=min_delta,
                )
                if stop:
                    print(f"Stopping at epoch {epoch + 1}. {early_state['status']}")
                    break

    return model, history, best_epoch

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""

    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    r2 = r2_score(
        y_true.detach().cpu().numpy().reshape(-1),
        y_pred.detach().cpu().numpy().reshape(-1),
    )
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_regression(model, X, y, title="Evaluation", n_examples=10, print_results=True):
    """Evaluate a regression model and return metrics plus a prediction table."""

    model.eval()

    with torch.inference_mode():
        preds = model(X)

    metrics = regression_metrics(y, preds)
    prediction_table = pd.DataFrame(
        {
            "actual_rings": y.detach().cpu().numpy().reshape(-1)[:n_examples],
            "predicted_rings": np.round(
                preds.detach().cpu().numpy().reshape(-1)[:n_examples],
                2,
            ),
        }
    )

    if print_results:
        print(title)
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f} rings")
        print(f"MAE:  {metrics['mae']:.4f} rings")
        print(f"R^2:  {metrics['r2']:.4f}")

    return metrics, prediction_table


# ---------------------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------------------


def choose_batch_size(batch_size):
    """Map a continuous suggestion to a simple batch-size choice."""

    allowed_batch_sizes = np.array([32, 64, 128])
    closest = np.argmin(np.abs(allowed_batch_sizes - batch_size))
    return int(allowed_batch_sizes[closest])


def make_hidden_sizes(neuron_pct, neuron_shrink, max_neurons=256, max_layers=4):
    """Build hidden-layer sizes from neuron percentage and shrink factor."""

    total_neurons = max(8, int(max_neurons * neuron_pct))
    relative_sizes = np.array([neuron_shrink**i for i in range(max_layers)])
    hidden_sizes = np.round(total_neurons * relative_sizes / relative_sizes.sum()).astype(int)
    return [max(4, int(size)) for size in hidden_sizes]


def make_bayesian_model(model_choice, num_feats, dropout, neuron_pct, neuron_shrink, device):
    """Build either a flexible Bayesian model or one of the fixed variants."""

    if isinstance(model_choice, str):
        hidden_sizes = make_hidden_sizes(neuron_pct, neuron_shrink)
        dropout_probability = dropout if model_choice == "bayes_dropout" else 0.0
        return AbaloneNetBayesian(num_feats, hidden_sizes, dropout_probability).to(device)

    return model_choice(num_feats).to(device)


def bayesian_objective_factory(
    model_choice,
    num_feats,
    x_train,
    y_train,
    x_val,
    y_val,
    device,
    max_epochs=80,
    random_state=42,
):
    """Create the objective function used by BayesianOptimization."""

    trials = []

    def objective(dropout, neuron_pct, neuron_shrink, log_lr, log_l1, log_l2, batch_size):
        params = {
            "dropout": dropout,
            "neuron_pct": neuron_pct,
            "neuron_shrink": neuron_shrink,
            "lr": 10**log_lr,
            "l1_lambda": 10**log_l1,
            "l2_lambda": 10**log_l2,
            "batch_size": choose_batch_size(batch_size),
        }

        torch.manual_seed(random_state + len(trials))
        model = make_bayesian_model(
            model_choice,
            num_feats,
            dropout=params["dropout"],
            neuron_pct=params["neuron_pct"],
            neuron_shrink=params["neuron_shrink"],
            device=device,
        )
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["l2_lambda"])
        scheduler = create_scheduler(optimizer, step_size=25, gamma=0.7)
        train_loader = make_dataloader(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            shuffle=True,
        )
        val_loader = make_dataloader(
            x_val,
            y_val,
            batch_size=len(y_val),
            shuffle=False,
        )

        model, history, best_epoch = basic_training_loop(
            model,
            train_loader,
            loss_function,
            optimizer,
            epochs=max_epochs,
            val_loader=val_loader,
            scheduler=scheduler,
            l1_lambda=params["l1_lambda"],
            patience=12,
            print_every=None,
        )

        metrics, _ = evaluate_regression(
            model,
            x_val,
            y_val,
            title="Bayesian validation",
            print_results=False,
        )
        params["hidden_sizes"] = make_hidden_sizes(neuron_pct, neuron_shrink)
        params["best_epoch"] = best_epoch
        params["val_rmse"] = metrics["rmse"]
        trials.append(params)

        # bayesian-optimization maximizes, so return negative RMSE.
        return -metrics["rmse"]

    return objective, trials


def run_bayesian_optimization(objective, pbounds=None, init_points=5, n_iter=10, random_state=42):
    """Run BayesianOptimization with simple default bounds."""

    try:
        from bayes_opt import BayesianOptimization
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install the package with: pip install bayesian-optimization"
        ) from error

    if pbounds is None:
        pbounds = {
            "dropout": (0.0, 0.4),
            "neuron_pct": (0.2, 1.0),
            "neuron_shrink": (0.3, 0.9),
            "log_lr": (-4, -2),
            "log_l1": (-8, -3),
            "log_l2": (-8, -3),
            "batch_size": (32, 128),
        }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer


def train_final_bayesian_model(
    model_choice,
    best_params,
    num_feats,
    x_dev,
    y_dev,
    device,
    epochs,
):
    """Train the selected Bayesian model on train+validation data."""

    model = make_bayesian_model(
        model_choice,
        num_feats,
        dropout=best_params["dropout"],
        neuron_pct=best_params["neuron_pct"],
        neuron_shrink=best_params["neuron_shrink"],
        device=device,
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["l2_lambda"],
    )
    scheduler = create_scheduler(optimizer, step_size=25, gamma=0.7)
    train_loader = make_dataloader(
        x_dev,
        y_dev,
        batch_size=int(best_params["batch_size"]),
        shuffle=True,
    )

    model, history, best_epoch = basic_training_loop(
        model,
        train_loader,
        loss_function,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        l1_lambda=best_params["l1_lambda"],
        print_every=None,
    )
    return model, history, best_epoch
