"""
experiment_tracker.py
---------------------
Tracks all ablation experiment results to a CSV file.
Used by experiments.py and train.py to log every run automatically.

Usage:
    from results.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    tracker.log(model_type='CNN', experiment='BatchNorm', config={...}, val_acc=0.72, test_acc=0.70, ...)
    tracker.save()
"""

import csv
import os
from datetime import datetime


# All columns we want to track across CNN and ViT experiments
FIELDNAMES = [
    'timestamp',
    'model_type',       # 'CNN' or 'ViT'
    'experiment',       # e.g. 'BatchNorm', 'Activation', 'NumLayers', 'Residual', 'NumHeads', 'PatchSize', 'PosEmb'
    'config',           # Human-readable description, e.g. 'activation=GELU, layers=5'

    # Shared hyperparameters
    'num_layers',
    'learning_rate',
    'batch_size',
    'num_epochs',
    'img_size',

    # CNN-specific
    'use_batchnorm',
    'activation',
    'use_residual',

    # ViT-specific
    'embed_dim',
    'num_heads',
    'patch_size',
    'use_pos_embedding',

    # Results
    'best_val_acc',     # Best validation accuracy across all epochs (0-100 scale)
    'test_acc',         # Final test accuracy at the chosen checkpoint (0-100 scale)
    'best_epoch',       # Epoch at which best_val_acc was achieved
    'total_params',     # Total learnable parameters in the model
    'inference_ms',     # Average inference time per image in milliseconds
    'notes',
]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'experiments.csv')


class ExperimentTracker:
    """Append-only CSV logger for experiment results."""

    def __init__(self, path: str = RESULTS_PATH):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        """Create CSV with header row if it doesn't exist yet."""
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            print(f"[Tracker] Created new experiment log at: {self.path}")

    def log(self, **kwargs) -> dict:
        """
        Log a single experiment result row.

        Provide any subset of FIELDNAMES as keyword arguments.
        Missing fields are filled with empty strings.

        Returns the row dict that was written.

        Example
        -------
        tracker.log(
            model_type='CNN',
            experiment='Activation',
            config='activation=GELU, layers=5, bn=True',
            activation='GELU',
            num_layers=5,
            use_batchnorm=True,
            best_val_acc=68.4,
            test_acc=66.1,
            best_epoch=12,
            total_params=1_234_567,
        )
        """
        row = {field: '' for field in FIELDNAMES}
        row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for key, value in kwargs.items():
            if key in FIELDNAMES:
                row[key] = value
            else:
                print(f"[Tracker] Warning: unknown field '{key}' — skipped.")

        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

        print(f"[Tracker] Logged: {row['model_type']} | {row['experiment']} | "
              f"val={row['best_val_acc']}% | test={row['test_acc']}%")
        return row

    def print_summary(self):
        """Print all logged results to stdout as a formatted table."""
        if not os.path.exists(self.path):
            print("[Tracker] No results logged yet.")
            return

        import pandas as pd
        df = pd.read_csv(self.path)
        cols = ['timestamp', 'model_type', 'experiment', 'config',
                'best_val_acc', 'test_acc', 'best_epoch', 'total_params']
        print(df[cols].to_string(index=False))