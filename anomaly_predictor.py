import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from solver import my_kl_loss, _normalize_prior


class AnomalyPredictor:
    """Utility class for fast anomaly inference."""

    def __init__(
        self,
        model_path: str,
        data_path: Optional[str],
        dataset: str = "SMD",
        data_arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        win_size: int = 100,
        step: int = 100,
        batch_size: int = 1024,
        num_workers: Optional[int] = None,
        anomaly_ratio: float = 4.0,
        device: Optional[str] = None,
    ):
        self.data_path = data_path
        self.data_arrays = data_arrays
        self.dataset = dataset
        self.win_size = win_size
        self.step = step
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.anomaly_ratio = anomaly_ratio
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # prepare data loaders
        if data_arrays is None:
            loader_fn = lambda mode: get_loader_segment(
                data_path,
                batch_size=batch_size,
                win_size=win_size,
                step=step,
                mode=mode,
                dataset=dataset,
                num_workers=self.num_workers,
            )
        else:
            from data_factory.data_loader import get_loader_segment_from_arrays

            loader_fn = lambda mode: get_loader_segment_from_arrays(
                data_arrays,
                batch_size=batch_size,
                win_size=win_size,
                step=step,
                mode=mode,
                dataset=dataset,
                num_workers=self.num_workers,
            )

        self.train_loader = loader_fn("train")
        self.thre_loader = loader_fn("thre")
        self.test_loader = loader_fn("test")

        feat_dim = self.train_loader.dataset.train.shape[-1]
        self.model = AnomalyTransformer(win_size=win_size, enc_in=feat_dim, c_out=feat_dim, e_layers=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.criterion = nn.MSELoss(reduction="none")
        self.temperature = 50

    def _compute_energy(self, loader, return_label: bool = False):
        energies = []
        labels = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            for batch, label in loader:
                batch = batch.float().to(self.device, non_blocking=True)
                output, series, prior, _ = self.model(batch)
                loss = torch.mean(self.criterion(batch, output), dim=-1)
                series_loss = torch.zeros_like(loss)
                prior_loss = torch.zeros_like(loss)
                for s, p in zip(series, prior):
                    p_norm = _normalize_prior(p)
                    series_loss += my_kl_loss(s, p_norm.detach())
                    prior_loss += my_kl_loss(p_norm, s.detach())
                metric = torch.softmax((-series_loss - prior_loss) * self.temperature, dim=-1)
                cri = metric * loss
                energies.append(cri.cpu())
                if return_label:
                    labels.append(label)
        energies = torch.cat(energies, dim=0).view(-1).numpy()
        if return_label:
            labels = torch.cat(labels, dim=0).view(-1).numpy()
            return energies, labels
        return energies

    def predict(self) -> Tuple[pd.DataFrame, Tuple[float, float, float, float]]:
        train_energy = self._compute_energy(self.train_loader)
        # thre_energy = self._compute_energy(self.thre_loader)
        thre_energy, thre_labels = self._compute_energy(self.thre_loader, return_label=True)

        combined_energy = np.concatenate([train_energy, thre_energy])
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        # thre_energy, thre_labels = self._compute_energy(self.thre_loader, return_label=True)


        pred = (thre_energy > thresh).astype(int)
        gt = thre_labels.astype(int)
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average="binary")

        results_df = pd.DataFrame({
        "anomaly_score": thre_energy,
        "prediction": pred,
        "ground_truth": gt,
    })

        return results_df#, (accuracy, precision, recall, f_score)


__all__ = ["AnomalyPredictor"]
