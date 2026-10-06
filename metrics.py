import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, roc_auc_score, top_k_accuracy_score
)
from src.config import GENRES

def get_predictions(model, loader, device, return_probs=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1]
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_labels.extend(labels.numpy())
            
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs) if return_probs else None
    )


def majority_vote(labels_seg, preds_seg, dataset_seg):
    song_ids = np.array([s[2] for s in dataset_seg.segments])
    unique_songs = np.unique(song_ids)
    song_true, song_pred = [], []
    
    for sid in unique_songs:
        mask = song_ids == sid
        song_true.append(np.bincount(labels_seg[mask]).argmax())
        song_pred.append(np.bincount(preds_seg[mask], minlength=len(GENRES)).argmax())
        
    return np.array(song_true), np.array(song_pred)


def compute_metrics(y_true, y_pred, y_prob=None, label="Model"):
    return {
        'Model': label,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 (macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 (weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Top-2 Accuracy': top_k_accuracy_score(y_true, y_prob, k=2) if y_prob is not None else np.nan,
        'Top-3 Accuracy': top_k_accuracy_score(y_true, y_prob, k=3) if y_prob is not None else np.nan,
        'Log Loss': log_loss(y_true, y_prob) if y_prob is not None else np.nan,
        'ROC-AUC (macro OvR)': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro') if y_prob is not None else np.nan,
    }
