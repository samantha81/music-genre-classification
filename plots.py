import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.config import GENRES, MUSICAL_SIMILARITIES
from src.utils import plt, PALETTE, HAS_SNS, save_fig

try:
    import seaborn as sns
except ImportError:
    sns = None
    HAS_SNS = False

def plot_hp_loss_curves(tune_results, best_cfg, title, fname):
    epochs_t = range(1, len(tune_results[0]["train_loss"]) + 1)
    colors = plt.cm.tab10.colors
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, r in enumerate(tune_results):
        is_best = r["name"] == best_cfg["name"]
        lw = 2.5 if is_best else 1.1
        ls = "-" if is_best else "--"
        c = colors[i % len(colors)]
        ax_train.plot(epochs_t, r["train_loss"], lw=lw, ls=ls, color=c, label=r["name"])
        ax_val.plot(epochs_t, r["val_loss"], lw=lw, ls=ls, color=c, label=r["name"])
        
    for ax, t in [(ax_train, "Train Loss"), (ax_val, "Validation Loss")]:
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax.legend(fontsize=8, ncol=2, framealpha=0.7)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fname)


def plot_hp_results_table(tune_results, best_cfg, sorted_results, col_labels, fname, extra_col=None):
    cell_data = []
    for rank, r in enumerate(sorted_results, start=1):
        star = " *" if r["name"] == best_cfg["name"] else ""
        row = [f"{rank}{star}", r["name"], f"{r['lr']:.3e}", f"{r['dropout']:.2f}"]
        if extra_col:
            row.append(str(r.get(extra_col, "")))
        row += [f"{r['wd']:.2e}", f"{r['best_val_loss']:.4f}", f"{r['best_val_acc']:.4f}"]
        cell_data.append(row)
        
    n_rows = len(cell_data)
    n_cols = len(col_labels)
    fig, ax_tbl = plt.subplots(figsize=(13, 0.55 * (n_rows + 2)))
    ax_tbl.axis("off")
    
    tbl = ax_tbl.table(cellText=cell_data, colLabels=col_labels, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    col_widths = ([0.06, 0.12, 0.13, 0.12, 0.16, 0.17, 0.17] if not extra_col
                  else [0.06, 0.10, 0.11, 0.10, 0.09, 0.14, 0.15, 0.15])
                  
    for col, w in enumerate(col_widths[:n_cols]):
        for row in range(n_rows + 1):
            tbl[(row, col)].set_width(w)
            
    for col in range(n_cols):
        c = tbl[(0, col)]
        c.set_facecolor("#1a252f")
        c.set_text_props(color="white", fontweight="bold", fontsize=11)
        c.set_edgecolor("#ffffff")
        
    for row in range(1, n_rows + 1):
        is_best = cell_data[row - 1][1] == best_cfg["name"]
        bg_default = "#f0f4f8" if row % 2 == 0 else "#ffffff"
        for col in range(n_cols):
            c = tbl[(row, col)]
            c.set_edgecolor("#d0d7de")
            if is_best:
                c.set_facecolor("#c8f7c5")
                c.set_text_props(fontweight="bold")
            else:
                c.set_facecolor(bg_default)
                
    ax_tbl.set_title(fname.replace("_", " ").replace(".png", "") + "  (* = best)", fontsize=12, fontweight="bold", pad=14)
    save_fig(fname)


def plot_spectrogram_grid(loader, n=6):
    GENRE_NAMES = {v: k for k, v in GENRES.items()}
    batch = next(iter(loader))
    mels = batch[0][:n]
    labels = batch[1][:n]
    
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i, (mel, lbl) in enumerate(zip(mels, labels)):
        axes[i].imshow(mel.squeeze().numpy(), origin='lower', aspect='auto', cmap='magma')
        axes[i].set_title(GENRE_NAMES[lbl.item()], fontsize=10)
        axes[i].axis('off')
        
    plt.suptitle("Sample Mel Spectrograms (normalised, post-augmentation)", y=1.02)
    plt.tight_layout()
    save_fig("fig_00_spectrogram_grid.png")
    print(f"Batch tensor shape: {mels.shape}")


def plot_training_curves(history, title, fname):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax1.set_title("Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    save_fig(fname)
    
    best_val_acc_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    best_val_loss_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    
    print(f"Best val acc : {max(history['val_acc']):.4f}  (epoch {best_val_acc_epoch})")
    print(f"Best val loss: {min(history['val_loss']):.4f}  (epoch {best_val_loss_epoch})")


def plot_confusion_matrix(y_true, y_pred, class_names, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax, data, fmt, subtitle in zip(axes, [cm, cm_norm], ['d', '.2f'], ['Count', 'Normalised (row %)']):
        if HAS_SNS:
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, linecolor='white', annot_kws={'size': 9}, ax=ax
            )
        else:
            ax.imshow(data, cmap='Blues', aspect='auto')
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = f"{data[i, j]:.2f}" if fmt == '.2f' else str(int(data[i, j]))
                    ax.text(j, i, val, ha='center', va='center', fontsize=8)
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names)
            
        ax.set_xlabel('Predicted Genre', fontsize=13)
        ax.set_ylabel('True Genre', fontsize=13)
        ax.set_title(f'{title} - {subtitle}', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
    plt.tight_layout()
    save_fig(fname, fig)
    
    np.fill_diagonal(cm, 0)
    flat = cm.flatten()
    sorted_idx = np.argsort(flat)[::-1]
    n = len(class_names)
    pairs = []
    
    for i in sorted_idx:
        if flat[i] > 0:
            r, c = divmod(i, n)
            pairs.append((class_names[r], class_names[c], flat[i]))
            
    print("\n  -> Most confused pair:")
    if pairs:
        a, b, v = pairs[0]
        print(f"    {a} -> {b}  ({int(v)} misclassifications)")
        sim = MUSICAL_SIMILARITIES.get(
            frozenset([a.lower(), b.lower()]),
            "These genres may share spectral or rhythmic characteristics."
        )
        print(f"    Interpretation: {sim}")
        
    if len(pairs) > 1:
        a, b, v = pairs[1]
        print(f"  -> Second most confused: {a} -> {b}  ({int(v)})")
    if pairs:
        a, b, v = pairs[-1]
        print(f"  -> Least confused: {a} -> {b}  ({int(v)})")


def plot_roc(y_true, y_prob, class_names, title, fname):
    n_classes = len(class_names)
    Y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        fpr_d[i], tpr_d[i], _ = roc_curve(Y_bin[:, i], y_prob[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])
        
    fpr_micro, tpr_micro, _ = roc_curve(Y_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    auc_macro = np.mean(list(auc_d.values()))
    
    all_fpr = np.unique(np.concatenate([fpr_d[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr_d[i], tpr_d[i]) for i in range(n_classes)) / n_classes

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.get_cmap('tab10', n_classes)
    
    for i, name in enumerate(class_names):
        axes[0].plot(fpr_d[i], tpr_d[i], lw=1.5, color=cmap(i), label=f'{name} (AUC={auc_d[i]:.3f})')
        
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{title}\nPer-class ROC')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(fpr_micro, tpr_micro, 'b-', lw=2, label=f'Micro AUC = {auc_micro:.4f}')
    axes[1].plot(all_fpr, mean_tpr, 'r-', lw=2, label=f'Macro AUC = {auc_macro:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{title}\nMacro / Micro ROC')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig(fname, fig)
    
    print(f"  Macro AUC = {auc_macro:.4f}  |  Micro AUC = {auc_micro:.4f}")
    print("  Per-class AUC:", {k: round(v, 4) for k, v in auc_d.items()})


def plot_pr_curve(y_true, y_prob, class_names, title, fname):
    n = len(class_names)
    Y_bin = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap('tab10', n)
    ap_scores = []
    
    for i, name in enumerate(class_names):
        p, r, _ = precision_recall_curve(Y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(Y_bin[:, i], y_prob[:, i])
        ap_scores.append(ap)
        ax.plot(r, p, lw=1.5, color=cmap(i), label=f'{name} (AP={ap:.3f})')
        
    macro_ap = np.mean(ap_scores)
    ax.axhline(1 / n, color='gray', linestyle='--', lw=1, label='Chance level')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title}\nMacro AP = {macro_ap:.4f}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_fig(fname)
    print(f"  Macro Average Precision = {macro_ap:.4f}")


def plot_training_history(histories, labels, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    idx = 0
    
    for hist, label in zip(histories, labels):
        for split, style in [('val', '-'), ('train', '--')]:
            lk = next((k for k in hist if 'loss' in k.lower() and split in k.lower()), None)
            ak = next((k for k in hist if 'acc' in k.lower() and split in k.lower()), None)
            c = colors[idx % len(colors)]
            
            if lk:
                axes[0].plot(hist[lk], linestyle=style, color=c, label=f'{label} {split}')
            if ak:
                axes[1].plot(hist[ak], linestyle=style, color=c, label=f'{label} {split}')
        idx += 1
        
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Training Dynamics', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fname)


def extract_embeddings(model, loader, device):
    embeddings, labels_list = [], []
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return None, None
        
    target_layer = linear_layers[-2][1] if len(linear_layers) >= 2 else linear_layers[-1][1]

    def hook_fn(module, input, output):
        embeddings.append(input[0].detach().cpu().numpy())

    handle = target_layer.register_forward_hook(hook_fn)
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            model(batch[0].to(device))
            labels_list.extend(batch[1].numpy())
            
    handle.remove()
    return (np.vstack(embeddings), np.array(labels_list)) if embeddings else (None, None)


def plot_embeddings(emb, lbl, class_names, title, fname, max_samples=2000):
    if len(emb) > max_samples:
        idx = np.random.choice(len(emb), max_samples, replace=False)
        emb, lbl = emb[idx], lbl[idx]
        
    emb_pca50 = PCA(n_components=50, random_state=42).fit_transform(emb)
    emb_pca2 = PCA(n_components=2, random_state=42).fit_transform(emb)
    emb_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(emb_pca50)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    
    for ax, coords, subtitle in zip(axes, [emb_pca2, emb_tsne], ['PCA (2D)', 't-SNE (2D)']):
        for i, name in enumerate(class_names):
            mask = lbl == i
            ax.scatter(coords[mask, 0], coords[mask, 1], s=15, color=cmap(i), label=name, alpha=0.7)
            
        ax.set_title(f'{title} - {subtitle}', fontsize=13)
        ax.legend(loc='best', fontsize=8, markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    save_fig(fname)

