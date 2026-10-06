import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report
from src.config import GENRES, PLOT_DIR, DEVICE, DATA_DIR
from src.utils import (
    PALETTE, HAS_PD, HAS_STATSMODELS, plt, save_fig,
    count_params, model_size_mb, measure_inference
)
from src.evaluation.metrics import get_predictions, majority_vote, compute_metrics
from src.evaluation.plots import (
    plot_confusion_matrix, plot_roc, plot_pr_curve,
    plot_training_history, extract_embeddings, plot_embeddings
)

try:
    import librosa
    import librosa.display as lrd
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

def run_full_evaluation(
    cnn_model, seg_model,
    test_dataset, test_loader,
    test_seg_dataset, test_seg_loader,
    history, history_seg,
    criterion, criterion_seg
):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    gn = list(GENRES.keys())

    # Collect predictions
    print("\nRunning CNN inference on test segments...")
    cnn_seg_true, cnn_seg_pred, cnn_seg_prob = None, None, None
    cnn_song_true, cnn_song_pred = None, None

    if cnn_model is not None and test_loader is not None:
        try:
            cnn_seg_true, cnn_seg_pred, cnn_seg_prob = get_predictions(
                cnn_model, test_loader, DEVICE
            )
            print(f"  Segments evaluated: {len(cnn_seg_true)}")
        except Exception as e:
            print(f"  CNN segment inference failed: {e}")

    if cnn_seg_true is not None and test_dataset is not None:
        try:
            cnn_song_true, cnn_song_pred = majority_vote(
                cnn_seg_true, cnn_seg_pred, test_dataset
            )
            print(f"  Songs (majority vote): {len(cnn_song_true)}")
        except Exception as e:
            print(f"  Majority vote failed: {e}")

    lstm_true, lstm_pred, lstm_prob = None, None, None
    if seg_model is not None and test_seg_loader is not None:
        try:
            lstm_true, lstm_pred, lstm_prob = get_predictions(
                seg_model, test_seg_loader, DEVICE
            )
            print(f"  CNN+LSTM songs evaluated: {len(lstm_true)}")
        except Exception as e:
            print(f"  CNN+LSTM inference failed: {e}")

    # 1. MODEL PERFORMANCE SUMMARY
    print("\n" + "=" * 60)
    print("  SECTION 1: Model Performance Summary")
    print("=" * 60)

    rows = []
    if cnn_seg_true is not None:
        rows.append(compute_metrics(cnn_seg_true, cnn_seg_pred, cnn_seg_prob, "CNN (Segment)"))
    if cnn_song_true is not None:
        rows.append(compute_metrics(cnn_song_true, cnn_song_pred, None, "CNN (Song, Majority Vote)"))
    if lstm_true is not None:
        rows.append(compute_metrics(lstm_true, lstm_pred, lstm_prob, "CNN+LSTM (Song)"))

    metrics_df = None
    if rows:
        if HAS_PD:
            metrics_df = pd.DataFrame(rows).set_index('Model')
            print("\n=== Model Performance Summary ===")
            print(metrics_df.round(4).to_string())

        plot_cols = [
            'Accuracy', 'F1 (macro)', 'F1 (weighted)',
            'Balanced Accuracy', "Cohen's Kappa", 'MCC'
        ]
        if HAS_PD and metrics_df is not None:
            plot_df = metrics_df[[c for c in plot_cols if c in metrics_df.columns]].T
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(plot_df))
            w = 0.8 / len(plot_df.columns)
            
            for i, col in enumerate(plot_df.columns):
                ax.bar(
                    x + i * w, plot_df[col], width=w, label=col,
                    color=PALETTE[i % len(PALETTE)], edgecolor='white', linewidth=0.6
                )
            ax.set_xticks(x + w * (len(plot_df.columns) - 1) / 2)
            ax.set_xticklabels(plot_df.index, rotation=25, ha='right')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison - Key Metrics')
            ax.legend(loc='lower right')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            save_fig('fig_s1_performance_comparison.png')
    else:
        print("[SKIP] No model results available for Section 1.")

    # 2. CLASSIFICATION REPORTS
    print("\n" + "=" * 60)
    print("  SECTION 2: Classification Reports")
    print("=" * 60)

    for title, y_true, y_pred in [
        ("CNN (Segment-level)", cnn_seg_true, cnn_seg_pred),
        ("CNN (Song-level, Majority Vote)", cnn_song_true, cnn_song_pred),
        ("CNN+LSTM (Song-level)", lstm_true, lstm_pred),
    ]:
        if y_true is None:
            print(f"[SKIP] {title}")
            continue
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
        print(classification_report(y_true, y_pred, target_names=gn, zero_division=0))

    # 3. CONFUSION MATRIX ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 3: Confusion Matrix Analysis")
    print("=" * 60)

    for title, y_true, y_pred, fname in [
        ("CNN Segment-level", cnn_seg_true, cnn_seg_pred, "fig_s3_cnn_segment_cm.png"),
        ("CNN Song-level (Majority Vote)", cnn_song_true, cnn_song_pred, "fig_s3_cnn_song_cm.png"),
        ("CNN+LSTM Song-level", lstm_true, lstm_pred, "fig_s3_lstm_cm.png"),
    ]:
        if y_true is None:
            print(f"[SKIP] {title}")
            continue
        plot_confusion_matrix(y_true, y_pred, gn, title, fname)

    # 4. ROC-AUC ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 4: ROC-AUC Analysis")
    print("=" * 60)

    roc_done = False
    for title, y_true, y_prob, fname in [
        ("CNN (Segment)", cnn_seg_true, cnn_seg_prob, "fig_s4_roc_cnn_seg.png"),
        ("CNN+LSTM (Song)", lstm_true, lstm_prob, "fig_s4_roc_lstm.png"),
    ]:
        if y_true is None or y_prob is None:
            print(f"[SKIP] {title}")
            continue
        plot_roc(y_true, y_prob, gn, title, fname)
        roc_done = True
    if not roc_done:
        print("[INFO] ROC-AUC skipped - no probability outputs available.")

    # 5. PRECISION-RECALL CURVE ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 5: Precision-Recall Curve Analysis")
    print("=" * 60)

    pr_done = False
    for title, y_true, y_prob, fname in [
        ("CNN (Segment)", cnn_seg_true, cnn_seg_prob, "fig_s5_pr_cnn.png"),
        ("CNN+LSTM (Song)", lstm_true, lstm_prob, "fig_s5_pr_lstm.png"),
    ]:
        if y_true is None or y_prob is None:
            print(f"[SKIP] {title}")
            continue
        plot_pr_curve(y_true, y_prob, gn, title, fname)
        pr_done = True
    if not pr_done:
        print("[INFO] Precision-Recall analysis skipped.")

    # 6. TRAINING DYNAMIC ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 6: Training Dynamic Analysis")
    print("=" * 60)

    avail_hist, avail_lbl = [], []
    if history is not None:
        avail_hist.append(history)
        avail_lbl.append('CNN')
    if history_seg is not None:
        avail_hist.append(history_seg)
        avail_lbl.append('CNN+LSTM')

    if avail_hist:
        plot_training_history(avail_hist, avail_lbl, 'fig_s6_training_dynamics.png')
        for hist, label in zip(avail_hist, avail_lbl):
            vlk = next((k for k in hist if 'val' in k.lower() and 'loss' in k.lower()), None)
            vak = next((k for k in hist if 'val' in k.lower() and 'acc' in k.lower()), None)
            tlk = next((k for k in hist if 'train' in k.lower() and 'loss' in k.lower()), None)
            
            if vlk:
                best_ep = int(np.argmin(hist[vlk])) + 1
                print(f"  [{label}] Best epoch (lowest val loss): {best_ep}  (val_loss={min(hist[vlk]):.4f})")
            if vak:
                best_ea = int(np.argmax(hist[vak])) + 1
                print(f"  [{label}] Best epoch (highest val acc): {best_ea}  (val_acc={max(hist[vak]):.4f})")
            if vlk and tlk:
                gap = hist[vlk][-1] - hist[tlk][-1]
                if gap > 0.1:
                    print(f"  [{label}] WARNING: Possible overfitting (train/val loss gap={gap:.4f})")
                else:
                    print(f"  [{label}] OK Well-generalised (gap={gap:.4f})")
    else:
        print("[SKIP] Training history not found.")

    # 7. GENRE-WISE PERFORMANCE ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 7: Genre-Wise Performance Analysis")
    print("=" * 60)

    def _genre_df(y_true, y_pred):
        if y_true is None or not HAS_PD:
            return None
        r = classification_report(
            y_true, y_pred, labels=list(range(len(gn))),
            target_names=gn, output_dict=True, zero_division=0
        )
        df = pd.DataFrame(r).T.loc[gn, ['precision', 'recall', 'f1-score']]
        df.columns = ['Precision', 'Recall', 'F1']
        return df.round(4)

    cnn_rep = _genre_df(cnn_seg_true, cnn_seg_pred)
    lstm_rep = _genre_df(lstm_true, lstm_pred)

    if cnn_rep is not None and lstm_rep is not None and HAS_PD:
        combined = pd.DataFrame({
            'CNN Precision': cnn_rep['Precision'].values,
            'CNN Recall': cnn_rep['Recall'].values,
            'CNN F1': cnn_rep['F1'].values,
            'LSTM Precision': lstm_rep['Precision'].values,
            'LSTM Recall': lstm_rep['Recall'].values,
            'LSTM F1': lstm_rep['F1'].values,
        }, index=gn)
        print(combined.to_string())
        
        improved = combined[combined['LSTM F1'] > combined['CNN F1']].index.tolist()
        degraded = combined[combined['LSTM F1'] < combined['CNN F1']].index.tolist()
        
        print(f"\n  Genres improved by LSTM: {improved}")
        print(f"  Genres degraded by LSTM:  {degraded}")
        print(f"  Best CNN F1:   {combined['CNN F1'].idxmax()} ({combined['CNN F1'].max():.4f})")
        print(f"  Worst CNN F1:  {combined['CNN F1'].idxmin()} ({combined['CNN F1'].min():.4f})")
        print(f"  Best LSTM F1:  {combined['LSTM F1'].idxmax()} ({combined['LSTM F1'].max():.4f})")
        print(f"  Worst LSTM F1: {combined['LSTM F1'].idxmin()} ({combined['LSTM F1'].min():.4f})")

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(gn))
        w = 0.35
        ax.bar(x - w / 2, combined['CNN F1'], w, label='CNN (segment)', color=PALETTE[0])
        ax.bar(x + w / 2, combined['LSTM F1'], w, label='CNN+LSTM (song)', color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(gn, rotation=35, ha='right')
        ax.set_ylabel('F1-score')
        ax.set_title('Genre-wise F1: CNN vs CNN+LSTM')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save_fig('fig_s7_genre_f1.png')

        # Sorted horizontal bar chart
        sorted_genres = combined['CNN F1'].sort_values().index
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(sorted_genres))
        ax.barh(y - 0.2, combined.loc[sorted_genres, 'CNN F1'], 0.35, label='CNN', color=PALETTE[0])
        ax.barh(y + 0.2, combined.loc[sorted_genres, 'LSTM F1'], 0.35, label='CNN+LSTM', color=PALETTE[1])
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_genres)
        ax.set_xlabel('F1-score')
        ax.set_title('Sorted Genre F1 Comparison')
        ax.legend()
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        save_fig('fig_s7_genre_f1_sorted.png')
    elif cnn_rep is not None:
        print("[INFO] Only CNN available for genre-wise plot.")
    else:
        print("[SKIP] Insufficient predictions for genre-wise analysis.")

    # 8. FEATURE EMBEDDING ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 8: Feature Embedding Analysis (PCA & t-SNE)")
    print("=" * 60)

    for model_name, m, l in [
        ("CNN", cnn_model, test_loader),
        ("CNN+LSTM", seg_model, test_seg_loader),
    ]:
        if m is None or l is None:
            print(f"[SKIP] {model_name}")
            continue
        try:
            emb, lbl = extract_embeddings(m, l, DEVICE)
            if emb is None:
                print(f"[SKIP] {model_name} - extraction failed")
                continue
            print(f"  {model_name}: embeddings shape {emb.shape}")
            safe_name = model_name.replace("+", "")
            plot_embeddings(emb, lbl, gn, model_name, f'fig_s8_embeddings_{safe_name}.png')
        except Exception as e:
            print(f"[SKIP] {model_name} embedding error: {e}")

    # 9. AUDIO ERROR ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 9: Audio Error Analysis")
    print("=" * 60)

    for tag, y_true, y_pred, y_prob in [
        ("CNN (segment)", cnn_seg_true, cnn_seg_pred, cnn_seg_prob),
        ("CNN+LSTM (song)", lstm_true, lstm_pred, lstm_prob),
    ]:
        if y_true is None:
            print(f"[SKIP] {tag}")
            continue
        wrong_idx = np.where(y_pred != y_true)[0]
        if len(wrong_idx) == 0:
            print(f"[{tag}] No misclassifications found!")
            continue

        if y_prob is not None:
            conf = y_prob[wrong_idx, y_pred[wrong_idx]]
            sort_order = np.argsort(-conf)
            top20_idx = wrong_idx[sort_order[:20]]
            top20_conf = conf[sort_order[:20]]
        else:
            top20_idx = wrong_idx[:20]
            top20_conf = np.ones(len(top20_idx))

        print(f"\n{'=' * 60}\n  {tag} - Top confident mistakes (up to 20)\n{'=' * 60}")
        for rank, (idx, conf_val) in enumerate(zip(top20_idx, top20_conf)):
            print(
                f"  {rank + 1:2d}. Sample {idx:4d}  "
                f"True: {gn[y_true[idx]]:<10}  Pred: {gn[y_pred[idx]]:<10}  Conf: {conf_val:.4f}"
            )

        if HAS_LIBROSA and test_dataset is not None:
            seg_list = getattr(test_dataset, 'segments', None)
            if seg_list is not None:
                n_plot = min(4, len(top20_idx))
                fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
                if n_plot == 1:
                    axes = axes[np.newaxis, :]
                    
                for row, (idx, conf_val) in enumerate(zip(top20_idx[:n_plot], top20_conf[:n_plot])):
                    seg_path = seg_list[idx][0]
                    genre_name = seg_path.parent.name
                    audio_stem = '_'.join(seg_path.stem.split('_')[:-1])
                    audio_path = DATA_DIR / "genres_original" / genre_name / (audio_stem + ".wav")
                    true_g, pred_g = gn[y_true[idx]], gn[y_pred[idx]]
                    ax_w, ax_m = axes[row, 0], axes[row, 1]
                    
                    if audio_path.exists():
                        try:
                            y_audio, sr = librosa.load(str(audio_path), sr=22050, duration=5.0)
                            ax_w.plot(np.linspace(0, 5, len(y_audio)), y_audio, lw=0.5, color=PALETTE[0])
                            ax_w.set_title(
                                f'True: {true_g} | Pred: {pred_g} | Conf: {conf_val:.2f}', fontsize=10
                            )
                            ax_w.set_xlabel('Time (s)')
                            ax_w.set_ylabel('Amplitude')
                            
                            mel_a = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, fmax=8000)
                            mel_db = librosa.power_to_db(mel_a, ref=np.max)
                            lrd.specshow(
                                mel_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax_m, cmap='magma'
                            )
                            ax_m.set_title('Mel Spectrogram')
                        except Exception as e:
                            ax_w.text(
                                0.5, 0.5, f'Error: {e}', ha='center', va='center',
                                transform=ax_w.transAxes, fontsize=8
                            )
                            ax_m.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_m.transAxes)
                    else:
                        ax_w.text(
                            0.5, 0.5, f'Audio not found\n{audio_path.name}', ha='center',
                            va='center', transform=ax_w.transAxes, fontsize=8
                        )
                        ax_m.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_m.transAxes)
                        
                safe_tag = tag[:3].replace("+", "")
                plt.suptitle(f'{tag} - Most Confident Mistakes', y=1.01, fontsize=14)
                plt.tight_layout()
                save_fig(f'fig_s9_errors_{safe_tag}.png')
            else:
                print("  [INFO] segment list not available - skipping waveform plot.")
        else:
            print("  [INFO] librosa not installed or dataset unavailable - skipping waveform plot.")

    # 10. COMPUTATIONAL ANALYSIS
    print("\n" + "=" * 60)
    print("  SECTION 10: Computational Analysis")
    print("=" * 60)

    comp_rows = []
    for name, m, l in [
        ("CNN", cnn_model, test_loader),
        ("CNN+LSTM", seg_model, test_seg_loader),
    ]:
        if m is None:
            print(f"[SKIP] {name} not available")
            continue
        total, trainable = count_params(m)
        size_mb = model_size_mb(m)
        if l is not None:
            avg_ms, throughput = measure_inference(m, l, DEVICE)
        else:
            avg_ms, throughput = np.nan, np.nan
            
        row = {
            'Model': name,
            'Total Params': f'{total:,}',
            'Trainable Params': f'{trainable:,}',
            'Size (MB)': size_mb,
            'Avg Inference (ms/batch)': avg_ms,
            'Throughput (samples/s)': throughput,
        }
        try:
            if 'cuda' in str(DEVICE):
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                row['GPU Mem Used (MB)'] = round(mem.used / 1e6, 1)
        except Exception:
            row['GPU Mem Used (MB)'] = 'N/A'
        comp_rows.append(row)

    if comp_rows:
        if HAS_PD:
            comp_df = pd.DataFrame(comp_rows).set_index('Model')
            print(comp_df.to_string())
            numeric_cols = ['Size (MB)', 'Avg Inference (ms/batch)', 'Throughput (samples/s)']
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(14, 5))
            
            for ax, col in zip(axes, numeric_cols):
                if col not in comp_df.columns:
                    continue
                vals = pd.to_numeric(comp_df[col], errors='coerce')
                ax.bar(comp_df.index, vals, color=PALETTE[:len(comp_df)], edgecolor='white')
                ax.set_title(col, fontsize=11)
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                
                for bar, val in zip(ax.patches, vals):
                    if not np.isnan(float(val)):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                            f'{float(val):.1f}', ha='center', va='bottom', fontsize=10
                        )
            plt.suptitle('Computational Efficiency Comparison', fontsize=14, y=1.01)
            plt.tight_layout()
            save_fig('fig_s10_computation.png')
    else:
        print("[SKIP] No models available for computational analysis.")

    # 11. STATISTICAL SIGNIFICANCE TESTING
    print("\n" + "=" * 60)
    print("  SECTION 11: Statistical Significance Testing (McNemar Test)")
    print("=" * 60)

    y_a = cnn_song_pred if cnn_song_pred is not None else cnn_seg_pred
    y_b = lstm_pred
    ref = cnn_song_true if cnn_song_true is not None else cnn_seg_true
    label_a = "CNN (song majority vote)" if cnn_song_pred is not None else "CNN (segment)"

    if y_a is not None and y_b is not None and ref is not None and HAS_STATSMODELS:
        min_len = min(len(y_a), len(y_b), len(ref))
        y_a = y_a[:min_len]
        y_b = y_b[:min_len]
        ref = ref[:min_len]
        cor_a = (y_a == ref)
        cor_b = (y_b == ref)
        
        table = np.array([
            [np.sum(cor_a & cor_b), np.sum(cor_a & ~cor_b)],
            [np.sum(~cor_a & cor_b), np.sum(~cor_a & ~cor_b)],
        ])
        
        # Box-style output
        print("+" + "-" * 54 + "+")
        print(f"| McNemar Test: {label_a:<15} vs CNN+LSTM   |")
        print("+" + "-" * 54 + "+")
        print(f"| Contingency Table:                                     |")
        print(f"|   Both correct     : {table[0, 0]:>4}                                |")
        print(f"|   A only correct   : {table[0, 1]:>4}                                |")
        print(f"|   B only correct   : {table[1, 0]:>4}                                |")
        print(f"|   Neither correct  : {table[1, 1]:>4}                                |")
        print("+" + "-" * 54 + "+")
        
        result = mcnemar(table, exact=True)
        print(f"| McNemar statistic = {result.statistic:>8.4f}                            |")
        print(f"| p-value           = {result.pvalue:>10.6f}                          |")
        if result.pvalue < 0.05:
            print("| [+] Difference is STATISTICALLY SIGNIFICANT (p < 0.05).   |")
        else:
            print("| [-] Difference is NOT statistically significant (p >= 0.05).|")
        print("+" + "-" * 54 + "+")
    else:
        print("[SKIP] Need both CNN and CNN+LSTM predictions + statsmodels installed.")

    # 12. FINAL COMPARISON DASHBOARD
    print("\n" + "=" * 60)
    print("  SECTION 12: Final Comparison Dashboard")
    print("=" * 60)

    if metrics_df is not None and HAS_PD:
        perf_cols = [
            'Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)',
            'Balanced Accuracy', "Cohen's Kappa", 'MCC', 'ROC-AUC (macro OvR)'
        ]
        perf_df = metrics_df[[c for c in perf_cols if c in metrics_df.columns]].copy()
        print("\n=== Final Performance Summary Dashboard ===")
        print(perf_df.round(4).to_string())

        # Radar chart
        radar_cols = [
            c for c in ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Balanced Accuracy', "Cohen's Kappa", 'MCC']
            if c in perf_df.columns
        ]
        N = len(radar_cols)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, (model_lbl, row) in enumerate(perf_df.iterrows()):
            vals = row[radar_cols].values.tolist() + [row[radar_cols[0]]]
            ax.plot(angles, vals, lw=2, color=PALETTE[i % len(PALETTE)], label=model_lbl)
            ax.fill(angles, vals, alpha=0.12, color=PALETTE[i % len(PALETTE)])
            
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_cols, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison - Radar Chart', pad=20, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        save_fig('fig_s12_radar.png')

        # Side-by-side bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(radar_cols))
        w = 0.8 / len(perf_df)
        
        for i, (model_lbl, row) in enumerate(perf_df.iterrows()):
            ax.bar(
                x + i * w, row[radar_cols].values, w, label=model_lbl,
                color=PALETTE[i % len(PALETTE)], edgecolor='white', linewidth=0.5
            )
        ax.set_xticks(x + w * (len(perf_df) - 1) / 2)
        ax.set_xticklabels(radar_cols, rotation=20, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Final Model Comparison - All Key Metrics')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        save_fig('fig_s12_final_bar.png')
    else:
        print("[SKIP] metrics_df not available (need pandas + both models).")

    print(f"\n{'=' * 60}")
    print(f"  All evaluation plots saved to: {PLOT_DIR}")
    print(f"{'=' * 60}")
    
    return metrics_df

