import argparse
import sys
from pathlib import Path
import torch.nn as nn

# Add the project root to python path to ensure imports work if run from anywhere
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.utils import print_header, pprint
from src.main_pipeline import run_training, build_data, interactive_menu
from src.evaluation.suite import run_full_evaluation
from src.prediction.predictor import (
    predict_genre, display_prediction, run_demo,
    _check_eval_ready, _load_saved_models
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Genre Classifier - CNN + Frozen CNN+LSTM on GTZAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                       # Interactive menu (recommended for demo)
  python main.py --train               # Train both models, full pipeline
  python main.py --train --no-hp       # Train without HP search (faster)
  python main.py --evaluate            # Evaluate + all 12 plot sections
  python main.py --predict song.wav    # Predict genre of an audio file
  python main.py --demo                # Demo on random test samples
        """
    )
    parser.add_argument("--train", action="store_true", help="Train both models")
    parser.add_argument("--no-hp", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--evaluate", action="store_true", help="Run full 12-section evaluation")
    parser.add_argument("--predict", metavar="FILE", help="Predict genre of an audio file")
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn", help="Model to use for prediction (cnn or lstm, default: cnn)")
    parser.add_argument("--demo", action="store_true", help="Demo mode")
    args = parser.parse_args()

    if args.train:
        run_training(skip_hp_search=args.no_hp)

    elif args.evaluate:
        _ok, _msg = _check_eval_ready()
        if not _ok:
            pprint(f"[red]{_msg}[/red]")
            sys.exit(1)
        raw_root = DATA_DIR / "genres_original"
        (_, _, _, _, _, _,
         _, _, test_ds, _, _, test_dl,
         _, _, test_seg_ds, _, _, test_seg_dl) = build_data(raw_root)
        cnn_model, seg_model, history, history_seg = _load_saved_models()
        run_full_evaluation(
            cnn_model, seg_model,
            test_ds, test_dl, test_seg_ds, test_seg_dl,
            history, history_seg,
            nn.CrossEntropyLoss(label_smoothing=0.1),
            nn.CrossEntropyLoss(label_smoothing=0.1),
        )

    elif args.predict:
        print_header()
        result = predict_genre(args.predict, model_type=args.model)
        display_prediction(result, args.predict)

    elif args.demo:
        run_demo()

    else:
        interactive_menu()

