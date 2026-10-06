"""Loads a trained checkpoint (logreg joblib or neural .pt) behind one
Predictor interface that returns P(positive) for raw review text. The CLI
prints the label, confidence, and both class probabilities for one review;
evaluate.py reuses the same Predictor so both paths preprocess identically."""

import argparse
import json

import joblib
import numpy as np
import torch

from dataset import EncodedReviews, make_loader
from models import build_neural_model
from tokenizer import TextPreprocessor, WordTokenizer
from utils import (LABEL_NAMES, apply_smoke_overrides, checkpoint_paths, get_device,
                   get_model_cfg, get_paths, load_config)


class Predictor:
    def __init__(self, cfg, model_name, paths, device=None):
        self.model_name = model_name
        self.model_cfg = get_model_cfg(cfg, model_name)
        self.type = self.model_cfg["type"]
        ck = checkpoint_paths(paths, model_name, self.type)
        if self.type == "logreg":
            if not ck["model"].exists():
                raise FileNotFoundError(f"{ck['model']} not found. Train '{model_name}' first.")
            self.device = torch.device("cpu")
            bundle = joblib.load(ck["model"])
            self.pipe = bundle["pipeline"]
            self.preprocessor = TextPreprocessor(bundle["preprocessing_cfg"])
        else:
            if not ck["best"].exists():
                raise FileNotFoundError(f"{ck['best']} not found. Train '{model_name}' first.")
            self.device = device or get_device()
            state = torch.load(ck["best"], map_location=self.device, weights_only=True)
            self.tokenizer = WordTokenizer(state["vocab"])
            self.max_length = state["tokenizer_cfg"]["max_length"]
            # Why: rebuild with the preprocessing and architecture saved inside the
            # checkpoint, so a later config edit cannot silently change a trained model.
            self.preprocessor = TextPreprocessor(state["preprocessing_cfg"])
            self.model = build_neural_model(state["model_cfg"], len(self.tokenizer)).to(self.device)
            self.model.load_state_dict(state["model_state"])
            self.model.eval()
            self.best_epoch = state.get("best_epoch")
            self.eval_batch_size = self.model_cfg.get("eval_batch_size", 512)
            self.seed = state["seed"]

    @torch.no_grad()
    def predict_proba_tokens(self, token_strings):
        """P(positive) for already-preprocessed token strings."""
        token_strings = list(token_strings)
        if self.type == "logreg":
            return self.pipe.predict_proba(token_strings)[:, 1]
        ds = EncodedReviews(token_strings, np.zeros(len(token_strings)), self.tokenizer, self.max_length)
        loader = make_loader(ds, self.eval_batch_size, False, self.seed)
        out = []
        for ids, mask, _ in loader:
            logits = self.model(ids.to(self.device), mask.to(self.device))
            out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def predict_texts(self, texts):
        tokens = [self.preprocessor.to_string(t) for t in texts]
        p_pos = self.predict_proba_tokens(tokens)
        results = []
        for p in p_pos:
            label = int(p >= 0.5)
            results.append({
                "label": label,
                "label_name": LABEL_NAMES[label],
                "confidence": float(max(p, 1 - p)),
                "probabilities": {"negative": float(1 - p), "positive": float(p)},
            })
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict sentiment for one review")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--smoke", action="store_true", help="use the smoke-run checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    pred = Predictor(cfg, args.model, get_paths(cfg, args.smoke))
    r = pred.predict_texts([args.text])[0]
    print(f"model:         {args.model}")
    print(f"text:          {args.text}")
    print(f"label:         {r['label']} ({r['label_name']})")
    print(f"confidence:    {r['confidence']:.4f}")
    print(f"probabilities: {json.dumps(r['probabilities'])}")
