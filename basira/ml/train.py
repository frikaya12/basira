"""Entraînement du modèle de tri sur une base publique (format APTOS 2019).

Exemple :
    python -m basira.ml.train --csv data/aptos/train.csv --images data/aptos/train_images \
        --out models/basira-v0.pt --epochs 10

Le meilleur modèle (selon l'AUC de validation) est enregistré avec son seuil
de décision, choisi pour atteindre la sensibilité cible sur la validation.
Ce seuil devra être revalidé sur des photos marocaines avant tout usage réel.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from basira.ml.dataset import FundusDataset, read_labels, stratified_split
from basira.ml.metrics import roc_auc, screening_metrics, threshold_for_sensitivity
from basira.ml.model import ARCHITECTURES, build_model


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne le modèle de tri Basira (rétinopathie à adresser).")
    p.add_argument("--csv", required=True, help="CSV avec id_code et diagnosis (format APTOS)")
    p.add_argument("--images", required=True, help="Dossier des images")
    p.add_argument("--out", default="models/basira.pt", help="Chemin du modèle à enregistrer")
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHITECTURES)
    p.add_argument("--no-pretrained", action="store_true", help="Partir de poids aléatoires (tests hors ligne)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--target-sensitivity", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=0, help="N'utiliser que N images (essai rapide)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, labels, losses = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        losses.append(loss_fn(logits, y.to(device)).item() * len(y))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    n = sum(len(lab) for lab in labels)
    return np.concatenate(probs), np.concatenate(labels), sum(losses) / max(n, 1)


def main(argv=None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = read_labels(args.csv)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    train_idx, val_idx = stratified_split(df["referable"].to_numpy(), args.val_fraction, args.seed)
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    n_pos = int(train_df["referable"].sum())
    n_neg = len(train_df) - n_pos
    print(f"Images : {len(train_df)} entraînement ({n_pos} à adresser), {len(val_df)} validation")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("Il faut des exemples des deux classes dans l'entraînement.")

    device = pick_device()
    print(f"Calcul sur : {device}")
    train_loader = DataLoader(FundusDataset(train_df, args.images, args.img_size, train=True, seed=args.seed),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(FundusDataset(val_df, args.images, args.img_size), batch_size=args.batch_size, num_workers=args.workers)

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    # Les malades sont minoritaires : on pèse davantage leurs erreurs.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_score, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        probs, labels, val_loss = predict(model, val_loader, device)
        threshold = threshold_for_sensitivity(labels, probs, args.target_sensitivity)
        metrics = screening_metrics(labels, probs, threshold)
        auc = roc_auc(labels, probs)
        score = auc if auc is not None else -val_loss
        history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 4), "val_loss": round(val_loss, 4), **metrics})
        print(f"Époque {epoch}/{args.epochs} · perte {running / max(seen, 1):.3f} · val {val_loss:.3f} · "
              f"AUC {metrics['auc']} · sensibilité {metrics['sensibilite']} · spécificité {metrics['specificite']} "
              f"· {time.time() - start:.0f} s")

        if score > best_score:
            best_score = score
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "img_size": args.img_size,
                "threshold": float(threshold),
                "target_sensitivity": args.target_sensitivity,
                "metrics": {k: v for k, v in metrics.items() if v is not None},
                "epoch": epoch,
                "created": datetime.now().isoformat(timespec="seconds"),
            }, out)

    report = {"model": str(out), "best_epoch": max(history, key=lambda h: h["auc"] or -1)["epoch"], "history": history}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modèle enregistré : {out}")
    return report


if __name__ == "__main__":
    main()
