"""
train.py — Fine-tune microsoft/cvt-13 on the archive/data dataset.

Archive structure expected:
    archive/
      data/
        fake/   ← AI-generated / fake food images
        real/   ← Genuine / real food images

PyTorch ImageFolder maps classes alphabetically:
    0 → fake
    1 → real

Run from the project root:
    python src/train.py
    python src/train.py --data_dir archive/data --epochs 10 --save_path models
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ── make sure src/ is importable when run from project root ──────────────────
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model import get_model
from custom_dataset import get_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy) on *loader*."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion,
    scheduler,
    total_epochs: int,
    save_path: str,
):
    """Train for *total_epochs* and save the best-val-accuracy checkpoint."""
    # Use the new torch.amp API (avoids DeprecationWarning on PyTorch ≥ 2.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(save_path, "best_model.pth")

    print(f"\n{'='*60}")
    print(f"  Dataset  : {len(train_loader.dataset)} train | {len(val_loader.dataset)} val")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {total_epochs}")
    _classes = getattr(train_loader.dataset, "subset", train_loader).dataset.classes if hasattr(train_loader.dataset, "subset") else train_loader.dataset.dataset.classes
    print(f"  Classes  : {_classes}")   # fake=0, real=1
    print(f"{'='*60}\n")

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += inputs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc  = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # ── Save every epoch (optional, keeps last few) ──────────────────────
        epoch_path = os.path.join(save_path, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            epoch_path,
        )

        # ── Track best checkpoint ─────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "classes": getattr(train_loader.dataset, "subset", train_loader).dataset.classes if hasattr(train_loader.dataset, "subset") else train_loader.dataset.dataset.classes,
                },
                best_ckpt_path,
            )
            marker = "  [BEST]"
        else:
            marker = ""

        print(
            f"Epoch {epoch:>2}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}"
        )

    print(f"\nTraining complete.  Best val-acc = {best_val_acc:.4f}")
    print(f"Best model saved → {best_ckpt_path}")
    return best_ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Real vs Fake food-image classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("archive", "data"),
        help="Root folder with 'fake/' and 'real/' sub-folders (default: archive/data)",
    )
    parser.add_argument("--epochs",    type=int,   default=10,     help="Training epochs (default: 10)")
    parser.add_argument("--batch",     type=int,   default=16,     help="Batch size (default: 16)")
    parser.add_argument("--lr",        type=float, default=1e-4,   help="Learning rate (default: 1e-4)")
    parser.add_argument("--val_split", type=float, default=0.15,   help="Validation fraction (default: 0.15)")
    parser.add_argument(
        "--save_path",
        type=str,
        default="models",
        help="Directory to save checkpoints (default: models)",
    )
    args = parser.parse_args()

    # ── Sanity check ──────────────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: '{args.data_dir}'\n"
            "Make sure you run this script from the project root, e.g.:\n"
            "  python src/train.py"
        )
    for cls in ("fake", "real"):
        cls_path = os.path.join(args.data_dir, cls)
        if not os.path.isdir(cls_path):
            raise FileNotFoundError(f"Expected class folder not found: '{cls_path}'")

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset & splits ──────────────────────────────────────────────────────
    full_dataset = get_dataset(args.data_dir)
    print(f"Total images: {len(full_dataset)}")
    print(f"Class map   : {full_dataset.class_to_idx}")   # {'fake': 0, 'real': 1}

    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    base_train, base_val = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    from custom_dataset import get_train_transform, get_transform, TransformSubset
    train_ds = TransformSubset(base_train, get_train_transform())
    val_ds = TransformSubset(base_val, get_transform())

    num_workers = 0  # safer on Windows; increase to 2-4 if comfortable
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── Model, optimiser, loss ────────────────────────────────────────────────
    model     = get_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    best_path = train_model(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        device      = device,
        optimizer   = optimizer,
        criterion   = criterion,
        scheduler   = scheduler,
        total_epochs= args.epochs,
        save_path   = args.save_path,
    )

    print(f"\nTo use this model in the Flask app, make sure 'models/best_model.pth' exists.")
    print(f"Then run:  python app.py")


if __name__ == "__main__":
    main()