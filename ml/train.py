import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from torchvision import datasets, transforms

from ml.model import SimpleCNN, class_labels


class FilteredEMNIST(Dataset):
    def __init__(self, emnist_dataset, target_transform):
        self.dataset = emnist_dataset
        self.target_transform = target_transform
        # Filter to only letters: labels 10-35 (A-Z) and 36-61 (a-z)
        self.indices = [i for i, label in enumerate(emnist_dataset.targets) if 10 <= label <= 61]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.dataset[actual_idx]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def emnist_label_to_index(y: int) -> int:
    # EMNIST byclass: 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z
    # We'll collapse case: A/a->0, B/b->1, ..., Z/z->25
    if 10 <= int(y) <= 35:  # Uppercase A-Z
        return int(y) - 10
    elif 36 <= int(y) <= 61:  # Lowercase a-z  
        return int(y) - 36
    else:
        raise ValueError(f"Unexpected EMNIST label: {y}")


def mnist_label_to_index(y: int) -> int:
    return int(y) + 26  # MNIST digits 0..9 -> 26..35


def build_datasets(data_dir: str, augment: bool = True) -> Tuple[ConcatDataset, ConcatDataset]:
    train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
    if augment:
        aug = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.6),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.3),
            transforms.RandomRotation(degrees=10, fill=0),  # Add rotation for orientation robustness
        ]
        # Apply aug before ToTensor
        train_transforms = aug + train_transforms

    test_transforms = [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]

    # Load EMNIST byclass and filter to only letters (both upper and lowercase)
    emnist_full_train = datasets.EMNIST(
        root=data_dir,
        split='byclass',
        train=True,
        download=True,
        transform=transforms.Compose(train_transforms),
    )
    emnist_full_test = datasets.EMNIST(
        root=data_dir,
        split='byclass',
        train=False,
        download=True,
        transform=transforms.Compose(test_transforms),
    )
    
    # Create filtered datasets with case-collapsed labels
    emnist_train = FilteredEMNIST(emnist_full_train, emnist_label_to_index)
    emnist_test = FilteredEMNIST(emnist_full_test, emnist_label_to_index)

    mnist_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(train_transforms),
        target_transform=mnist_label_to_index,
    )
    mnist_test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(test_transforms),
        target_transform=mnist_label_to_index,
    )

    train_ds = ConcatDataset([emnist_train, mnist_train])
    test_ds = ConcatDataset([emnist_test, mnist_test])
    return train_ds, test_ds


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def train(args):
    set_seed(args.seed)

    # Prefer Apple MPS if available (macOS), else CUDA, else CPU
    device = torch.device(
        'mps' if torch.backends.mps.is_available() and not args.cpu else (
            'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        )
    )
    print(f"Using device: {device}")

    labels = class_labels()
    assert len(labels) == 36

    train_ds, test_ds = build_datasets(args.data, augment=not args.no_augment)

    # Split a validation set out of training for model selection
    val_size = int(0.05 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_subset, val_subset = random_split(train_ds, [train_size, val_size])

    # num_workers=0 for maximum portability across environments
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = SimpleCNN(num_classes=len(labels)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # OneCycleLR for rapid convergence and stable training
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=10.0,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_ckpt_path = os.path.join(os.path.dirname(args.out), 'checkpoint.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, val_acc = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{args.epochs} - lr={current_lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping on validation accuracy
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'val_acc': val_acc}, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    # Load best checkpoint if available
    if os.path.exists(best_ckpt_path):
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"Loaded best checkpoint with val_acc={state.get('val_acc', 0.0):.4f}")

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")

    # Export TorchScript
    model.eval()
    example = torch.randn(1, 1, 28, 28, device=device)
    _ = model(example)
    scripted = torch.jit.script(model)
    scripted.save(args.out)
    print(f"Saved TorchScript model to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 36-class EMNIST+MNIST model')
    parser.add_argument('--data', type=str, default='data', help='dataset directory')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--clip-grad', type=float, default=1.0, help='max grad norm; 0 disables')
    parser.add_argument('--early-stop', type=int, default=3, help='epochs without val improvement before stop; 0 disables')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='models/emnist36.pt')
    args = parser.parse_args()
    train(args)
