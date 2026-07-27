import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageCsvDataset(Dataset):
    def __init__(self, frame, image_col, image_root, target_col=None, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.image_root = Path(image_root)
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(self.image_root / str(row[self.image_col])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_col:
            return image, int(row[self.target_col])
        return image


def build_model(num_classes, pretrained):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    transform = weights.transforms() if weights else transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


def predict_proba(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            logits = model(batch.to(device))
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-col", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    classes = np.sort(train[args.target].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    train["_label"] = train[args.target].map(class_to_idx)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros((len(train), len(classes)))
    test_pred = np.zeros((len(test), len(classes)))

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, train["_label"]), start=1):
        model, base_transform = build_model(len(classes), args.pretrained)
        model.to(device)
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(), base_transform])
        train_ds = ImageCsvDataset(train.iloc[tr_idx], args.image_col, args.image_root, "_label", train_transform)
        valid_ds = ImageCsvDataset(train.iloc[va_idx], args.image_col, args.image_root, "_label", base_transform)
        test_ds = ImageCsvDataset(test, args.image_col, args.image_root, None, base_transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), labels)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            print(f"fold={fold} epoch={epoch} loss={np.mean(losses):.5f}")

        oof[va_idx] = predict_proba(model, valid_loader, device)
        test_pred += predict_proba(model, test_loader, device) / args.folds
        print(f"fold {fold} done")

    print(f"cv_logloss={log_loss(train['_label'], oof):.6f}")
    print(f"cv_accuracy={accuracy_score(train['_label'], np.argmax(oof, axis=1)):.6f}")

    out = pd.DataFrame()
    out[args.id_col or "id"] = test[args.id_col] if args.id_col else np.arange(len(test))
    if len(classes) == 2:
        out[args.target] = test_pred[:, 1]
    else:
        for i, cls in enumerate(classes):
            out[f"proba_{cls}"] = test_pred[:, i]
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

