#!/usr/bin/env python3
"""
Evaluate a trained ViT model on the `data/pokemon` dataset.

Usage:
  python test.py --model vit_pokemon.pth

Outputs:
  - Prints overall accuracy and per-class accuracy.
  - Optionally writes `predictions.csv` with columns: image_path,true_label,pred_label,prob
"""
import argparse
import csv
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from main import VisionTransformer


def make_test_loader(data_dir, img_size=224, batch_size=32, num_workers=4):
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tfms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, dataset.classes, dataset


def evaluate(model, loader, device, out_csv=None, classes=None):
    model.eval()
    correct = 0
    total = 0
    num_classes = len(classes)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    rows = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = softmax(logits)
            preds = probs.argmax(dim=1)
            for t, p, prob, inp in zip(labels.cpu().tolist(), preds.cpu().tolist(), probs.cpu().tolist(), getattr(loader.dataset, 'imgs', None) or []):
                # We will fill rows below per-batch using indices
                pass

            # accumulate
            for t, p in zip(labels.view(-1).cpu(), preds.view(-1).cpu()):
                conf[t.long(), p.long()] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Build per-image rows (iterate dataset to get paths)
    if out_csv:
        # Re-run inference per-sample to capture image paths and probabilities (small overhead)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'true_label', 'pred_label', 'prob'])
            model.eval()
            with torch.no_grad():
                for idx in range(len(loader.dataset)):
                    img, label = loader.dataset[idx]
                    img = img.unsqueeze(0).to(device)
                    logits = model(img)
                    prob = nn.functional.softmax(logits, dim=1)[0]
                    pred = int(prob.argmax().item())
                    p = float(prob[pred].item())
                    img_path = loader.dataset.imgs[idx][0] if hasattr(loader.dataset, 'imgs') else ''
                    writer.writerow([img_path, classes[label], classes[pred], f"{p:.4f}"])

    overall_acc = correct / total if total > 0 else 0.0
    per_class = {}
    for i, cname in enumerate(classes):
        total_i = int(conf[i, :].sum().item())
        correct_i = int(conf[i, i].item())
        per_class[cname] = (correct_i, total_i, (correct_i / total_i) if total_i > 0 else 0.0)

    return overall_acc, per_class, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_pokemon.pth', help='Path to model checkpoint')
    parser.add_argument('--data', default='data/pokemon', help='Path to dataset root (ImageFolder)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-csv', help='Optional CSV to write predictions')
    args = parser.parse_args()

    device = torch.device(args.device)

    loader, classes, dataset = make_test_loader(args.data, img_size=args.img_size, batch_size=args.batch_size)
    num_classes = len(classes)

    model = VisionTransformer(img_size=args.img_size, num_classes=num_classes, embed_dim=256, depth=6, num_heads=8)
    model.to(device)

    ckpt = torch.load(args.model, map_location=device)
    try:
        model.load_state_dict(ckpt)
    except Exception:
        # Try loading a state_dict that may have module prefix
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt)

    overall_acc, per_class, conf = evaluate(model, loader, device, out_csv=args.out_csv, classes=classes)

    print(f"Overall accuracy: {overall_acc*100:.2f}%")
    print("Per-class accuracy:")
    for cname, (correct_i, total_i, acc_i) in per_class.items():
        print(f"  {cname}: {correct_i}/{total_i} = {acc_i*100:.2f}%")

    # Optionally print confusion matrix summary for top classes
    print('\nConfusion matrix (rows=true, cols=pred):')
    print(conf.numpy())


if __name__ == '__main__':
    main()
