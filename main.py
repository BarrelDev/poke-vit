# Goal: Use PyTorch to train a Vision Transformer (ViT) model to classify images of Pokemon across generations and artstyles.
from pathlib import Path
import argparse
import random
import csv
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim

# Simple ViT components
class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        # x: (B, N, D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=3, num_classes=1000, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]  # class token
        return self.head(cls)

# Training / dataset utilities
def make_dataloaders(data_dir, img_size=128, batch_size=16, val_split=0.2, seed=42, num_workers=4):
    """Create stratified train/val dataloaders from an ImageFolder.

    Tries to use sklearn.model_selection.train_test_split with stratify; if sklearn
    is not available, falls back to a simple per-class split.
    """
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tfms)

    # Build stratified split indices
    targets = [s[1] for s in dataset.samples]
    indices = list(range(len(targets)))
    try:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=targets, random_state=seed)
    except Exception:
        # fallback: per-class split
        from collections import defaultdict
        idx_by_class = defaultdict(list)
        for i, t in enumerate(targets):
            idx_by_class[t].append(i)
        train_idx = []
        val_idx = []
        random.seed(seed)
        for cls, idxs in idx_by_class.items():
            random.shuffle(idxs)
            n_val = max(1, int(len(idxs) * val_split))
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, dataset.classes


class CSVDataset(torch.utils.data.Dataset):
    """Dataset built from a CSV list of (image_path,label).

    `entries` should be a list of (image_rel_path, label) where image_rel_path
    is relative to `root`.
    `class_to_idx` is a mapping label->index to ensure consistent encoding.
    """

    def __init__(self, entries, root, class_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        for rel, lbl in entries:
            path = self.root / rel
            idx = class_to_idx[lbl]
            self.samples.append((str(path), idx))
        # build classes list
        self.classes = [None] * len(class_to_idx)
        for k, v in class_to_idx.items():
            self.classes[v] = k

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def load_csv_entries(csv_path):
    entries = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            entries.append((r['image_path'], r['label']))
    return entries


def make_dataloaders_from_csv(train_csv, val_csv, dataset_root, img_size=128, batch_size=16, num_workers=4):
    train_entries = load_csv_entries(train_csv)
    val_entries = load_csv_entries(val_csv)
    # build class list from both splits to ensure consistent mapping
    labels = sorted({lbl for _, lbl in (train_entries + val_entries)})
    class_to_idx = {c: i for i, c in enumerate(labels)}

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = CSVDataset(train_entries, dataset_root, class_to_idx, transform=train_tfms)
    val_ds = CSVDataset(val_entries, dataset_root, class_to_idx, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def train_one_epoch(model, opt, criterion, loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss/total, correct/total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a small ViT on a Pokemon dataset')
    parser.add_argument('--data', default=None, help='Path to dataset root (ImageFolder)')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', default='vit_pokemon.pth', help='Path to save checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained ViT from timm if available')
    parser.add_argument('--train-csv', help='Optional train CSV (image_path,label) to load instead of ImageFolder')
    parser.add_argument('--val-csv', help='Optional val CSV (image_path,label) to load instead of ImageFolder')
    parser.add_argument('--dataset-root', default='pokemon-dataset-1000', help='Root directory that image paths in CSV are relative to')
    args = parser.parse_args()

    # choose sensible default dataset path if not provided
    default_new = Path('pokemon-dataset-1000') / 'dataset'
    if args.train_csv and args.val_csv:
        # Use CSV-based loaders
        data_dir = None
    elif args.data:
        data_dir = args.data
    elif default_new.exists():
        data_dir = str(default_new)
    else:
        data_dir = "data/pokemon"

    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.train_csv and args.val_csv:
        train_loader, val_loader, classes = make_dataloaders_from_csv(args.train_csv, args.val_csv, args.dataset_root, img_size=img_size, batch_size=batch_size, num_workers=args.num_workers)
    else:
        train_loader, val_loader, classes = make_dataloaders(data_dir, img_size, batch_size, val_split=args.val_split, seed=args.seed, num_workers=args.num_workers)
    num_classes = len(classes)

    if args.pretrained:
        try:
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)
        except Exception as e:
            print('timm not available or model creation failed, falling back to local VisionTransformer:', e)
            model = VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=256, depth=6, num_heads=8).to(device)
    else:
        model = VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=256, depth=6, num_heads=8).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, opt, criterion, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    torch.save(model.state_dict(), args.out)