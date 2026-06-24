"""
run.py — Entry point for Syriac Bipartite DeFoG

Usage:
    python run.py                     # synthetic data, quick run
    python run.py --fetch             # fetch SEDRA API data (needs internet)
    python run.py --epochs 100        # longer training
    python run.py --eval              # eval only (loads best.pt)
    python run.py --demo              # print qualitative generation examples
"""

import argparse
import random
import torch
from torch.utils.data import DataLoader

from data import build_synthetic_dataset, build_dataset
from graph import collate_fn
from model import BipartiteDeFoG
from train import MorphDataset, split_by_root, Trainer, print_generation_example


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Syriac Bipartite DeFoG')
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch real SEDRA data (requires internet)')
    parser.add_argument('--max-roots', type=int, default=150,
                        help='Max roots to fetch from SEDRA')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-coupling', type=int, default=3,
                        help='Number of cross-attention coupling layers')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval', action='store_true',
                        help='Eval only (load best.pt)')
    parser.add_argument('--demo', action='store_true',
                        help='Print qualitative generation examples')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    if args.fetch:
        print("\nFetching real SEDRA data...")
        raw_data = build_dataset(max_roots=args.max_roots, verbose=True)
        if len(raw_data) < 50:
            print("  WARNING: Very few items from SEDRA. Supplementing with synthetic data.")
            raw_data += build_synthetic_dataset(n_samples=500)
    else:
        print("\nUsing synthetic data (run with --fetch for real SEDRA data)")
        raw_data = build_synthetic_dataset(n_samples=2000)

    print(f"\nTotal dataset: {len(raw_data)} items")

    # ── Split ────────────────────────────────────────────────────────────────
    train_items, val_items, zeroshot_items = split_by_root(raw_data, holdout_frac=0.15)

    train_ds = MorphDataset(train_items)
    val_ds   = MorphDataset(val_items)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = BipartiteDeFoG(
        d_model=args.d_model,
        n_heads=4,
        n_coupling_layers=args.n_coupling,
        dropout=0.1,
    )
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        zeroshot_items=zeroshot_items,
        device=device,
        lr=args.lr,
    )

    if args.eval or args.demo:
        print("\nLoading checkpoint...")
        try:
            trainer.load_checkpoint('best.pt')
        except FileNotFoundError:
            print("  No checkpoint found. Run without --eval first.")
            return
    else:
        trainer.train(n_epochs=args.epochs, eval_every=5)

    # ── Final zero-shot eval ─────────────────────────────────────────────────
    print("\n── Zero-Shot Root Transfer Evaluation ──")
    zs = trainer.zero_shot_eval(n_samples=100)
    if zs:
        print(f"  Samples evaluated:      {zs['n_zeroshot']}")
        print(f"  Root accuracy:          {zs['zs_root_acc']:.3f}  (should be ~1.0, root is conditioned)")
        print(f"  Template accuracy:      {zs['zs_templ_acc']:.3f}  (KEY metric — novel root → correct template)")
        print(f"  Edge accuracy:          {zs['zs_edge_acc']:.3f}  (interdigitation pattern)")
        print(f"  Full graph accuracy:    {zs['zs_full_acc']:.3f}  (all three correct)")
    else:
        print("  (no zero-shot items available)")

    # ── Qualitative examples ─────────────────────────────────────────────────
    if args.demo or True:   # always show a few examples
        print("\n── Qualitative Generation Examples ──")
        print("\n-- Seen root (root-conditioned template generation) --")
        for item in random.sample(val_items, min(3, len(val_items))):
            print_generation_example(model, item, device, mode='root_conditioned')

        print("\n-- Zero-shot root (novel root → inferred template) --")
        for item in random.sample(zeroshot_items, min(3, len(zeroshot_items))):
            print_generation_example(model, item, device, mode='root_conditioned')

    print("\nDone.")


if __name__ == '__main__':
    main()
