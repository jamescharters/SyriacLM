"""
run.py — Entry point for Syriac Bipartite DeFoG

Run as a package from the repository root:
    python -m defog.run                 # real SEDRA IV data (local corpora cache)
    python -m defog.run --synthetic     # offline synthetic toy data
    python -m defog.run --epochs 100    # longer training
    python -m defog.run --eval          # eval only (loads best.pt)
    python -m defog.run --demo          # print qualitative generation examples
"""

import argparse
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader

from .data import build_synthetic_dataset, build_dataset
from .graph import collate_fn
from .model import BipartiteDeFoG
from .train import MorphDataset, split_by_root, Trainer, print_generation_example


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Syriac Bipartite DeFoG')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use the offline synthetic generator instead of the local SEDRA cache')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force a fresh scan of the SEDRA cache (ignore the memoised triples)')
    parser.add_argument('--max-roots', type=int, default=150,
                        help='Max roots to draw from the SEDRA cache')
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
    parser.add_argument('--no-morph', action='store_true',
                        help='Ablation: disable morphological-feature conditioning '
                             '(makes root->template underdetermined)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    if args.synthetic:
        print("\nUsing synthetic data (offline toy generator)")
        raw_data = build_synthetic_dataset(n_samples=2000)
    else:
        try:
            print("\nLoading real SEDRA IV data from the local corpora cache...")
            raw_data = build_dataset(max_roots=args.max_roots, rebuild=args.rebuild,
                                     verbose=True)
        except FileNotFoundError as e:
            print(f"  {e}")
            print("  Falling back to synthetic data (run -m corpora.sedra_scrape "
                  "to enable real data).")
            raw_data = build_synthetic_dataset(n_samples=2000)
        if len(raw_data) < 50:
            print("  Very few items from SEDRA; supplementing with synthetic data.")
            raw_data += build_synthetic_dataset(n_samples=500)

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
        use_morph=not args.no_morph,
    )
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters "
          f"| morph-conditioning: {'OFF (ablation)' if args.no_morph else 'ON'}")

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
    # Per-slot majority baseline: predict the single most common slot label for
    # every slot. The compositional claim is that conditioning on (root,
    # features) beats this marginal-pattern floor on roots never seen in training.
    slot_counts = Counter(s for it in train_items for s in it['template'])
    maj_label = slot_counts.most_common(1)[0][0] if slot_counts else None
    zs_templates = [it['template'] for it in zeroshot_items if it['template']]
    base_slot = (sum(sum(s == maj_label for s in t) / len(t) for t in zs_templates)
                 / len(zs_templates)) if zs_templates else 0.0
    base_exact = (sum(all(s == maj_label for s in t) for t in zs_templates)
                  / len(zs_templates)) if zs_templates else 0.0

    print("\n── Zero-Shot Root Transfer Evaluation ──")
    print(f"  (novel roots, unseen in training; generation conditioned on "
          f"root + morphological features)")
    zs = trainer.zero_shot_eval(n_samples=200)
    if zs:
        print(f"  Samples evaluated:      {zs['n_zeroshot']}")
        print(f"  Root accuracy:          {zs['zs_root_acc']:.3f}  (conditioned, should be ~1.0)")
        print(f"  Template per-slot acc:  {zs['zs_templ_acc']:.3f}  (vs majority baseline {base_slot:.3f})")
        print(f"  Template EXACT match:   {zs['zs_templ_exact']:.3f}  (whole pattern; baseline {base_exact:.3f})")
        print(f"  Edge (interdigitation): {zs['zs_edge_acc']:.3f}  (binds unseen root into the pattern)")
        print(f"  Full graph accuracy:    {zs['zs_full_acc']:.3f}  (root+template+edges all correct)")
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
