"""
train.py — Training loop + zero-shot root transfer evaluation

TRAINING:
  Standard discrete flow matching: sample t ~ U[0,1], mask graph with
  forward process at time t, predict clean graph.

EVALUATION — the key experiment:
  Zero-shot root transfer: hold out N_HOLDOUT root IDs during training.
  At test time:
    (a) Seen-root eval: reconstruct words for seen roots → sanity check
    (b) Zero-shot eval: given an UNSEEN root's consonants (never in training),
        generate template distribution and compare to ground truth.

  Metrics:
    - Root accuracy: % of root consonants correctly recovered
    - Template accuracy: % of template slot types correctly recovered
    - Edge accuracy: % of interdigitation edges correctly predicted
    - Full-graph accuracy: all three correct simultaneously
    - Template-only accuracy: when conditioning on root, recover template
"""

import os
import json
import random
import argparse
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import build_synthetic_dataset, build_dataset, SYRIAC_CONSONANT_LIST
from graph import (
    build_graph, collate_fn, BipartiteMorphGraph,
    CONSONANT_VOCAB, SLOT_VOCAB, EDGE_VOCAB, SYRIAC_CONSONANT_LIST,
    N_CONSONANTS, N_SLOTS, MAX_ROOT_LEN, MAX_TEMPL_LEN
)
from model import BipartiteDeFoG, mask_graph, compute_loss, sample


# ── Dataset ───────────────────────────────────────────────────────────────

class MorphDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.graphs = []
        skipped = 0
        for item in items:
            g = build_graph(item)
            if g is not None:
                self.graphs.append(g)
            else:
                skipped += 1
        print(f"  Dataset: {len(self.graphs)} valid graphs ({skipped} skipped)")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def split_by_root(
    dataset_items: list[dict],
    holdout_frac: float = 0.15,
    seed: int = 42
) -> tuple[list, list, list]:
    """
    Split dataset such that held-out roots are completely absent from train.
    Returns train_items, val_items, zeroshot_items.
    Zero-shot items have root IDs not seen in training.
    """
    random.seed(seed)
    root_ids = sorted(set(d['root_id'] for d in dataset_items))
    n_holdout = max(1, int(len(root_ids) * holdout_frac))
    holdout_ids = set(random.sample(root_ids, n_holdout))
    seen_ids = set(root_ids) - holdout_ids

    seen_items = [d for d in dataset_items if d['root_id'] in seen_ids]
    zeroshot_items = [d for d in dataset_items if d['root_id'] in holdout_ids]

    # Val split from seen roots
    random.shuffle(seen_items)
    n_val = max(1, int(len(seen_items) * 0.1))
    val_items = seen_items[:n_val]
    train_items = seen_items[n_val:]

    print(f"  Split: {len(train_items)} train / {len(val_items)} val / {len(zeroshot_items)} zero-shot")
    print(f"  Roots: {len(seen_ids)} seen / {len(holdout_ids)} held out")
    return train_items, val_items, zeroshot_items


# ── Trainer ───────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: BipartiteDeFoG,
        train_loader: DataLoader,
        val_loader: DataLoader,
        zeroshot_items: list[dict],
        device: torch.device,
        lr: float = 3e-4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.zeroshot_items = zeroshot_items
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = []

    def batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self) -> dict:
        self.model.train()
        total_losses = defaultdict(float)
        n_batches = 0

        for batch in self.train_loader:
            batch = self.batch_to_device(batch)
            B = batch['root_nodes'].shape[0]

            # Sample time uniformly
            t = torch.rand(B, device=self.device)

            # Forward masking
            noisy_batch = mask_graph(batch, t)

            # Model forward
            out = self.model(noisy_batch, t)

            # Loss against CLEAN targets
            losses = compute_loss(out, batch)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k, v in losses.items():
                total_losses[k] += v.item()
            n_batches += 1

        self.scheduler.step()
        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_losses = defaultdict(float)
        n_batches = 0

        for batch in loader:
            batch = self.batch_to_device(batch)
            B = batch['root_nodes'].shape[0]
            t = torch.rand(B, device=self.device)
            noisy_batch = mask_graph(batch, t)
            out = self.model(noisy_batch, t)
            losses = compute_loss(out, batch)
            for k, v in losses.items():
                total_losses[k] += v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def zero_shot_eval(self, n_samples: int = 50) -> dict:
        """
        Zero-shot root transfer experiment.

        For each held-out root:
          1. Take its ground-truth (root, template) pair
          2. Present the ROOT to the model as conditioning
          3. Sample the TEMPLATE from scratch
          4. Compare predicted template to ground truth

        This tests whether the model can generalise to novel root consonant
        combinations it has never seen during training.
        """
        if not self.zeroshot_items:
            return {}

        self.model.eval()
        items = random.sample(self.zeroshot_items, min(n_samples, len(self.zeroshot_items)))

        correct_root = 0
        correct_templ = 0
        correct_edge = 0
        correct_full = 0
        n_evaluated = 0

        for item in items:
            g = build_graph(item)
            if g is None:
                continue

            # Build a batch of 1
            batch = collate_fn([g])
            batch = self.batch_to_device(batch)

            # Mode: condition on ROOT, generate TEMPLATE
            # This is the hard zero-shot case:
            #   root consonants are novel (unseen during training)
            #   model must infer what template pattern they'd appear in
            result = sample(
                self.model,
                root_mask=batch['root_mask'],
                templ_mask=batch['templ_mask'],
                device=self.device,
                n_steps=20,
                condition_root=batch['root_nodes'],   # give it the unseen root
                condition_templ=None,                  # generate template
            )

            # Compare predictions to ground truth
            r_mask = batch['root_mask'][0]
            t_mask = batch['templ_mask'][0]
            re_mask = r_mask.unsqueeze(1) & t_mask.unsqueeze(0)

            root_acc = (result['root_nodes'][0][r_mask] == batch['root_nodes'][0][r_mask]).float().mean()
            templ_acc = (result['templ_nodes'][0][t_mask] == batch['templ_nodes'][0][t_mask]).float().mean()
            edge_acc = (result['edges'][0][re_mask] == batch['edges'][0][re_mask]).float().mean()

            correct_root += root_acc.item()
            correct_templ += templ_acc.item()
            correct_edge += edge_acc.item()
            correct_full += float(root_acc > 0.99 and templ_acc > 0.99 and edge_acc > 0.99)
            n_evaluated += 1

        if n_evaluated == 0:
            return {}

        return {
            'zs_root_acc':  correct_root / n_evaluated,
            'zs_templ_acc': correct_templ / n_evaluated,
            'zs_edge_acc':  correct_edge / n_evaluated,
            'zs_full_acc':  correct_full / n_evaluated,
            'n_zeroshot':   n_evaluated,
        }

    def train(self, n_epochs: int = 50, eval_every: int = 5) -> None:
        print(f"\nTraining BipartiteDeFoG for {n_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        best_val_loss = float('inf')

        for epoch in range(1, n_epochs + 1):
            train_losses = self.train_epoch()
            log = {'epoch': epoch, **{f'train_{k}': v for k, v in train_losses.items()}}

            if epoch % eval_every == 0:
                val_losses = self.eval_epoch(self.val_loader)
                zs_metrics = self.zero_shot_eval()
                log.update({f'val_{k}': v for k, v in val_losses.items()})
                log.update(zs_metrics)

                val_total = val_losses.get('total', float('inf'))
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    self.save_checkpoint('best.pt')

                print(
                    f"Epoch {epoch:3d} | "
                    f"Train {train_losses['total']:.3f} "
                    f"(R={train_losses['root']:.3f} T={train_losses['templ']:.3f} E={train_losses['edge']:.3f}) | "
                    f"Val {val_losses['total']:.3f} | "
                    f"ZS-templ {zs_metrics.get('zs_templ_acc', 0):.3f} "
                    f"ZS-full {zs_metrics.get('zs_full_acc', 0):.3f}"
                )
            else:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train {train_losses['total']:.3f} "
                    f"(R={train_losses['root']:.3f} T={train_losses['templ']:.3f} E={train_losses['edge']:.3f})"
                )

            self.history.append(log)

        self.save_checkpoint('final.pt')
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. Best val loss: {best_val_loss:.3f}")

    def save_checkpoint(self, name: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
        }, self.checkpoint_dir / name)

    def load_checkpoint(self, name: str):
        ckpt = torch.load(self.checkpoint_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt.get('history', [])


# ── Qualitative Inspection ────────────────────────────────────────────────

def decode_graph(result: dict, idx: int = 0) -> dict:
    """Convert a tensor graph back to human-readable form."""
    INV_CONS = {v: k for k, v in CONSONANT_VOCAB.items()}
    INV_SLOT = {v: k for k, v in SLOT_VOCAB.items()}
    INV_EDGE = {v: k for k, v in EDGE_VOCAB.items()}

    root_nodes = result['root_nodes'][idx]
    templ_nodes = result['templ_nodes'][idx]
    edges = result['edges'][idx]
    root_mask = result['root_mask'][idx]
    templ_mask = result['templ_mask'][idx]

    roots = [INV_CONS.get(c.item(), '?') for c, m in zip(root_nodes, root_mask) if m]
    slots = [INV_SLOT.get(s.item(), '?') for s, m in zip(templ_nodes, templ_mask) if m]

    edge_list = []
    for ri, rm in enumerate(root_mask):
        if not rm:
            continue
        for ti, tm in enumerate(templ_mask):
            if not tm:
                continue
            e = edges[ri, ti].item()
            if e != EDGE_VOCAB.get('no_edge', 0):
                edge_list.append(f"R[{ri}]→T[{ti}]:{INV_EDGE.get(e,'?')}")

    return {'roots': roots, 'slots': slots, 'edges': edge_list}


def print_generation_example(
    model: BipartiteDeFoG,
    item: dict,
    device: torch.device,
    mode: str = 'root_conditioned'
):
    """Print a qualitative example of generation."""
    g = build_graph(item)
    if g is None:
        print("  (graph construction failed)")
        return

    batch = collate_fn([g])
    batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    condition_root = batch_d['root_nodes'] if mode == 'root_conditioned' else None
    condition_templ = batch_d['templ_nodes'] if mode == 'templ_conditioned' else None

    result = sample(model, batch_d['root_mask'], batch_d['templ_mask'], device,
                    condition_root=condition_root, condition_templ=condition_templ)
    result['root_mask'] = batch_d['root_mask']
    result['templ_mask'] = batch_d['templ_mask']

    truth = decode_graph(batch_d)
    pred = decode_graph(result)

    print(f"  Surface:    {item['surface']}")
    print(f"  Root:       {item['root_syriac']}  {item['root_consonants']}")
    print(f"  Template:   {item.get('template_name', '?')}  {item['template']}")
    print(f"  True slots: {truth['slots']}")
    print(f"  Pred slots: {pred['slots']}")
    print(f"  True edges: {truth['edges']}")
    print(f"  Pred edges: {pred['edges']}")
    print()
