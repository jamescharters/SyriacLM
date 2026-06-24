"""
model.py — Bipartite DeFoG Architecture

THE CORE ML NOVELTY:
  Coupled discrete flow matching over two interdependent discrete graph components
  (root-consonant nodes R, template-slot nodes T) with ASYMMETRIC EQUIVARIANCE:
    - Root encoder: Deep Sets (permutation-INVARIANT over consonant set)
    - Template encoder: Transformer (position-SENSITIVE over slot sequence)
    - Coupling: cross-attention between R and T representations
    - Flow: discrete CTMC over joint categorical state

DISCRETE FLOW MATCHING (DeFoG-style):
  We define a continuous-time Markov chain (CTMC) over the joint state space
  (root_nodes, templ_nodes, edges). The flow model learns the rate matrix of
  this CTMC by predicting the clean data from a noisy (partially masked) state.

  The loss is the CTMC rate-matching loss:
    L = E_{t~U[0,1], x_t~q(x_t|x_0)} [ -log p_theta(x_0 | x_t, t) ]

  where q(x_t | x_0) is the forward masking process:
    At time t, each token is independently replaced by MASK with probability t,
    and kept as-is with probability (1-t).

  This is simpler than the full DeFoG formulation but captures the essential idea.
  The key extension over standard DeFoG is that we have TWO coupled categorical
  state vectors (root, template) that must be denoised jointly.

ASYMMETRIC EQUIVARIANCE in detail:
  Root nodes have no canonical ordering (ܟ-ܬ-ܒ and ܒ-ܬ-ܟ are the same root).
  The Deep Sets encoder achieves permutation invariance by:
    1. Embed each consonant independently: phi(c_i)
    2. Pool: h_R = MeanPool({phi(c_i)}) + MaxPool({phi(c_i)})
  This h_R is invariant to root consonant order by construction.

  Template nodes ARE ordered (slot 0 before slot 1 before slot 2).
  The Transformer encoder uses sinusoidal position embeddings:
    h_T = TransformerEncoder(embed(t_j) + pos_enc(j))
  This h_T is equivariant to template slot permutation IF we permute positions,
  but not invariant — the order matters.

  The cross-attention layer allows T to attend to the ROOT SET and R to attend
  to the TEMPLATE SEQUENCE, coupling the two flows.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import (
    N_CONSONANTS, N_SLOTS, N_EDGE_TYPES,
    MAX_ROOT_LEN, MAX_TEMPL_LEN,
    CONSONANT_VOCAB, SLOT_VOCAB, EDGE_VOCAB
)


# ── Positional Encoding ───────────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ── Deep Sets Root Encoder (permutation-invariant) ────────────────────────

class DeepSetsRootEncoder(nn.Module):
    """
    Encodes a SET of root consonants into a permutation-invariant representation.
    Per-element transform phi, then symmetric aggregation (mean + max).
    Output: per-element representations (for edge prediction) + global set summary.
    """
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(N_CONSONANTS, d_model)
        self.phi = nn.Sequential(
            *[block for _ in range(n_layers)
              for block in [nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)]]
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # mean + max concat
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, root_nodes: torch.Tensor, root_mask: torch.Tensor):
        """
        root_nodes: [B, R] int
        root_mask:  [B, R] bool
        Returns:
          per_elem: [B, R, D]   — per-consonant representations
          global:   [B, D]      — permutation-invariant global summary
        """
        x = self.embed(root_nodes)                     # [B, R, D]
        x = self.phi(x)                                # [B, R, D]

        # Mask padding before pooling
        mask_f = root_mask.float().unsqueeze(-1)       # [B, R, 1]
        x_masked = x * mask_f

        # Mean pool (over real nodes)
        n_real = root_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        mean_pool = x_masked.sum(dim=1) / n_real       # [B, D]

        # Max pool
        x_for_max = x_masked + (1 - mask_f) * (-1e9)
        max_pool = x_for_max.max(dim=1).values         # [B, D]

        global_rep = self.rho(torch.cat([mean_pool, max_pool], dim=-1))  # [B, D]
        return x, global_rep


# ── Transformer Template Encoder (position-sensitive) ────────────────────

class TransformerTemplateEncoder(nn.Module):
    """
    Encodes an ORDERED SEQUENCE of template slots with position sensitivity.
    Uses standard Transformer encoder with sinusoidal position embeddings.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(N_SLOTS, d_model)
        self.pos_enc = SinusoidalPosEnc(d_model, max_len=MAX_TEMPL_LEN + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, templ_nodes: torch.Tensor, templ_mask: torch.Tensor):
        """
        templ_nodes: [B, T] int
        templ_mask:  [B, T] bool (True = real, False = pad)
        Returns:
          per_slot: [B, T, D]
        """
        x = self.embed(templ_nodes)               # [B, T, D]
        x = self.pos_enc(x)                       # adds positional signal
        # TransformerEncoder expects src_key_padding_mask where True=IGNORE
        pad_mask = ~templ_mask                    # [B, T], True = pad
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return x


# ── Cross-Attention Coupling Layer ────────────────────────────────────────

class CrossAttentionCoupling(nn.Module):
    """
    Bidirectional cross-attention between root (set) and template (sequence).
    - Template attends to root set: allows slots to query which consonant they receive
    - Root attends to template: allows each consonant to know its morphological context
    This is where the asymmetric equivariance matters:
      - Attention from T→R sees an UNORDERED set (attention is already permutation-equivariant)
      - Attention from R→T sees an ORDERED sequence (positions are encoded in T)
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t_to_r = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.r_to_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.ff_r = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.ff_t = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, r_feats, t_feats, root_mask, templ_mask):
        """
        r_feats:    [B, R, D]
        t_feats:    [B, T, D]
        root_mask:  [B, R] bool (True = real)
        templ_mask: [B, T] bool (True = real)
        """
        # T attends to R (template slot queries root set)
        r_key_mask = ~root_mask   # True = ignore
        t_out, _ = self.t_to_r(t_feats, r_feats, r_feats, key_padding_mask=r_key_mask)
        t_feats = self.norm_t(t_feats + t_out)
        t_feats = t_feats + self.ff_t(t_feats)

        # R attends to T (root consonant queries template sequence)
        t_key_mask = ~templ_mask  # True = ignore
        r_out, _ = self.r_to_t(r_feats, t_feats, t_feats, key_padding_mask=t_key_mask)
        r_feats = self.norm_r(r_feats + r_out)
        r_feats = r_feats + self.ff_r(r_feats)

        return r_feats, t_feats


# ── Time Conditioning ────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Scalar time t ∈ [0,1] → D-dimensional embedding via sinusoidal + MLP."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )

    def forward(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        # t: [B]
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)    # [B, D/2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, D]
        return self.mlp(emb)                        # [B, D]


# ── Edge Predictor ────────────────────────────────────────────────────────

class EdgePredictor(nn.Module):
    """
    Predicts edge type for each (root_node_i, template_slot_j) pair.
    Input: concatenated r_i and t_j features.
    Output: logits over {no_edge, fills, affix, mask}
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_EDGE_TYPES),
        )

    def forward(self, r_feats: torch.Tensor, t_feats: torch.Tensor) -> torch.Tensor:
        """
        r_feats: [B, R, D]
        t_feats: [B, T, D]
        Returns: [B, R, T, E] edge logits
        """
        B, R, D = r_feats.shape
        T = t_feats.shape[1]
        r_exp = r_feats.unsqueeze(2).expand(B, R, T, D)   # [B, R, T, D]
        t_exp = t_feats.unsqueeze(1).expand(B, R, T, D)   # [B, R, T, D]
        return self.mlp(torch.cat([r_exp, t_exp], dim=-1)) # [B, R, T, E]


# ── BipartiteDeFoG (full model) ───────────────────────────────────────────

class BipartiteDeFoG(nn.Module):
    """
    Bipartite Discrete Flow model for Syriac morphological generation.

    Given a MASKED graph state (root_t, templ_t, edges_t) and time t,
    predicts the CLEAN graph (root_0, templ_0, edges_0).

    Architecture:
      1. Encode root SET with Deep Sets (permutation-invariant)
      2. Encode template SEQUENCE with Transformer (position-sensitive)
      3. Add time conditioning to both
      4. N layers of cross-attention coupling
      5. Predict clean root, template, edge distributions
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_coupling_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoders
        self.root_encoder = DeepSetsRootEncoder(d_model, n_layers=2)
        self.templ_encoder = TransformerTemplateEncoder(d_model, n_heads, n_layers=2, dropout=dropout)

        # Time embedding
        self.time_embed = TimeEmbedding(d_model)

        # Edge embedding (for noisy edge inputs)
        self.edge_embed = nn.Embedding(N_EDGE_TYPES, d_model)

        # Time conditioning projection
        self.time_proj_r = nn.Linear(d_model, d_model)
        self.time_proj_t = nn.Linear(d_model, d_model)

        # Cross-attention coupling layers
        self.coupling_layers = nn.ModuleList([
            CrossAttentionCoupling(d_model, n_heads, dropout)
            for _ in range(n_coupling_layers)
        ])

        # Output heads
        self.root_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_CONSONANTS),   # predicts clean consonant
        )
        self.templ_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_SLOTS),         # predicts clean slot type
        )
        self.edge_predictor = EdgePredictor(d_model)

    def forward(self, batch: dict, t: torch.Tensor) -> dict:
        """
        batch: dict with root_nodes, templ_nodes, edges, root_mask, templ_mask
        t:     [B] float in [0, 1]

        Returns dict of logits:
          root_logits:  [B, R, N_CONSONANTS]
          templ_logits: [B, T, N_SLOTS]
          edge_logits:  [B, R, T, N_EDGE_TYPES]
        """
        B = batch['root_nodes'].shape[0]

        # 1. Encode root set (Deep Sets — permutation invariant)
        r_feats, r_global = self.root_encoder(
            batch['root_nodes'], batch['root_mask']
        )  # r_feats: [B, R, D], r_global: [B, D]

        # 2. Encode template sequence (Transformer — position sensitive)
        t_feats = self.templ_encoder(
            batch['templ_nodes'], batch['templ_mask']
        )  # [B, T, D]

        # 3. Inject edge information into node features
        # Aggregate edge embeddings: each root node gets mean of its edge slots
        e_emb = self.edge_embed(batch['edges'])   # [B, R, T, D]
        templ_mask_f = batch['templ_mask'].float()  # [B, T]
        # Weighted mean over template slots for each root node
        e_for_r = (e_emb * templ_mask_f.unsqueeze(1).unsqueeze(-1)).sum(2)  # [B, R, D]
        n_slots = templ_mask_f.sum(1, keepdim=True).unsqueeze(-1).clamp(min=1)
        e_for_r = e_for_r / n_slots                 # [B, R, D]

        # Mean over root nodes for each template slot
        root_mask_f = batch['root_mask'].float()    # [B, R]
        e_for_t = (e_emb * root_mask_f.unsqueeze(2).unsqueeze(-1)).sum(1)   # [B, T, D]
        n_roots = root_mask_f.sum(1, keepdim=True).unsqueeze(-1).clamp(min=1)
        e_for_t = e_for_t / n_roots                 # [B, T, D]

        r_feats = r_feats + e_for_r
        t_feats = t_feats + e_for_t

        # 4. Time conditioning
        t_emb = self.time_embed(t, self.d_model)    # [B, D]
        r_feats = r_feats + self.time_proj_r(t_emb).unsqueeze(1)   # broadcast [B,1,D]
        t_feats = t_feats + self.time_proj_t(t_emb).unsqueeze(1)

        # 5. Coupled cross-attention
        for layer in self.coupling_layers:
            r_feats, t_feats = layer(r_feats, t_feats, batch['root_mask'], batch['templ_mask'])

        # 6. Output heads
        root_logits = self.root_out(r_feats)        # [B, R, N_CONSONANTS]
        templ_logits = self.templ_out(t_feats)      # [B, T, N_SLOTS]
        edge_logits = self.edge_predictor(r_feats, t_feats)  # [B, R, T, N_EDGE_TYPES]

        return {
            'root_logits': root_logits,
            'templ_logits': templ_logits,
            'edge_logits': edge_logits,
        }


# ── Forward (Masking) Process ─────────────────────────────────────────────

def mask_graph(batch: dict, t: torch.Tensor) -> dict:
    """
    Forward masking process: independently mask each token with probability t.

    batch: clean batch
    t:     [B] float in [0, 1]  — noise level

    Returns noisy batch with same structure.
    """
    B = batch['root_nodes'].shape[0]
    noisy = dict(batch)   # shallow copy

    # For each sample in batch, mask each token independently with prob t[b]
    t_b = t.view(B, 1)  # [B, 1] for broadcasting

    # Root nodes
    root_clean = batch['root_nodes'].clone()                    # [B, R]
    mask_r = torch.bernoulli(t_b.expand_as(root_clean)).bool() # [B, R]
    noisy_roots = root_clean.clone()
    noisy_roots[mask_r & batch['root_mask']] = CONSONANT_VOCAB['<MASK>']
    noisy['root_nodes'] = noisy_roots

    # Template nodes
    templ_clean = batch['templ_nodes'].clone()                  # [B, T]
    mask_t = torch.bernoulli(t_b.expand_as(templ_clean)).bool()
    noisy_templs = templ_clean.clone()
    noisy_templs[mask_t & batch['templ_mask']] = SLOT_VOCAB['<MASK>']
    noisy['templ_nodes'] = noisy_templs

    # Edges: mask independently
    edges_clean = batch['edges'].clone()                        # [B, R, T]
    t_b2 = t_b.unsqueeze(-1)                                   # [B, 1, 1]
    mask_e = torch.bernoulli(t_b2.expand_as(edges_clean)).bool()
    # Only mask real edges (where both root and template nodes are real)
    real_edge_mask = (batch['root_mask'].unsqueeze(2) & batch['templ_mask'].unsqueeze(1))
    noisy_edges = edges_clean.clone()
    noisy_edges[mask_e & real_edge_mask] = EDGE_VOCAB['mask']
    noisy['edges'] = noisy_edges

    return noisy


# ── Loss ──────────────────────────────────────────────────────────────────

def compute_loss(model_out: dict, clean_batch: dict) -> dict:
    """
    Compute the discrete flow matching loss:
      L = -log p(x0 | xt, t)
    summed over root nodes, template nodes, and edges.

    Returns dict of individual + total losses.
    """
    # Root loss — only over real root nodes
    root_logits = model_out['root_logits']           # [B, R, N_CONSONANTS]
    root_target = clean_batch['root_nodes']          # [B, R]
    root_mask = clean_batch['root_mask']             # [B, R]

    B, R, V_R = root_logits.shape
    root_loss = F.cross_entropy(
        root_logits.reshape(B * R, V_R),
        root_target.reshape(B * R),
        reduction='none',
    ).reshape(B, R)
    root_loss = (root_loss * root_mask.float()).sum() / root_mask.float().sum().clamp(min=1)

    # Template loss — only over real template nodes
    templ_logits = model_out['templ_logits']         # [B, T, N_SLOTS]
    templ_target = clean_batch['templ_nodes']        # [B, T]
    templ_mask = clean_batch['templ_mask']           # [B, T]

    B, T, V_T = templ_logits.shape
    templ_loss = F.cross_entropy(
        templ_logits.reshape(B * T, V_T),
        templ_target.reshape(B * T),
        reduction='none',
    ).reshape(B, T)
    templ_loss = (templ_loss * templ_mask.float()).sum() / templ_mask.float().sum().clamp(min=1)

    # Edge loss — only over real (root_i, templ_j) pairs
    edge_logits = model_out['edge_logits']           # [B, R, T, N_EDGE_TYPES]
    edge_target = clean_batch['edges']               # [B, R, T]
    real_edge_mask = (root_mask.unsqueeze(2) & templ_mask.unsqueeze(1))  # [B, R, T]

    B, R, T_e, E = edge_logits.shape
    edge_loss = F.cross_entropy(
        edge_logits.reshape(B * R * T_e, E),
        edge_target.reshape(B * R * T_e),
        reduction='none',
    ).reshape(B, R, T_e)
    edge_loss = (edge_loss * real_edge_mask.float()).sum() / real_edge_mask.float().sum().clamp(min=1)

    total = root_loss + templ_loss + 0.5 * edge_loss

    return {
        'total': total,
        'root': root_loss,
        'templ': templ_loss,
        'edge': edge_loss,
    }


# ── Inference (Denoising Sampler) ─────────────────────────────────────────

@torch.no_grad()
def sample(
    model: BipartiteDeFoG,
    root_mask: torch.Tensor,       # [B, R] known root structure
    templ_mask: torch.Tensor,      # [B, T] known template structure
    device: torch.device,
    n_steps: int = 20,
    condition_root: torch.Tensor = None,   # [B, R] if we want to condition on root
    condition_templ: torch.Tensor = None,  # [B, T] if we want to condition on template
) -> dict:
    """
    Sample from the model by iterative denoising from t=1 (fully masked) to t=0.

    Supports three generation modes:
      1. Unconditional: generate both root and template jointly
      2. Root-conditioned: given root consonants, generate template
      3. Template-conditioned: given template, generate root consonants (novel!)

    Zero-shot root transfer is mode 1 or 3 with a new unseen root.
    """
    B = root_mask.shape[0]
    R = root_mask.shape[1]
    T = templ_mask.shape[1]

    # Initialise fully masked state
    current = {
        'root_nodes':  torch.full((B, R), CONSONANT_VOCAB['<MASK>'], dtype=torch.long, device=device),
        'templ_nodes': torch.full((B, T), SLOT_VOCAB['<MASK>'], dtype=torch.long, device=device),
        'edges':       torch.full((B, R, T), EDGE_VOCAB['mask'], dtype=torch.long, device=device),
        'root_mask':   root_mask.to(device),
        'templ_mask':  templ_mask.to(device),
    }

    # If conditioning on known root, fix it
    if condition_root is not None:
        current['root_nodes'] = condition_root.to(device)

    # If conditioning on known template, fix it
    if condition_templ is not None:
        current['templ_nodes'] = condition_templ.to(device)

    model.eval()
    timesteps = torch.linspace(1.0, 0.0, n_steps + 1)[:-1]   # [n_steps]

    for step_idx, t_val in enumerate(timesteps):
        t = torch.full((B,), t_val.item(), device=device)
        out = model(current, t)

        # Greedy decode the predicted clean tokens (argmax)
        pred_root = out['root_logits'].argmax(-1)    # [B, R]
        pred_templ = out['templ_logits'].argmax(-1)  # [B, T]
        pred_edges = out['edge_logits'].argmax(-1)   # [B, R, T]

        # Compute update probability: prob of unmasking at this step
        # In a simple linear schedule: prob = 1 / (n_steps - step_idx)
        p_unmask = 1.0 / max(n_steps - step_idx, 1)

        # For each masked token, unmask with prob p_unmask
        next_root = current['root_nodes'].clone()
        is_masked_r = (current['root_nodes'] == CONSONANT_VOCAB['<MASK>']) & root_mask
        unmask_r = torch.bernoulli(torch.full_like(is_masked_r.float(), p_unmask)).bool()
        if condition_root is None:
            next_root[is_masked_r & unmask_r] = pred_root[is_masked_r & unmask_r]
        else:
            # Keep conditioned root fixed
            pass

        next_templ = current['templ_nodes'].clone()
        is_masked_t = (current['templ_nodes'] == SLOT_VOCAB['<MASK>']) & templ_mask
        unmask_t = torch.bernoulli(torch.full_like(is_masked_t.float(), p_unmask)).bool()
        if condition_templ is None:
            next_templ[is_masked_t & unmask_t] = pred_templ[is_masked_t & unmask_t]
        else:
            pass

        next_edges = current['edges'].clone()
        is_masked_e = (current['edges'] == EDGE_VOCAB['mask'])
        real_e = root_mask.unsqueeze(2) & templ_mask.unsqueeze(1)
        unmask_e = torch.bernoulli(torch.full_like(is_masked_e.float(), p_unmask)).bool()
        next_edges[is_masked_e & real_e & unmask_e] = pred_edges[is_masked_e & real_e & unmask_e]

        current = {**current, 'root_nodes': next_root, 'templ_nodes': next_templ, 'edges': next_edges}

    # Final deterministic prediction for any remaining masked tokens
    t_final = torch.zeros(B, device=device)
    out = model(current, t_final)
    final_root = out['root_logits'].argmax(-1)
    final_templ = out['templ_logits'].argmax(-1)
    final_edges = out['edge_logits'].argmax(-1)

    still_masked_r = (current['root_nodes'] == CONSONANT_VOCAB['<MASK>']) & root_mask
    still_masked_t = (current['templ_nodes'] == SLOT_VOCAB['<MASK>']) & templ_mask
    still_masked_e = (current['edges'] == EDGE_VOCAB['mask']) & (root_mask.unsqueeze(2) & templ_mask.unsqueeze(1))

    current['root_nodes'][still_masked_r] = final_root[still_masked_r]
    current['templ_nodes'][still_masked_t] = final_templ[still_masked_t]
    current['edges'][still_masked_e] = final_edges[still_masked_e]

    return current
