"""
graph.py — BipartiteMorphGraph construction

A Syriac word is modelled as a bipartite graph G = (R, T, E) where:

  R = root nodes   — one per consonant in the root (set, unordered)
  T = template nodes — one per slot in the template (sequence, ordered)
  E = interdigitation edges — connects each template slot to the root
      consonant that fills it (if any; affix slots connect to a special
      AFFIX node)

Node features:
  Root nodes:   one-hot over 22 Syriac consonants + MASK token
  Template nodes: one-hot over slot-type vocabulary + position embedding

Edge features:
  'fills': root consonant r fills template slot t
  'affix': slot t is filled by an affix (not a root consonant)
  'no_edge': t is empty / unconnected (used as null class)

For the flow model, we operate over the CATEGORICAL state of each node:
  - Root nodes flow over consonant vocabulary V_R (22 consonants + MASK)
  - Template nodes flow over slot vocabulary V_T (slot types + MASK)
  - Edges flow over {fills, affix, no_edge}

During generation, we start from a fully-MASKED state and flow toward
a valid (root, template) configuration.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── Vocabularies ──────────────────────────────────────────────────────────

SYRIAC_CONSONANT_LIST = [
    'ܐ', 'ܒ', 'ܓ', 'ܕ', 'ܗ', 'ܘ', 'ܙ', 'ܚ', 'ܛ', 'ܝ',
    'ܟ', 'ܠ', 'ܡ', 'ܢ', 'ܣ', 'ܥ', 'ܦ', 'ܨ', 'ܩ', 'ܪ', 'ܫ', 'ܬ'
]
CONSONANT_VOCAB = {c: i for i, c in enumerate(SYRIAC_CONSONANT_LIST)}
CONSONANT_VOCAB['<MASK>'] = len(SYRIAC_CONSONANT_LIST)
CONSONANT_VOCAB['<PAD>'] = len(SYRIAC_CONSONANT_LIST) + 1
N_CONSONANTS = len(CONSONANT_VOCAB)   # 24

SLOT_TYPE_LIST = [
    'C',    # consonant, no vowel
    'Ca',   # + patah (a)
    'Ci',   # + hbasa (i)
    'Cu',   # + esasa (u)
    'Ce',   # + zqapha (e/a)
    'Co',   # + rwaha (o)
    'C:',   # geminate
    'Cat',  # + a + t (feminine ending)
    'Ciu',  # + compound vowel
    'Ceu',  # + compound vowel
]
SLOT_VOCAB = {s: i for i, s in enumerate(SLOT_TYPE_LIST)}
SLOT_VOCAB['<MASK>'] = len(SLOT_TYPE_LIST)
SLOT_VOCAB['<PAD>'] = len(SLOT_TYPE_LIST) + 1
N_SLOTS = len(SLOT_VOCAB)   # 12

EDGE_VOCAB = {'no_edge': 0, 'fills': 1, 'affix': 2, 'mask': 3}
N_EDGE_TYPES = len(EDGE_VOCAB)   # 4

MAX_ROOT_LEN = 5    # max radicals
MAX_TEMPL_LEN = 8   # max template slots


# ── Graph Data Structure ──────────────────────────────────────────────────

@dataclass
class BipartiteMorphGraph:
    """
    A bipartite graph representing one Syriac word's morphological structure.

    Tensors:
      root_nodes:   [R]    — consonant indices for each root node
      templ_nodes:  [T]    — slot-type indices for each template node
      root_mask:    [R]    — 1 for real nodes, 0 for padding
      templ_mask:   [T]    — 1 for real nodes, 0 for padding
      edges:        [R, T] — edge type matrix (fills/affix/no_edge)
      root_pos:     [R]    — position of consonant in canonical root order
      templ_pos:    [T]    — position in template (0,1,2,...)
      label:        str    — surface form (target)
      root_id:      int
      template_name: str
    """
    root_nodes: torch.Tensor    # [MAX_ROOT_LEN]
    templ_nodes: torch.Tensor   # [MAX_TEMPL_LEN]
    root_mask: torch.Tensor     # [MAX_ROOT_LEN]
    templ_mask: torch.Tensor    # [MAX_TEMPL_LEN]
    edges: torch.Tensor         # [MAX_ROOT_LEN, MAX_TEMPL_LEN]
    root_pos: torch.Tensor      # [MAX_ROOT_LEN]
    templ_pos: torch.Tensor     # [MAX_TEMPL_LEN]
    label: str = ''
    root_id: int = -1
    template_name: str = ''

    def to(self, device):
        return BipartiteMorphGraph(
            root_nodes=self.root_nodes.to(device),
            templ_nodes=self.templ_nodes.to(device),
            root_mask=self.root_mask.to(device),
            templ_mask=self.templ_mask.to(device),
            edges=self.edges.to(device),
            root_pos=self.root_pos.to(device),
            templ_pos=self.templ_pos.to(device),
            label=self.label,
            root_id=self.root_id,
            template_name=self.template_name,
        )


def build_graph(item: dict) -> Optional[BipartiteMorphGraph]:
    """
    Build a BipartiteMorphGraph from a dataset item.

    The edge assignment heuristic:
    - Root consonants appear in the surface in order.
    - We find the positions of root consonants within the surface consonant sequence.
    - Each such position maps to a template slot.
    - Remaining slots are labeled 'affix'.
    """
    root_cons = item['root_consonants']   # list of str consonants
    template = item['template']            # list of slot-type strings
    surface = item['surface']             # str

    if len(root_cons) < 2 or len(root_cons) > MAX_ROOT_LEN:
        return None
    if len(template) < 2 or len(template) > MAX_TEMPL_LEN:
        return None

    # Find positions of root consonants in surface consonant sequence
    from .data import extract_root_consonants, SYRIAC_CONSONANTS
    surface_cons = extract_root_consonants(surface)

    # Greedy alignment: find first occurrence of each root consonant in order
    slot_assignments = {}  # slot_idx -> root_idx
    cons_pos = 0
    for ri, rc in enumerate(root_cons):
        for si in range(cons_pos, len(surface_cons)):
            if surface_cons[si] == rc:
                slot_assignments[si] = ri
                cons_pos = si + 1
                break

    # Build tensors
    R = MAX_ROOT_LEN
    T = MAX_TEMPL_LEN

    root_nodes = torch.full((R,), CONSONANT_VOCAB['<PAD>'], dtype=torch.long)
    templ_nodes = torch.full((T,), SLOT_VOCAB['<PAD>'], dtype=torch.long)
    root_mask = torch.zeros(R, dtype=torch.bool)
    templ_mask = torch.zeros(T, dtype=torch.bool)
    edges = torch.zeros(R, T, dtype=torch.long)   # default: no_edge
    root_pos = torch.zeros(R, dtype=torch.long)
    templ_pos = torch.arange(T, dtype=torch.long)

    # Fill root nodes
    for ri, rc in enumerate(root_cons):
        root_nodes[ri] = CONSONANT_VOCAB.get(rc, CONSONANT_VOCAB['<MASK>'])
        root_mask[ri] = True
        root_pos[ri] = ri

    # Fill template nodes + edges
    for ti, slot_type in enumerate(template):
        templ_nodes[ti] = SLOT_VOCAB.get(slot_type, SLOT_VOCAB['<MASK>'])
        templ_mask[ti] = True

        if ti in slot_assignments:
            ri = slot_assignments[ti]
            if ri < R:
                edges[ri, ti] = EDGE_VOCAB['fills']
        else:
            # Affix slot: connect to root node 0 with 'affix' type
            # (encodes "this slot is filled by morphological material, not root")
            edges[0, ti] = EDGE_VOCAB['affix']

    return BipartiteMorphGraph(
        root_nodes=root_nodes,
        templ_nodes=templ_nodes,
        root_mask=root_mask,
        templ_mask=templ_mask,
        edges=edges,
        root_pos=root_pos,
        templ_pos=templ_pos,
        label=surface,
        root_id=item.get('root_id', -1),
        template_name=item.get('template_name', ''),
    )


def graphs_to_batch(graphs: list[BipartiteMorphGraph]) -> dict:
    """Stack a list of graphs into a batch dict of tensors."""
    return {
        'root_nodes':  torch.stack([g.root_nodes for g in graphs]),   # [B, R]
        'templ_nodes': torch.stack([g.templ_nodes for g in graphs]),  # [B, T]
        'root_mask':   torch.stack([g.root_mask for g in graphs]),    # [B, R]
        'templ_mask':  torch.stack([g.templ_mask for g in graphs]),   # [B, T]
        'edges':       torch.stack([g.edges for g in graphs]),        # [B, R, T]
        'root_pos':    torch.stack([g.root_pos for g in graphs]),     # [B, R]
        'templ_pos':   torch.stack([g.templ_pos for g in graphs]),    # [B, T]
        'labels':      [g.label for g in graphs],
        'root_ids':    [g.root_id for g in graphs],
    }


def collate_fn(graphs: list[BipartiteMorphGraph]) -> dict:
    """PyTorch DataLoader collate function."""
    return graphs_to_batch(graphs)
