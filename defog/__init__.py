"""Syriac Bipartite DeFoG — a generative, root-and-pattern morphology model.

A fourth, *generative* approach to Syriac morphology (alongside the FastText
stylometry, the neural vocaliser and the disentanglement probe): coupled discrete
flow matching over bipartite root–template graphs. A word is modelled as a
bipartite graph of root-consonant nodes (an unordered set) and template-slot
nodes (an ordered sequence) joined by interdigitation edges; the model denoises a
masked graph back to a valid (root, template) configuration.

Run it as a package from the repository root, e.g.::

    .venv/bin/python -m defog.run               # train on the local SEDRA IV data
    .venv/bin/python -m defog.run --synthetic    # offline toy data (no SEDRA)

Data comes from the shared ``corpora`` package (``corpora/sedra_cache``); see
``defog.data`` and ``neural/docs/DATA.md``.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.0.1"
