#!/usr/bin/env python3
"""Any HuggingFace encoder as a Syriac word/document encoder (SOTA-currency check).

Generalizes the CANINE approach (``neural.canine_encoder``) to *any* HuggingFace
encoder that has a tokenizer -- in particular **Glot500-m** (``cis-lmu/glot500-base``),
an XLM-R extended to 511 predominantly low-resource languages that *may* cover
Syriac. The point is to test whether a newer massively-multilingual base beats
off-the-shelf CANINE-c (authorship AUC 0.870 / 0.849) on the paper's exact cohort.

Crucial difference from CANINE: CANINE is tokenizer-free over codepoints, so it is
**zero-OOV by construction**. A subword model can instead map Syriac to ``<unk>``,
which would silently invalidate the embeddings. So this module runs an explicit
**tokenizer-coverage check** first (fraction of Syriac forms that hit ``<unk>``)
and prints it before any authorship number is reported.

TLS to HuggingFace is verified through the OS trust store via ``truststore`` (the
same fix used in ``canine_encoder``); no certificate verification is disabled.

    # coverage check only (fast; tells you if the base even covers Syriac)
    .venv/bin/python -m neural.hf_encoder --model cis-lmu/glot500-base --coverage-only

    # full authorship AUC on the paper cohort, comparable to Table 6
    .venv/bin/python -m neural.hf_encoder --model cis-lmu/glot500-base --floors 1000,2000
    .venv/bin/python -m neural.hf_encoder --model cis-lmu/glot500-base --av-head
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:  # pragma: no cover - environment dependent
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from script import DEFAULT_CACHE, ensure_corpus
from stylometry import load_texts, filter_min_texts, filter_min_tokens
from authorship import parse_ids
# reuse the exact scoring pipeline (cohort + centering + AUC + AV head + CI)
from neural.canine_encoder import authorship_auc, DISPUTED_DEFAULT

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _HF = True
except Exception:  # pragma: no cover
    _HF = False

try:
    import sentencepiece as _spm
    _SPM = True
except Exception:  # pragma: no cover
    _SPM = False


class SPTokenizerShim:
    """Minimal XLM-R-style tokenizer over a raw SentencePiece model.

    transformers 5.x cannot load XLM-R/Glot500 SentencePiece tokenizers (it
    misroutes them through a tiktoken reader). This shim reproduces the exact
    XLM-R id scheme directly from the ``.model`` file: ``<s>``=0, ``<pad>``=1,
    ``</s>``=2, ``<unk>``=3, and every content piece is ``sp_id + 1`` (the
    fairseq offset). It exposes just the slice of the HF tokenizer API this module
    uses, so ``HFWordVectors`` and ``tokenizer_coverage`` work unchanged.
    """

    CLS, PAD, SEP, UNK, OFFSET = 0, 1, 2, 3, 1

    def __init__(self, sp_model_path: str):
        self.sp = _spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.unk_token_id = self.UNK
        self.vocab_size = self.sp.GetPieceSize() + 2  # + fairseq offset + <mask>

    def _content_ids(self, text: str) -> list[int]:
        out = []
        for i in self.sp.EncodeAsIds(text):
            out.append(self.UNK if i == self.sp.unk_id() else i + self.OFFSET)
        return out

    def __call__(self, forms, return_tensors=None, padding=False,
                 truncation=False, max_length=None, add_special_tokens=True):
        single = isinstance(forms, str)
        batch = [forms] if single else list(forms)
        seqs = []
        for f in batch:
            ids = self._content_ids(f)
            if add_special_tokens:
                ids = [self.CLS] + ids + [self.SEP]
            if truncation and max_length:
                ids = ids[:max_length]
            seqs.append(ids)
        if return_tensors != "pt":
            return {"input_ids": seqs[0] if single else seqs}
        maxlen = max(len(s) for s in seqs)
        ii = [s + [self.PAD] * (maxlen - len(s)) for s in seqs]
        am = [[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am)}


def _load_tokenizer(model_id: str):
    """Load a tokenizer robustly.

    transformers 5.x can misread an XLM-R SentencePiece model as a tiktoken file;
    fall back to a raw-SentencePiece shim (``SPTokenizerShim``) built from the
    cached ``sentencepiece.bpe.model``.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - version dependent
        print(f"  HF tokenizer unavailable ({type(exc).__name__}); using a raw "
              f"SentencePiece shim (XLM-R id scheme) ...", file=sys.stderr)
        if not _SPM:
            raise RuntimeError("sentencepiece is required for the fallback shim.")
        from huggingface_hub import hf_hub_download
        sp_path = hf_hub_download(model_id, "sentencepiece.bpe.model")
        return SPTokenizerShim(sp_path)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenizer_coverage(tokenizer, forms, sample: int = 3000) -> dict:
    """Fraction of Syriac forms that tokenize to real (non-``<unk>``) subwords.

    ``frac_all_unk`` near 1 means the base does not cover the Syriac *script* at
    all (authorship numbers would be meaningless); a low ``frac_with_unk`` and a
    moderate ``mean_subwords`` means genuine coverage.
    """
    unk = tokenizer.unk_token_id
    forms = list(forms)
    if len(forms) > sample:
        rng = np.random.default_rng(0)
        forms = [forms[i] for i in rng.choice(len(forms), size=sample, replace=False)]
    n_with_unk = n_all_unk = total_sub = counted = 0
    for f in forms:
        ids = tokenizer(f, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        counted += 1
        total_sub += len(ids)
        if unk is not None and unk in ids:
            n_with_unk += 1
            if all(i == unk for i in ids):
                n_all_unk += 1
    counted = max(counted, 1)
    return {
        "forms_checked": counted,
        "mean_subwords_per_form": round(total_sub / counted, 2),
        "frac_with_any_unk": round(n_with_unk / counted, 4),
        "frac_entirely_unk": round(n_all_unk / counted, 4),
    }


if _HF:

    class HFWordVectors:
        """A ``wv``-compatible encoder over an arbitrary HF tokenizer+model.

        ``wv[token]`` = mean-pooled ``last_hidden_state`` over the token's subword
        pieces (including the model's special tokens, mirroring the CANINE wrapper
        so the two are compared on equal footing). Cached.
        """

        def __init__(self, model, tokenizer, device, batch_size: int = 64,
                     max_length: int = 64):
            self.model = model
            self.tok = tokenizer
            self.device = device
            self.batch_size = batch_size
            self.max_length = max_length
            self.vector_size = int(model.config.hidden_size)
            self._cache: dict[str, np.ndarray] = {}

        def __contains__(self, token: str) -> bool:
            return True  # the subword tokenizer always yields *some* ids

        def _encode_batch(self, forms: list[str]) -> None:
            enc = self.tok(forms, return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            h = out.last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).type_as(h)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            pooled = pooled.float().cpu().numpy().astype(np.float64)
            for f, v in zip(forms, pooled):
                self._cache[f] = v

        def precompute(self, forms) -> None:
            todo = sorted({f for f in forms if f and f not in self._cache}, key=len)
            for i in range(0, len(todo), self.batch_size):
                self._encode_batch(todo[i:i + self.batch_size])

        def __getitem__(self, token: str) -> np.ndarray:
            v = self._cache.get(token)
            if v is None:
                self._encode_batch([token])
                v = self._cache[token]
            return v


def _all_forms(normalize: bool) -> list[str]:
    data_dir = ensure_corpus(DEFAULT_CACHE)
    genuine = load_texts(data_dir, normalize,
                         exclude_ids=set(parse_ids(DISPUTED_DEFAULT)),
                         drop_anonymous=True)
    return sorted({tok for t in genuine for tok in t.counts})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cis-lmu/glot500-base",
                    help="any HuggingFace encoder id (default: Glot500-m)")
    ap.add_argument("--floors", default="1000,2000")
    ap.add_argument("--av-head", action="store_true",
                    help="apply the supervised leave-one-author-out AV head")
    ap.add_argument("--coverage-only", action="store_true",
                    help="report Syriac tokenizer coverage and exit (fast)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args(argv)

    if not _HF:
        print("transformers + torch required. Install:\n"
              "    .venv/bin/python -m pip install -r neural/requirements-neural.txt",
              file=sys.stderr)
        return 2

    normalize = not args.no_normalize
    print(f"loading tokenizer for {args.model} ...", file=sys.stderr)
    tok = _load_tokenizer(args.model)

    forms = _all_forms(normalize)
    cov = tokenizer_coverage(tok, forms)
    print(f"\n=== Syriac tokenizer coverage: {args.model} ===")
    print(f"  forms checked         : {cov['forms_checked']:,}")
    print(f"  mean subwords / form  : {cov['mean_subwords_per_form']}")
    print(f"  forms with any <unk>  : {cov['frac_with_any_unk']:.1%}")
    print(f"  forms entirely <unk>  : {cov['frac_entirely_unk']:.1%}")
    if cov["frac_entirely_unk"] > 0.5:
        print("  VERDICT: the base does NOT meaningfully cover the Syriac script;")
        print("           authorship numbers below would be unreliable.")
    elif cov["frac_with_any_unk"] > 0.25:
        print("  VERDICT: partial coverage (many forms hit <unk>); read AUC with care.")
    else:
        print("  VERDICT: Syriac script is covered; authorship AUC is meaningful.")

    if args.coverage_only:
        return 0

    device = _device()
    print(f"\nloading {args.model} on {device} ...", file=sys.stderr)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    # Guard the SP-shim id scheme: the model's embedding table must cover every id
    # the shim can emit (a wrong fairseq offset would otherwise index garbage).
    if isinstance(tok, SPTokenizerShim):
        mvocab = int(getattr(model.config, "vocab_size", 0))
        if mvocab and abs(mvocab - tok.vocab_size) > 2:
            print(f"  WARNING: model vocab_size {mvocab:,} != shim vocab_size "
                  f"{tok.vocab_size:,}; the SentencePiece id offset may be wrong, "
                  f"so authorship numbers could be unreliable.", file=sys.stderr)
        else:
            print(f"  SP-shim id scheme verified (model vocab {mvocab:,} ~ "
                  f"shim {tok.vocab_size:,}).", file=sys.stderr)
    wv = HFWordVectors(model, tok, device)

    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    rows = authorship_auc(wv, floors, normalize=normalize,
                          use_av_head=args.av_head, seed=args.seed)
    tag = f"{args.model}" + (" + AV head (LOAO)" if args.av_head else "")
    print(f"\n=== authorship separation AUC: {tag} ===")
    print("(centered cosine; same cohort + metric as paper Table 6)")
    has_ci = any(r.get("ci") for r in rows)
    header = f"  {'floor':>6}  {'AUC':>6}" + (f"  {'95% CI':>16}" if has_ci else "") + "  texts  authors"
    print(header)
    for r in rows:
        line = f"  {r['floor']:>6}  {r['auc']:.3f}"
        if has_ci:
            line += f"  [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r.get("ci") else "  {:>16}".format("")
        line += f"   {r['texts']:>3}    {r['authors']:>2}"
        print(line)
    print("\nReference: CANINE-c off-the-shelf 0.870/0.849, + AV head 0.991/0.916;")
    print("FastText 0.915/0.900, word2vec 0.946/0.938, Delta 0.907/0.940 (Table 6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
