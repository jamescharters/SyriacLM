# Pointing as Morphology: A Self-Supervised Vocaliser and Cross-Script Transfer for Classical Syriac

**Anonymous** · *TODO: affiliation*

> **Note.** This is a Markdown rendering of the companion paper for easy reading.
> The canonical, citable source is [`main.tex`](main.tex) (XeLaTeX). All numbers are
> produced by [`../results.py`](../results.py), which runs each experiment, writes a
> JSON manifest, and emits the LaTeX tables; this file mirrors those numbers. Syriac
> forms are shown in script where a Syriac font is available, always with
> transliteration. This paper is a self-contained companion to the FastText/stylometry
> paper in [`../../paper/`](../../paper/) and reuses its authorship cohort and metric.

---

## Abstract

Classical Syriac has no large pretrained language model, and from-scratch pretraining
is ruled out by scale: the openly digitised corpus is on the order of a few million
tokens, one to two orders of magnitude short of what monolingual pretraining needs. We
ask how to build useful neural representations for Syriac by injecting information from
sources other than more Syriac text, and report what works and what does not on the
same authorship cohort and metric as a prior count-based study. Our central result is a
vocaliser that treats the restoration of Syriac pointing as a morphological task: in a
defective abjad the unwritten vowels are the *pattern* morpheme of a root-and-pattern
system, so predicting them is recovering morphology, not orthographic decoration.
Trained on the SEDRA lexicon, the model restores held-out pointing at 0.825 per slot.
It generalises to a different register: although the Digital Syriac Corpus was assumed
to be consonantal, 56% of its tokens in fact carry vowel points, which we turn into a
134,000-form classical gold set by deriving the mark-to-vowel mapping from data; on it
the lexicon-trained vocaliser reaches 0.638 ± 0.028 vowel accuracy on in-vocabulary
forms and 0.594 ± 0.019 on forms whose consonantal skeleton it never saw. For
document-level authorship we compare three transfer routes for a zero-resource abjad: a
tokenizer-free byte model (CANINE) reaches AUC 0.870 off the shelf; transliterating
Syriac into Hebrew script and encoding it with a Hebrew model reaches 0.888; and a
larger multilingual model whose tokenizer represents Syriac as character fallback
reaches only 0.798 — so script coverage, not language count, is what matters. A
supervised verification head over either encoder gives the strongest separation
(≈ 0.96 ± 0.03), though it is seed-sensitive on the small author pool. Continued
pretraining improves the intrinsic language-model objective but not document-level
authorship; an explicit root/pattern factorisation does not beat a flat encoder when
the consonantal skeleton is already visible. Code, the evaluation harness, and the
gold-set construction are released; SEDRA-derived data is regenerated locally under its
licence rather than redistributed.

---

## 1. Introduction

Classical Syriac — a dialect of Aramaic and a principal literary language of late
antique Christianity — has a large, partly anonymous or contested corpus and very
little modern language technology. There is no large pretrained Syriac language model,
and the reason is not architectural but quantitative: a from-scratch monolingual
Transformer expects on the order of 10⁸–10⁹ tokens, whereas the Digital Syriac Corpus
(DSC) holds about 2.18M, and even aggregating every openly digitised Syriac text
reaches only the low tens of millions. Pretraining from scratch is the wrong tool, and
so is the tempting shortcut of generating "more Syriac" from a model fit on Syriac,
which adds no information and risks degenerate feedback (Shumailov et al., 2024).

If new information cannot come from more raw Syriac, it must come from elsewhere. We
organise the design around three sources. **Structure**: the root-and-pattern grammar
can be supplied as explicit supervision. **Transfer**: knowledge can be carried over
from byte- or character-level models and from related Semitic languages. **Real text**:
the genuinely new tokens that do exist — here, a vocalisation signal the corpus turns
out to contain — can be put to work. This paper develops one method in each category and
evaluates them against the cohort and metric of a prior count-based study of the same
corpus, so the neural numbers are directly comparable.

The work is deliberately a study of *methods and findings* rather than of new
architectures: the components — a tokenizer-free encoder (CANINE), low-rank adaptation
(LoRA), a Hebrew model (AlephBERT), a supervised-contrastive head, a sequence labeller —
are all established. The contribution is what they reveal about Syriac and about
modelling a zero-resource Semitic abjad.

**Contributions.**

1. **Pointing as morphology.** We frame vocalisation restoration as recovery of the
   root-and-pattern *pattern* morpheme and build a vocaliser for Classical Syriac (§5).
   To our knowledge it is the first neural Syriac vocaliser; diacritisation itself is
   established for Arabic and Hebrew, so the novelty is the language and the
   morphological framing.
2. **A classical vocalisation gold set.** We show the DSC is 56% vocalised, derive the
   mapping from its Unicode vowel points to a phonemic label set directly from data, and
   release the construction of a 134,000-form classical gold set on which cross-register
   transfer can be measured (§5).
3. **Transfer routes for an abjad.** We compare a tokenizer-free byte encoder,
   cross-script transfer through Hebrew, and a large multilingual model on identical
   authorship evaluation, and explain the ordering by tokenizer coverage (§6).
4. **What does and does not help.** A supervised verification head gives the best
   authorship separation but is seed-sensitive on a small author pool; continued
   pretraining helps the intrinsic objective but not authorship; and an explicit
   root/pattern factorisation does not beat a flat encoder (§6, §7).

---

## 2. Related Work

**Tokenizer-free and multilingual encoders.** Subword vocabularies built for
high-resource languages cover an abjad like Syriac poorly. Tokenizer-free models read
bytes or codepoints — CANINE encodes Unicode codepoints (Clark et al., 2022), ByT5
operates on bytes (Xue et al., 2022) — extending character-aware modelling to the
Transformer. Massively multilingual subword models such as XLM-R and Glot500 cover
hundreds of languages, but nominal coverage of a script does not guarantee useful
subwords for it.

**Cross-lingual transfer and adaptation.** When target data is scarce, knowledge is
transferred from related languages or adapted with few parameters: continued in-domain
pretraining (Gururangan et al., 2020) and parameter-efficient methods — adapters
(Houlsby et al., 2019) and LoRA (Hu et al., 2022). Syriac and Hebrew are both 22-letter
Aramaic-derived abjads, motivating transliteration of Syriac into Hebrew to reuse a
Hebrew model (Seker et al., 2022); a Syriac morphological parser has likewise borrowed
Hebrew data (Naaijer et al., 2023).

**Diacritisation and vocalisation.** Restoring vowels to a consonantal script is well
studied for Arabic (Zalmout & Habash, 2019) and Hebrew (Gershuni & Pinter, 2022),
typically as sequence labelling or a reading aid. We reuse that machinery but reframe
the objective: for a root-and-pattern language the pointing *is* the pattern morpheme,
so vocalisation is morphological self-supervision.

**Authorship and restoration.** Burrows's Delta and the broader attribution literature
(Stamatatos, 2009) provide the stylometric baselines; a supervised-contrastive head
(Khosla et al., 2020) provides a learned metric; uncertainty is the author-cluster
bootstrap. Neural restoration of damaged ancient text is established for Greek and Latin
(Assael et al., 2022); we include a Syriac character-level restorer as a demonstrator.

---

## 3. Data and Resources

**Corpora.** The DSC (`srophe/syriac-corpus`) provides 632 authored texts and 2.18M
tokens of classical prose and verse. The ETCBC Syriac New Testament and Peshitta provide
additional consonantal running text with morphology. The SEDRA 3 lexicon (Kiraz, 1994)
provides 29,699 word forms with explicit vocalisation, root, and morphology; it is the
vocaliser's supervision. SEDRA is distributed for academic use without redistribution of
altered versions, so we ship the code that regenerates the derived data from a
user-provided SEDRA source and cite Kiraz as required.

**The CAL skeleton alphabet and transliteration.** SEDRA encodes forms in a CAL-style
ASCII alphabet that is *non-phonetic*: the letter is a fixed slot in the Aramaic
alphabet order, so `K` is *Heth*, `C` is *Kaph*, `;` is *Yudh*, `/` is *Sadhe*, `X` is
*Qaph*. We add a deterministic Syriac↔CAL map (and a Syriac→Hebrew map for transfer);
all 29,699 SEDRA skeletons round-trip exactly. A vocalised SEDRA form interleaves
lowercase vowels and diacritics with the skeleton, e.g. `AaB,oHaOH_;` has skeleton
`ABHOH;` (ܐܒܗܘܗܝ, *abāhaw(hy)*).

**The corpus is vocalised.** The DSC was assumed consonantal, and the prior study strips
combining marks. In fact 55.9% of DSC tokens — across 600 of 632 files — carry Syriac
vowel points (U+0730–U+074A: *pthaha*, *zqapha*, *rbasa*, *hbasa*, *esasa*, the
East-Syriac dotted *zlama*s, *quššaya*/*rukkaka*). This is a genuine classical
vocalisation signal, distinct in register from SEDRA's New-Testament lexicon, and §5
turns it into an evaluation gold set.

---

## 4. Evaluation

Document-level authorship uses the prior study's evaluation: centred-cosine separation
AUC between same- and cross-author text pairs, on the same cohort, at token floors 1000
and 2000. AUC is the probability a same-author pair is scored more similar than a
cross-author pair; 0.5 is chance. The vocaliser uses per-slot pointing accuracy on
held-out SEDRA, and vowel accuracy on pointed slots for classical DSC.

---

## 5. Pointing as Morphology: A Syriac Vocaliser

**Objective.** A Syriac word is a consonantal root interleaved with a vocalic pattern;
the script writes the consonants and usually omits the pattern, so restoring the pointing
is recovering the pattern morpheme. We frame it as sequence labelling: the input is the
consonant skeleton, and for each consonant slot the model predicts the vowel/diacritic
string that follows it (one of 30 classes including "bare"). Every position is supervised,
so the model cannot collapse to the prior the way a sparsely-masked objective can on
small data. The model is a two-layer bidirectional LSTM (603K parameters).

**Held-out lexicon accuracy.** On a 90/10 type-level split of SEDRA, the vocaliser
restores 0.825 of slots correctly (per-consonant majority baseline 0.507) and reproduces
0.397 of full words exactly (baseline 0.003). These are held-out *New-Testament-vocabulary*
forms, which raises the question of register.

**Cross-register evaluation on classical text.** Because the corpus marks vowels with
Unicode points and SEDRA with CAL letters, we need a mapping; rather than assert one, we
*derive* it from data. Aligning the 541,416 DSC tokens whose skeleton has a unique SEDRA
vocalisation, we tally per slot which Unicode mark co-occurs with which CAL vowel. The
vowels map cleanly — *pthaha*→`a`, *zqapha*→`o`, *rbasa*→`e`, *hbasa*→`i`, *esasa*→`u` at
0.90–0.98 purity — while reading and grammatical marks fall below threshold and are
treated as unvocalised (Table 2). Applying the derived map to all vocalised DSC tokens
yields 133,960 unique (skeleton, pattern) forms, of which 47,973 have a skeleton in
SEDRA's vocabulary and 85,987 do not.

Classical scribes point selectively, so the fair metric is vowel accuracy on slots that
are actually pointed. The lexicon-trained vocaliser, applied unchanged to classical text,
reaches 0.638 ± 0.028 on in-vocabulary skeletons and 0.594 ± 0.019 on out-of-vocabulary
skeletons — forms whose consonantal skeleton never appeared in training (mean ± SD over
five seeds; per-consonant baselines 0.40 and 0.45). The small gap between the splits
indicates the model learned the root-and-pattern regularity rather than memorising
lexical entries.

**Table 1. The vocaliser** (held-out SEDRA lexicon, upper; classical DSC, lower; cross-register figures are vowel accuracy on pointed slots, 5-seed mean ± SD).

| | acc. | baseline |
|---|---|---|
| *Held-out SEDRA (NT lexicon)* | | |
| per-position pointing | 0.825 | 0.507 |
| full-word exact match | 0.397 | 0.003 |
| *Classical DSC, 5 seeds* | | |
| in SEDRA vocabulary | 0.638 ± 0.028 | 0.403 |
| out of SEDRA vocab. | 0.594 ± 0.019 | 0.447 |

**Table 2. The Unicode-point → CAL-vowel mapping, derived from data** (aligning DSC tokens to uniquely-vocalised SEDRA skeletons; marks accepted as vowels shown).

| Syriac mark (U+0730–074A) | → CAL vowel | purity | count |
|---|---|---|---|
| Zqapha Dotted | o | 0.85 | 122,748 |
| Pthaha Dotted | a | 0.90 | 122,652 |
| Pthaha Above | a | 0.97 | 102,975 |
| Zqapha Above | o | 0.97 | 85,556 |
| Rbasa Above | e | 0.96 | 72,855 |
| Dotted Zlama Horizontal | e | 0.90 | 47,265 |
| Esasa Above | u | 0.98 | 41,127 |
| Dotted Zlama Angular | e | 0.78 | 29,055 |
| Hbasa Above | i | 0.96 | 28,988 |

---

## 6. Transfer for a Zero-Resource Abjad

**A tokenizer-free byte encoder transfers.** Off the shelf, with no Syriac training,
CANINE encodes each form by mean-pooling its codepoint encoding and reaches AUC 0.870 /
0.849 — the first demonstration that a pretrained multilingual byte model transfers to
Syriac authorship, already exceeding from-scratch byte- and character-level models.

**Continued pretraining helps the LM, not authorship.** Adapting CANINE with LoRA (1.06M
of 133M parameters) lowers held-out masked pseudo-bits-per-byte from 2.000 to 1.859 and
raises masked-codepoint accuracy from 0.307 to 0.342 against a frozen probe (Table 4), so
the encoder demonstrably learns Syriac. It does *not* improve authorship (AUC 0.857 /
0.838, slightly below off the shelf): the masked objective is local, and document-mean
pooling washes out what it sharpens.

**Script coverage beats language count.** Glot500, covering 500+ languages including a
nominal Syriac, does worse than CANINE (AUC 0.798 / 0.781). Its tokenizer breaks Syriac
into 7.05 pieces per word with no Syriac subwords — character fallback. Nominal language
coverage is not script coverage.

**Cross-script transfer through Hebrew.** Mapping Syriac into Hebrew script (a near 1:1
abjad correspondence) and encoding it with AlephBERT reaches AUC 0.888 / 0.857, above
CANINE. The tokenizer confirms the mechanism: AlephBERT segments transliterated Syriac
into 2.51 real subwords per word (96.5% covered) against Glot500's 7.05. The ordering for
a zero-resource abjad: char-fallback multilingual (0.798) < tokenizer-free byte (0.870) <
shared-script Semitic transfer (0.888).

**A supervised head gives the best separation, with a caveat.** Adding the prior study's
leave-one-author-out supervised-contrastive head over the encoder vectors gives the
strongest separation: CANINE + head 0.961 ± 0.030 at floor 1000, Hebrew + head
0.946 ± 0.040. Two cautions: the head trains on an 11-author cohort and is
nondeterministic across runs even at a fixed seed, so we report five-seed mean ± SD
rather than a single value (an earlier single run read 0.99, an optimistic draw); and at
floor 2000 the spread is large (± 0.10 for CANINE), so CANINE and Hebrew with the head
overlap within seed variation and we do not rank them there.

**Table 3. Authorship separation AUC** at token floors 1000 / 2000, same cohort and metric as the prior count-based study (its numbers shown for reference; AV-head rows are 5-seed mean ± SD).

| Representation | AUC (floor 1000) | AUC (floor 2000) |
|---|---|---|
| FastText (parent) | 0.915 | 0.900 |
| word2vec (parent) | 0.946 | 0.938 |
| Burrows's Delta (parent) | 0.907 | 0.940 |
| Glot500-m (char fallback) | 0.798 | 0.781 |
| CANINE-c off-the-shelf | 0.870 | 0.849 |
| &nbsp;&nbsp;+ AV head (LOAO) | 0.961 ± 0.030 | 0.917 ± 0.099 |
| Hebrew transliteration | 0.888 | 0.857 |
| &nbsp;&nbsp;+ AV head (LOAO) | 0.946 ± 0.040 | 0.924 ± 0.046 |

**Table 4. Intrinsic LM effect of LoRA** vs. a frozen linear probe (identical masked pseudo-bits-per-byte scorer; comparable only among CANINE variants).

| | trainable params | masked acc. | pseudo-bpb |
|---|---|---|---|
| Frozen linear probe | 23,839 | 0.307 | 2.000 |
| LoRA continued-pretrain | 1,056,031 | 0.342 | 1.859 |

---

## 7. Factored Morphology and Textual Restoration

**Explicit root/pattern factorisation does not help here.** SEDRA's encoding splits every
form into a consonant (root) channel and a vowel (pattern) channel for free, inviting a
two-stream encoder. We compare it against a flat encoder reading the raw vocalised
characters, both trained with the same supervised-contrastive objective on the root and
evaluated by root-nearest-neighbour retrieval (Table 5). The factored model does not beat
the flat one (0.972 vs. 0.978 on seen roots, 0.994 vs. 0.994 on unseen). Retrieval is near
ceiling for both because the consonantal skeleton is directly visible in the input, so
surface overlap already encodes the root; making the split explicit adds nothing when the
split is free.

**Character-level restoration.** As an application demonstrator we train a causal
character Transformer (620K parameters) on the DSC and evaluate lacuna restoration by
synthetic masking. It restores masked characters at 0.443 and whole spans exactly at
0.089, against a unigram floor near 0.19, with morphologically valid fills. A
bidirectional masked objective at this scale collapses to the unigram prior; the causal
objective, which supervises every position, is what learns.

**Table 5. Factored vs. flat encoder**, root-nearest-neighbour retrieval on SEDRA forms split by root ("unseen" roots are absent from training).

| | params | seen-root NN | unseen-root NN |
|---|---|---|---|
| Flat (vocalised chars) | 217,216 | 0.978 | 0.994 |
| Factored (root/pattern) | 251,536 | 0.972 | 0.994 |
| Δ (factored − flat) | | −0.007 | 0.000 |

---

## 8. Discussion

The results divide along the line between intrinsic and document-level objectives, the
same division the prior count-based study found. Intrinsically, structure and transfer
both pay off: the vocaliser learns the pattern morpheme well enough to generalise across
register and to unseen skeletons, and continued pretraining measurably improves the
language-model objective. At the document level, the picture differs: continued
pretraining does not help authorship, because mean-pooling discards the local signal it
sharpens; the supervised head helps most, because it learns a metric directly on the
pooled vectors. The recommendation for a zero-resource abjad follows from the transfer
ordering: prefer a tokenizer-free encoder or transliteration into a related-language
model's script over a large multilingual model whose tokenizer misses the script, and add
a supervised head where author labels exist.

Two results are negative under their stated conditions, and both are informative.
Factoring root from pattern does not help when the skeleton is already visible, which says
the intrinsic root signal is easy and locates the difficulty elsewhere. The supervised
head's instability on an 11-author cohort is a caution about single-run headline numbers
on small pools generally. The broader method — inject structure through a morphological
objective, transfer through script rather than language count, and learn a metric for the
document task — should carry to other templatic, low-resource languages.

---

## 9. Limitations

The vocaliser's supervision is the New-Testament-scoped SEDRA lexicon; we measure
cross-register transfer but still train on one register. The classical gold has limits we
make explicit: scribes point partially, so we score only pointed slots; one East-Syriac
mark (U+073C) is genuinely ambiguous between a vowel and the *rukkaka* dot; and a few
archaic variant letters pass through the CAL map unconverted. The authorship cohort is
small — about 11 authors at the higher token floor, from a single corpus — so absolute AUC
values are cohort-specific and support relative comparison rather than population
estimates; the supervised head's seed sensitivity is a consequence of that scale, and its
floor-2000 numbers are too noisy to rank. The neural models are intentionally small; they
probe method under data scarcity, not an upper bound.

---

## 10. Conclusion

We treated the problem of building neural representations for a language with too little
text to pretrain on by injecting information from three sources: structure, through a
vocaliser that restores Syriac pointing as the pattern morpheme; transfer, through
tokenizer-free and cross-script encoders; and the corpus's own latent vocalisation signal,
turned into a classical gold set by deriving its mark-to-vowel mapping from data. The
vocaliser generalises across register and to unseen skeletons; cross-script transfer
through Hebrew outperforms both a byte encoder and a larger multilingual model whose
tokenizer misses the script; and a supervised head gives the best authorship separation,
within a seed range we report rather than a single number. The negative results — no gain
from explicit factorisation, no authorship gain from continued pretraining — sharpen the
account of where each kind of information helps. Code, evaluation harness, and gold-set
construction are released, with SEDRA-derived data regenerated locally under its licence.

---

## Reproducibility

All numbers are produced by a single driver ([`../results.py`](../results.py)) that runs
each experiment, writes a JSON manifest, and emits the table files, so no value is
transcribed by hand; off-the-shelf and intrinsic numbers are deterministic given the
encoder, and the seed-sensitive AV-head numbers are five-seed mean ± SD. Experiments run
on a single Apple-silicon (MPS) device. The DSC and ETCBC corpora are openly licensed; the
SEDRA lexicon is licence-restricted, so we release the code that regenerates the derived
data from a user-provided SEDRA source rather than the data, and cite Kiraz as required.
