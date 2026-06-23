# Character *n*-gram Embeddings for Classical Syriac: Subword Modeling of a Templatic Low-Resource Language, with an Application to Authorship Analysis

**Anonymous** · *TODO: affiliation*  

> **Note.** This is a Markdown rendering of the paper for easy reading. The
> canonical, citable source is [`paper/main.tex`](main.tex) (XeLaTeX). All numbers
> are produced by [`../paper_experiments.py`](../paper_experiments.py); the neural
> baselines by [`../nn_baselines.py`](../nn_baselines.py). Syriac forms are shown
> in script where a Syriac font is available, always followed by transliteration
> and an English gloss.

---

## Abstract

Classical Syriac is a major literary language of late antiquity that remains
severely under-resourced in NLP, yet it raises rich, unresolved questions of
authorship. Our primary contribution is a released character *n*-gram **FastText**
model trained on the Digital Syriac Corpus (632 texts; 2.18M tokens); we argue
that subword representations are the appropriate inductive bias for a templatic,
root-and-pattern language whose vocabulary is 50% *hapax legomena*. Intrinsically,
the embeddings are morphologically coherent — forms sharing a triconsonantal root
are closer than semantically adjacent forms built on different roots — and a
subword ablation against **word2vec** shows the advantage is largest exactly where
it should be: on rare and out-of-vocabulary forms. Applied unchanged to two
independent ETCBC biblical corpora, the model covers their vocabulary and
vectorizes every unseen form. As an application we turn to authorship: averaged
document vectors separate same- from cross-author text pairs with AUC up to
**0.90** (author-cluster bootstrap 95% CIs; stable across five seeds, 0.886 ±
0.007), validated by a negative control in which a single author split in two
yields chance AUC (≈ 0.51); mean-centering to remove a dominant anisotropic
component is essential (AUC 0.73 → 0.89). We compare against Burrows's Delta,
byte- and character-level neural language models, and a supervised verification
head (best separation, AUC 0.966), quantify a length-dependent genre confound (the
signal survives genre-matched testing, 0.900 → 0.883), and read three disputed or
pseudonymous works with historically sensible results. The honest picture: the subword
advantage is intrinsic — on the long tail — while at the document level a plain
**word2vec** is competitive with FastText and Delta, so we present authorship as
an application of the representation rather than a claim that subword embeddings
win it. Code, seeds, and trained models are released.

---

## 1. Introduction

Classical Syriac — a dialect of Aramaic and a principal language of eastern
Christianity — preserves a vast literature, much of it anonymous, pseudonymous, or
of contested authorship. Computational stylometry could help, but Syriac is
low-resource: there is no large pretrained language model, limited annotated
data, and the writing system carries optional diacritics that fragment surface
forms.

Syriac morphology is *templatic* (root-and-pattern): most words are built by
interleaving a consonantal root, typically three consonants, with vocalic and
affixal patterns. As a result, forms that share a root — e.g. ܡܠܟܐ (*malkā*,
'king') and ܡܠܟܘܬܐ (*malkūtā*, 'kingdom') — are semantically and morphologically
related but orthographically distinct. This, together with heavy inflection and a
long tail of rare forms, motivates a *subword* model: character *n*-grams let
related forms share parameters and let unseen forms receive a vector at all.

This paper makes a dual contribution — a representation-learning study and a
digital-humanities application. To our knowledge it is the first application of
distributional subword embeddings to authorship analysis in Classical Syriac; the
methods themselves are established, so we emphasize careful controls, baseline
comparison, and honest reporting over methodological novelty.

**Contributions.**

1. A released character *n*-gram **FastText** model for Classical Syriac (2.18M
   tokens), with `min_count=1` so that every form, including the 50% of the
   vocabulary that is hapax, receives a representation (§4).
2. Intrinsic validation: a morphological-coherence probe and a subword ablation
   against **word2vec**, showing the subword advantage grows for rare forms, plus
   an out-of-vocabulary generalization test (§6).
3. A representation bake-off on identical data and evaluation — char *n*-gram
   **FastText** vs. **word2vec** vs. Burrows's Delta vs. a from-scratch
   byte-level recurrent LM vs. a tiny character Transformer vs. a small
   supervised authorship-verification head (evaluated leave-one-author-out) (§7).
4. An embedding-geometry finding: averaged document vectors are strongly
   anisotropic, and label-free common-component removal is necessary for the
   authorship signal to emerge — it lifts AUC from 0.73 to 0.89 (§4, §7).
5. A robust same-/cross-author signal (centered AUC up to 0.90, bootstrap CIs,
   multi-seed) validated by a negative control (§7).
6. A quantified, length-dependent genre confound, and evidence that the
   authorship signal survives genre-matched testing (§7).
7. Attribution of three disputed/pseudonymous works with historically sensible
   outcomes, and a cross-corpus validation on two independent ETCBC biblical
   corpora that confirms vocabulary generalization and externally anchors the
   translationese finding (§7).

---

## 2. Related Work

**Subword and character embeddings.** FastText represents a word as a bag of
character *n*-grams (Bojanowski et al. 2017), extending word2vec (Mikolov et al.
2013) to compose vectors for unseen forms — attractive for morphologically rich
and low-resource languages. Character-aware neural models (Kim et al. 2016) and
token-free byte models such as ByT5 (Xue et al. 2022) pursue the same goal with
neural architectures (Vaswani et al. 2017).

**Embedding geometry.** Averaged embeddings are anisotropic: they share a
dominant common direction that inflates cosine similarities. "All-but-the-top"
postprocessing (Mu & Viswanath 2018) and analyses of contextual geometry
(Ethayarajh 2019) motivate the mean-centering we apply.

**Stylometry and authorship.** Burrows's Delta (Burrows 2002), with its
geometric and probabilistic interpretations (Argamon 2008; Evert et al. 2017), is
the standard baseline; function words are classic markers (Kestemont 2014), and
the field is surveyed in Stamatatos (2009). Topic can confound authorship
(Seroussi et al. 2014); we test an analogous genre confound. Translated texts
attribute less reliably than originals — across the European Literary Text
Collection, translations score lower than their source-language counterparts
(Schöch et al. 2024) — which frames our finding that the (translated)
Pseudo-Clementines read as translationese. We assess significance with the
cluster bootstrap (Efron & Tibshirani 1986).

**Syriac and digital classics.** Syriac NLP is nascent: recent work includes a
Transformer morphological parser that borrows Hebrew data to offset Syriac's
scarcity (Naaijer et al. 2023), but we are not aware of prior *stylometric* or
authorship-attribution work on Classical Syriac using distributional
representations. The authorship questions themselves are long-standing in
philology — for instance, the corpus attributed to Ephrem has been reassessed on
text-critical and metrical grounds (Hartung 2018); our contribution is a
complementary, quantitative perspective rather than a replacement for that
scholarship. For the language and its study see Brock (2006) and Butts (2019); our
training data is the Digital Syriac Corpus (Syriaca.org) and our independent
validation corpora are the ETCBC Syriac datasets (Vlaardingerbroek et al.).
Tooling: gensim (Řehůřek & Sojka 2010) and PyTorch (Paszke et al. 2019).

---

## 3. Corpus and Preprocessing

We use the Digital Syriac Corpus (the `srophe/syriac-corpus` TEI/XML release):
632 texts, one author credited per text for 631 of them. Table 1 summarizes the
corpus. The most striking property for a representation model is the vocabulary's
long tail: half of all distinct forms occur exactly once.

**Table 1. Corpus statistics** (after preprocessing; normalized = diacritics
stripped).

| Statistic | Value |
|---|---:|
| TEI files parsed | 632 |
| Single-author texts | 631 |
| Distinct authors | 45 |
| &nbsp;&nbsp;authors with ≥ 2 texts | 22 |
| Syriac word tokens | 2,179,065 |
| Surface word forms | 371,922 |
| Forms (diacritics stripped) | 141,595 |
| Hapax legomena (*n* = 1) | 70,870 (50.0%) |
| Rare forms (*n* ≤ 5) | 111,911 (79.0%) |

**Tokenization.** We extract maximal runs of Syriac letters (U+0710–U+074F) and
combining marks, requiring at least one letter, and discarding invisible joiners
and bidi controls. By default we *normalize* by stripping combining diacritics
(seyame, vowel points), which aligns inflected forms by their consonantal
skeleton; we report this setting throughout.

**Metadata caveats.** Genre is not encoded in these headers; we recover an
approximate genre label from the series title (§7). Texts marked *Anonymous*
collapse to a single non-author and are excluded from author-level analyses.
Authors named without a `syriaca.org` identifier are merged into their
URI-identified counterpart by name.

---

## 4. Model and Representations

**FastText.** We train a skip-gram FastText model: 100 dimensions, character
*n*-grams of length 2–5, $2\times10^{5}$ hash buckets, window 5, 10 epochs, and
crucially `min_count=1` so every form is in the vocabulary. Each TEI text is one
"sentence." The rationale is templatic morphology (§1): shared character
*n*-grams tie together forms of a common root, and the same mechanism synthesizes
vectors for forms unseen in training.

**Document and word vectors.** A document is the frequency-weighted mean of its
word vectors. For stylometry we also consider a *function-word* variant
restricted to the 200 most frequent forms, the topic-independent markers of
classical stylometry.

**Anisotropy and mean-centering.** Averaged document vectors share a large common
component: raw cosine similarities cluster near 1 even across authors. We
therefore subtract the corpus-mean document vector (a label-free, unsupervised
step) before computing cosines. §7 shows this is essential rather than cosmetic.

**Neural baselines.** As architectural contrasts we train two tiny causal
language models from scratch on the same corpus: a byte-level LSTM (**byte-LM**)
over UTF-8 bytes, and a small character Transformer (**char-Transformer**) over
Unicode codepoints. Both yield a document vector by mean-pooling final-layer
hidden states, and a word vector by the same pooling over an isolated form, so
they enter the same intrinsic and downstream evaluations. On 2.18M tokens these
models are deliberately small and data-limited; they probe architecture, not
scale.

---

## 5. Stylometry Methods

**Separation AUC.** Given document vectors and author labels, we score every text
pair by cosine similarity and report

$$\mathrm{AUC} = \Pr(\text{same-author pair more similar than cross-author pair}),$$

computed via the Mann–Whitney statistic; 0.5 is chance. We also report Cohen's
*d* between the two pair-similarity distributions.

**Burrows's Delta.** For each text we form relative frequencies of the top-*k*
most frequent corpus forms, *z*-score each feature across the corpus, and take the
mean absolute *z*-difference (Manhattan distance) as the Delta distance.

**Attribution.** We attribute a text to the nearest author *centroid* — cosine
for the embeddings, mean Delta for Delta — under strict leave-one-out: the
held-out text is removed from its own author's centroid before scoring. We report
top-1 and top-3 accuracy. As a learned alternative we also train a small
supervised-contrastive projection (Khosla et al. 2020) (a one-hidden-layer MLP)
on the document vectors so that same-author texts are pulled together; it is
evaluated *leave-one-author-out* (the head is trained only on other authors), so
its embeddings are out-of-sample and comparable to the unsupervised methods.

**Uncertainty.** Because text pairs are dependent, we bootstrap over *authors*
(clusters): resampling author groups with replacement ($B=1000$) and recomputing
AUC gives percentile 95% confidence intervals. We additionally retrain FastText
under five seeds to report variance.

---

## 6. Results: The Model

**Morphological coherence and the subword ablation.** Table 2 reports, for a
fixed set of 8 corpus-verified root families (49 related pairs), the mean cosine
between root-sharing forms and a frequency-matched control on a different root.
FastText shows the largest margin (+0.32 frequent, +0.29 rare). The decisive
contrast is with the no-subword word2vec baseline, whose margin is smaller and
*degrades for rare forms* (+0.125 → +0.076; accuracy 0.85 → 0.77), while FastText
stays robust (0.96 → 0.95). The neural models rank related above control but with
cosines compressed by anisotropy.

**Table 2. Morphological coherence by frequency band.** Mean cosine between
root-sharing forms (*related*) vs. a frequency-matched form on a different root
(*control*); *acc* is the fraction of pairs with related > control.

| Representation | Band | *n* | related | control | margin | acc |
|---|---|---:|---:|---:|---:|---:|
| FastText (char *n*-gram) | freq ≥ 100 | 27 | 0.601 | 0.284 | **+0.317** | 0.96 |
| FastText (char *n*-gram) | rare < 100 | 22 | 0.640 | 0.353 | **+0.287** | 0.95 |
| word2vec (no subword) | freq ≥ 100 | 27 | 0.490 | 0.365 | +0.125 | 0.85 |
| word2vec (no subword) | rare < 100 | 22 | 0.498 | 0.422 | +0.076 | 0.77 |
| byte-LM | freq ≥ 100 | 27 | 0.988 | 0.967 | +0.021 | 0.93 |
| byte-LM | rare < 100 | 22 | 0.992 | 0.968 | +0.025 | 1.00 |
| char-Transformer | freq ≥ 100 | 27 | 0.916 | 0.790 | +0.126 | 0.96 |
| char-Transformer | rare < 100 | 22 | 0.936 | 0.774 | +0.162 | 1.00 |

**Out-of-vocabulary generalization.** With models trained on a 90% file split,
9,023 forms (23.3% of the held-out vocabulary) are truly unseen. FastText
synthesizes a vector for *every* one from its character *n*-grams (word2vec:
none), and for a 300-form sample the nearest in-vocabulary neighbour shares the
three-consonant root prefix 56% of the time (Table 3) — the synthesized vectors
are morphologically coherent.

**Table 3. Out-of-vocabulary generalization.**

| Representation | OOV forms vectorized | coverage | NN shares root prefix |
|---|---:|---:|---:|
| FastText (char *n*-gram) | 9,023 / 9,023 | 100% | 169 / 300 (56%) |
| word2vec (no subword) | 0 / 9,023 | 0% | n/a |

**Hyperparameter sensitivity.** The morphology margin and downstream AUC are
stable across *n*-gram range, dimension, and skip-gram vs. CBOW (Table 4; AUC in
[0.888, 0.916]), so the results are not an artefact of a particular setting.

**Table 4. FastText hyperparameter sensitivity** (restricted cohort; base:
dim = 100, *n*-grams 2–5, skip-gram).

| Configuration | morph. cosine | centered AUC |
|---|---:|---:|
| *n*-grams 2–4 | 0.633 | 0.896 |
| *n*-grams 2–5 (default) | 0.618 | 0.898 |
| *n*-grams 3–6 | 0.606 | 0.916 |
| dim = 50 | 0.635 | 0.888 |
| dim = 200 | 0.611 | 0.901 |
| CBOW (sg = 0) | 0.585 | 0.910 |

**Neural LM quality.** Table 5 confirms the from-scratch LMs learn Syriac:
held-out bits-per-byte of 0.86 (byte-LM) and 0.82 (char-Transformer), far below
the 8-bpb uniform-byte baseline.

**Table 5. Neural language-model quality** (single seed, 2000 steps,
Apple-Silicon MPS). Perplexity is per modeling unit (byte vs. character).

| Model | Params | bits/byte | perplexity | train (s) |
|---|---:|---:|---:|---:|
| byte-LM (LSTM, UTF-8 bytes) | 1,020,672 | 0.861 | 1.8 | 102 |
| char-Transformer (codepoints) | 289,824 | 0.823 | 2.8 | 107 |

---

## 7. Results: Authorship

**Same- vs. cross-author separation.** Table 6 reports AUC with author-cluster
bootstrap CIs for the full and restricted cohorts, all-words and function-words.
Mean-centered AUC reaches 0.885 (full) and 0.900 (restricted, all words); the raw
(uncentered) figures are far lower (e.g. 0.73 vs. 0.89 full), demonstrating that
the anisotropic common component must be removed. Across five training seeds the
restricted AUC is 0.886 ± 0.007, so the signal is stable.

**Table 6a. Same/cross-author separation AUC** (mean-centered cosine;
author-cluster bootstrap 95% CI, *B* = 1000). AUC = 0.5 is chance.

| Cohort | Features | AUC | 95% CI |
|---|---|---:|---|
| Full (≥ 2 texts; 20 authors, 494 texts) | all words | 0.885 | [0.791, 0.961] |
| Full | function words | 0.840 | [0.758, 0.942] |
| Restricted (≥ 3 texts, ≥ 2000 tok; 11 authors, 201 texts) | all words | 0.900 | [0.830, 0.950] |
| Restricted | function words | 0.890 | [0.838, 0.952] |

**Table 6b. Variation across five FastText seeds** (all words, centered).

| Cohort | mean ± SD AUC | min | max |
|---|---:|---:|---:|
| Full | 0.874 ± 0.006 | 0.870 | 0.885 |
| Restricted | 0.886 ± 0.007 | 0.880 | 0.900 |

**Negative control.** Splitting a single author (Ephrem) into two pseudo-authors
over 20 random halves yields mean AUC ≈ 0.51: the pipeline finds no boundary where
there is none, so the cross-author signal is genuine.

**Representation bake-off.** Table 7 compares all representations on AUC and
attribution. Among *unsupervised* methods the count-based ones are strongest and
statistically comparable — word2vec (0.946), FastText (0.915), and Delta (0.907)
at the 1000-token floor — with overlapping CIs; document-level mean pooling washes
out FastText's subword advantage, which is intrinsic rather than downstream. The
tiny from-scratch neural LMs are data-limited, the char-Transformer (0.845) ahead
of the byte-LM (0.762). A small *supervised* contrastive AV head trained on the
FastText vectors and evaluated leave-one-author-out gives the best separation AUC
(0.966); a deliberately leaky variant that trains on the test author reaches
0.999, so the LOAO number is a genuine out-of-sample gain. The head does not
improve closed-set attribution (top-1 0.922), as its objective targets pairwise
verification, not nearest-centroid classification.

**Table 7. Representation bake-off.** Centered cosine AUC (author-cluster
bootstrap 95% CI) and leave-one-out nearest-centroid attribution (11 authors).
All rows are unsupervised except the AV head, which is trained leave-one-author-out
(no test author seen in training) on the FastText vectors; **bold** marks the best
AUC per token floor.

| Floor | Representation | AUC | 95% CI | top-1 | top-3 |
|---|---|---:|---|---:|---:|
| ≥ 1000 tok | FastText (char *n*-gram) | 0.915 | [0.866, 0.956] | 0.930 | 0.973 |
| ≥ 1000 tok | word2vec | 0.946 | [0.873, 0.977] | 0.965 | 0.988 |
| ≥ 1000 tok | byte-LM | 0.762 | [0.695, 0.853] | 0.680 | 0.855 |
| ≥ 1000 tok | char-Transformer | 0.845 | [0.791, 0.925] | 0.812 | 0.961 |
| ≥ 1000 tok | Burrows's Delta (MFW 100) | 0.907 | — | 0.957 | 0.992 |
| ≥ 1000 tok | Burrows's Delta (MFW 200) | 0.898 | — | 0.969 | 0.988 |
| ≥ 1000 tok | *AV head (LOAO, supervised)* | **0.966** | [0.877, 0.991] | 0.922 | 0.980 |
| ≥ 2000 tok | FastText (char *n*-gram) | 0.900 | [0.830, 0.950] | 0.930 | 0.965 |
| ≥ 2000 tok | word2vec | 0.938 | [0.842, 0.976] | 0.975 | 0.990 |
| ≥ 2000 tok | byte-LM | 0.763 | [0.696, 0.866] | 0.721 | 0.871 |
| ≥ 2000 tok | char-Transformer | 0.819 | [0.789, 0.900] | 0.836 | 0.960 |
| ≥ 2000 tok | Burrows's Delta (MFW 100) | 0.940 | — | 0.970 | 0.985 |
| ≥ 2000 tok | Burrows's Delta (MFW 200) | 0.924 | — | 0.975 | 0.985 |
| ≥ 2000 tok | *AV head (LOAO, supervised)* | **0.954** | [0.779, 0.987] | 0.905 | 0.960 |

**Genre confound.** Within Ephrem, hymns vs. prose separate weakly when short
hymns dominate but strongly at matched length (AUC 0.59 → 0.86); genre/register is
thus a real confound. Yet restricting cross-author comparisons to same-genre pairs
barely lowers separation (Table 8; 0.900 → 0.883), so the authorship signal is not
merely a genre effect.

**Table 8. Genre-matched cross-author separation** (restricted cohort). Genre
classifier coverage 60%; labels: *mēmrē* 106, prose 9, letter 4, *madrāšē* 1,
other 81.

| Cross-author pairs | AUC | same pairs | cross pairs |
|---|---:|---:|---:|
| Unmatched (all genres) | 0.900 | 3,872 | 16,228 |
| Genre-matched (same genre) | 0.883 | 3,645 | 5,202 |

**Disputed texts.** Table 9 applies the attributor (leave-one-out top-1 0.93 on
known authors) to three held-out cases. The letter transmitted under Ephrem's name
ranks him *last* of eleven and at the 0th percentile of his genuine texts; the
Pseudo-Clementines cluster with translated works (Paul, Eusebius) rather than with
each other; and the Chronicle of Zuqnin is nearest Eusebius, a historiographic
register.

**Table 9. Attribution of held-out disputed texts** against known-author
centroids.

| Text (files) | Tokens | Nearest known authors (cosine) / verdict |
|---|---:|---|
| Letter to Mar Papa, under Ephrem's name (690) | 1,126 | Paul 0.48, Aphrahat 0.44, Eusebius 0.36; Ephrem ranks **last (11/11)**, 0th percentile of his genuine texts ⇒ *not* Ephrem. |
| Pseudo-Clementines (219–227) | 1.7k–13k | Per text nearest Paul / Eusebius (other Greek→Syriac translations); group cohesion 0.70 < best external 0.78 ⇒ translationese. |
| Chronicle of Zuqnin, "Pseudo-Dionysius" (519) | 25,395 | Eusebius 0.75, Dionysius bar Ṣalibi 0.57, Aphrahat 0.56 ⇒ historiographic register. |

**Cross-corpus validation.** To test that these findings are not an artefact of a
single corpus, we apply the model unchanged to two independent, openly licensed
ETCBC corpora — the Syriac New Testament (a translation from Greek) and the
Peshitta Old Testament (from Hebrew) — neither seen in training. Coverage
generalizes strongly (Table 10a): the entire SyrNT vocabulary is in-vocabulary save
one form, and the 12,424 out-of-vocabulary types of the more distant Peshitta are
all vectorized by FastText via character *n*-grams, 55% of them landing next to a
root-sharing neighbour. The independent SyrNT also anchors the translationese
finding (Table 10b): scoring every author by cosine to the (Greek-translated)
SyrNT, the Pseudo-Clementines rank second — more SyrNT-like than ten of eleven
genuine Syriac authors — behind only the (also translated) Pauline corpus, while
native verse composers (Ephrem, Jacob of Serugh, Narsai) rank last. An external
translation corpus thus corroborates that the Pseudo-Clementines read as
translationese.

**Table 10a. Out-of-vocabulary generalization to independent ETCBC corpora.** The
DSC-trained FastText model applied unchanged; same tokenizer and consonantal
normalization. root-NN = fraction of OOV forms whose nearest in-vocabulary
neighbour shares the 3-consonant prefix.

| Corpus | tokens | types | type cov. | token cov. | OOV types | root-NN |
|---|---:|---:|---:|---:|---:|---:|
| SyrNT (Greek source) | 109,715 | 16,422 | 100.0% | 100.0% | 1 | n/a |
| Peshitta (Hebrew source) | 426,286 | 45,148 | 72.5% | 95.4% | 12,424 | 55% |

(word2vec covers 0 of the OOV types by construction; FastText vectorizes all via
character *n*-grams.)

**Table 10b. Translationese, anchored externally.** Mean-centered cosine of each
author/disputed-group centroid to the independent SyrNT corpus.

| rank | cos. to SyrNT | entity |
|---:|---:|---|
| 1 | 0.965 | *Paul the Apostle* (genuine, but itself a Greek→Syriac translation) |
| 2 | **0.773** | Pseudo-Clementine Recognitions/Homilies (disputed) |
| 7 | 0.386 | Peshitta OT (Hebrew-source translation) |
| 8 | 0.374 | Chronicle of Zuqnin (disputed) |
| … | | native Syriac poets last: Ephrem −0.03, Jacob of Serugh −0.37, Narsai −0.63 |

---

## 8. Discussion

The results separate two questions that are often conflated. *Where does the
subword inductive bias help?* Intrinsically and on the long tail: FastText's
morphological margin is largest for rare forms, and it alone assigns a vector to
every one of the 23% of held-out forms that are out-of-vocabulary — vectors that
are root-consistent for a majority of cases (a 56% nearest-neighbour root match)
and that carry over to two unseen biblical corpora, including the more distant
Peshitta whose 12,424 out-of-vocabulary types are all synthesized from character
*n*-grams. *What wins the document-level stylometry task?* Here a plain word2vec
is competitive with, even slightly ahead of, FastText: averaging hundreds of word
vectors per document washes out the subword detail, and frequency-profile methods
(Delta) remain strong — so the subword advantage is intrinsic, not a
document-level stylometry win.

Two methodological findings generalize beyond Syriac. First, averaged document
vectors are strongly anisotropic, and the authorship signal emerges only after
label-free common-component removal (AUC 0.73→0.89) — an inexpensive, transferable
preprocessing step for embedding-based stylometry. Second, where labels exist, a
small supervised-contrastive verification head over the same vectors, evaluated
leave-one-author-out, gives the strongest separation (AUC 0.966), showing that the
embeddings carry authorial signal a learned metric can sharpen.

The authorship signal itself is genuine — it passes a negative control and
survives genre matching — and reflects style rather than topic alone. The
Pseudo-Clementine result is a reminder that translated texts carry the
translator's, not the named author's, fingerprint, a reading independently
corroborated by the external SyrNT corpus. Finally, the tiny from-scratch neural
LMs trail the count-based methods: on 2.18M tokens they are data-starved, which is
itself the low-resource condition this language presents. We expect the broader
recipe — a `min_count`=1 character *n*-gram model with mean-centered document
vectors — to transfer to other templatic, low-resource languages such as Classical
Arabic, Hebrew, Ge'ez, and other Aramaic dialects, where the same hapax-heavy,
root-and-pattern morphology obtains.

---

## 9. Limitations

The most important limitation is scale: the attributed author pool is small
(11 in the restricted cohort, 20 in the full) and drawn from a *single* authored
corpus.
Although we validate that the representation generalizes to two independent
biblical corpora, those are translations that add register and reference points,
*not* new authors, so they do not enlarge the pool. The absolute AUC values
should therefore be read as cohort-specific and as supporting *relative*
comparisons between representations rather than as population estimates of
attributability for Syriac at large. The few author clusters also limit the
author-level bootstrap itself: with so few groups the resampled confidence
intervals are necessarily coarse and likely optimistic, so we read them as
indicative rather than exact.

Several further caveats apply. We normalize by stripping combining diacritics
(Section 3), which aligns inflected forms by their consonantal skeleton but
discards seyame and vocalic information that can distinguish otherwise identical
skeletons; we do not test robustness to retaining diacritics. Author identity is
taken from corpus metadata, and our merging of name-only authors into their
URI-identified counterparts (Section 3) can introduce label noise. Composition
date is uncontrolled, so part of the cross-author signal may track diachronic
change rather than individual style — an analogue of the genre confound we do
test. Finally, the disputed-text studies are illustrative, without
expert-adjudicated gold labels; the neural baselines are tiny and from-scratch (no
large pretrained Syriac LM exists), so they probe architecture under data
scarcity, not an upper bound; the supervised AV head is likewise trained on only
ten-odd authors, so its margin may change in either direction as the pool grows;
and the genre classifier is an approximate series-title heuristic.

---

## 10. Conclusion

We presented a released character *n*-gram FastText model for Classical Syriac
and the controls needed to use it. The subword inductive bias is well suited to a
templatic, hapax-heavy, low-resource language — most clearly intrinsically and on
the long tail, where it stays morphologically coherent on rare forms and
vectorizes every out-of-vocabulary form, including across two unseen biblical
corpora. We further show that averaged document vectors are anisotropic and that
label-free mean-centering is what makes a downstream signal usable. As an
application, the resulting document vectors carry a robust, controlled authorship
signal — competitive with Burrows's Delta, and sharpened further by a supervised
verification head — that yields historically sensible readings of disputed texts.
Future work includes per-stanza representations for short verse, an ablation that
retains diacritics, diachronic and larger multi-corpus author pools, larger
pretrained byte-level models, and further disputed dossiers.

**Reproducibility.** All experiments are seeded and driven by released scripts
(`fasttext_model.py`, `stylometry.py`, `authorship.py`, `nn_baselines.py`,
`paper_experiments.py`); the corpus is CC BY 4.0 (Syriaca.org); trained models are
released.

---

## References

- Argamon, S. (2008). Interpreting Burrows's Delta: Geometric and probabilistic foundations. *Literary and Linguistic Computing* 23(2), 131–147.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *TACL* 5, 135–146.
- Brock, S. P. (2006). *An Introduction to Syriac Studies.* Gorgias Press.
- Burrows, J. (2002). 'Delta': a measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing* 17(3), 267–287.
- Butts, A. M. (2019). The Classical Syriac language. In D. King (ed.), *The Syriac World*, 222–242. Routledge.
- Efron, B., & Tibshirani, R. (1986). Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy. *Statistical Science* 1(1), 54–75.
- Ethayarajh, K. (2019). How contextual are contextualized word representations? *EMNLP-IJCNLP*, 55–65.
- Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017). Understanding and explaining Delta measures for authorship attribution. *Digital Scholarship in the Humanities* 32(suppl. 2), ii4–ii16.
- Hartung, B. (2018). The authorship and dating of the Syriac corpus attributed to Ephrem of Nisibis: A reassessment. *Zeitschrift für Antikes Christentum* 22(2), 296–321.
- Kestemont, M. (2014). Function words in authorship attribution: From black magic to theory? *CLfL*, 59–66.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. *NeurIPS* 33, 18661–18673.
- Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2016). Character-aware neural language models. *AAAI*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS* 26.
- Mu, J., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. *ICLR*.
- Naaijer, M., Sikkel, C., Coeckelbergs, M., Attema, J., & Van Peursen, W. Th. (2023). A Transformer-based parser for Syriac morphology. *Proceedings of the Ancient Language Processing Workshop (ALP)*, 23–29. INCOMA Ltd.
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS* 32.
- Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *LREC Workshop on New Challenges for NLP Frameworks*, 45–50.
- Schöch, C., Dudar, J., Fileva, E., & Šeļa, A. (2024). Multilingual stylometry: The influence of language on the performance of authorship attribution using corpora from the European Literary Text Collection (ELTeC). *Computational Humanities Research (CHR)*, CEUR-WS vol. 3834.
- Seroussi, Y., Zukerman, I., & Bohnert, F. (2014). Authorship attribution with topic models. *Computational Linguistics* 40(2), 269–310.
- Stamatatos, E. (2009). A survey of modern authorship attribution methods. *JASIST* 60(3), 538–556.
- Syriaca.org. *The Digital Syriac Corpus.* https://syriaccorpus.org/ (srophe/syriac-corpus; accessed 2026).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS* 30.
- Vlaardingerbroek, H., Roorda, D., & van Peursen, W. ETCBC Syriac datasets: Syriac New Testament and Peshitta in Text-Fabric. https://github.com/ETCBC/syrnt, https://github.com/ETCBC/peshitta (MIT; accessed 2026).
- Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models. *TACL* 10, 291–306.

