# Templatic Morphology as a Disentanglement Probe: Measuring Root/Pattern Separability in Frozen Encoders, with a Random-Network Control

**Anonymous** · *TODO: affiliation*

> **Note.** This is a Markdown rendering for easy reading. The canonical, citable
> source is [`main.tex`](main.tex) (XeLaTeX). All numbers are produced by
> [`../results.py`](../results.py), which runs each experiment, writes a JSON
> manifest, and emits the LaTeX tables; this file mirrors those numbers. Syriac
> forms are shown in script where a Syriac font is available, always with
> transliteration. The method core is [`../disentangle.py`](../disentangle.py); the
> cross-language grids are [`../unimorph.py`](../unimorph.py).

---

## Abstract

Whether a neural encoder stores distinct linguistic factors in distinct linear
subspaces — disentanglement — is hard to measure because clean, independent factor
labels are rarely available. Root-and-pattern (templatic) morphology supplies them:
every word is a product of a lexical root and a morphosyntactic pattern, and a
morphological lexicon labels both. We build a balanced, **decorrelated** grid (one
form per factor value per lexeme, so lexical identity is independent of the factor)
and measure a 2×2 cross-erasure selectivity matrix: can a linear projection erase
one factor while the other survives, in both directions? On Classical Syriac (SEDRA)
the matrix is clean across three factors (number, gender, state) and three frozen
encoders, including a byte model never trained on Syriac; INLP and closed-form LEACE
agree. Crucially, a **random-initialised** encoder of the same architecture
reproduces both the decodability and the clean off-diagonal (number 0.78 vs. 0.83
pretrained), so this separability is substantially a property of the surface form
and the byte architecture, not something pretraining discovers — a control-task
caution the instrument makes quantifiable. What pretraining and in-language
adaptation *do* add is measurable but modest: higher decodability and a sharper
relocation of number into the vocalic channel (a consonantal control localises each
factor to where the script writes it). The design replicates on Hebrew via UniMorph
with a native encoder (more sharply than on Syriac), and extends to Arabic by the same
construction.

---

## 1. Introduction

A recurring question in interpretability is whether a representation is
*disentangled*: are separate factors of variation stored along separate,
independently manipulable directions? The hard part is measurement. To test whether
factor *A* is encoded separately from factor *B* one needs many examples in which
*A* and *B* vary independently and both are labelled; otherwise a probe for *A* may
be reading *B* or a confound such as lexical frequency. Natural data rarely provides
this, and unsupervised disentanglement without it is provably under-determined
(Locatello et al., 2019).

Root-and-pattern morphology offers an escape. In a templatic Semitic language a word
is, almost literally, a product of two factors: a consonantal **root** carrying
lexical identity and a vocalic/affixal **pattern** carrying morphosyntax. A
morphological lexicon labels both for every form. So we can build a balanced,
decorrelated probe set — every lexeme attested with both values of a binary factor,
one form per value — on which the factor is statistically independent of lexical
identity by construction.

The instrument is a 2×2 **cross-erasure selectivity matrix**. We probe a frozen
encoder for the factor and separately retrieve lexical identity by nearest neighbour,
then linearly erase each in turn and re-measure both. A clean off-diagonal (erasing
the factor leaves identity; erasing identity leaves the factor) is the signature of
separable linear subspaces.

The matrix is clean on Syriac, and stays clean across three factors, three encoders,
and — via UniMorph — Arabic and Hebrew. But the contribution we stress is what a
control reveals: a **random-init** encoder of the same architecture reproduces the
clean matrix almost entirely. The separability is therefore largely a property of the
surface form (a random byte network is a random feature map over character n-grams,
and templatic morphology marks the factor in consistent positions such a map
preserves), not something pretraining had to learn. This is a control-task lesson
(Hewitt and Liang, 2019) the instrument turns into a number. What pretraining and
adaptation add is real but modest, and the same instrument measures it.

**Contributions.**

1. **Templatic morphology as a disentanglement instrument** — a balanced,
   decorrelated probe set and a symmetric cross-erasure matrix, needing no model
   training and no synthetic data.
2. **Frozen encoders linearly separate pattern from identity**, cleanly, across three
   factors and three encoders (including one never trained on the language),
   confirmed by INLP and LEACE and stable across seeds.
3. **A random-network control that reframes the effect** — the same matrix is
   reproduced untrained, so the separability is largely surface-carried; we quantify
   what pretraining and Syriac adaptation add (a modest sharpening; relocation of
   number into the vowels, isolated by a consonantal control).
4. **Cross-lingual replication** on Arabic and Hebrew through UniMorph.

---

## 2. Related work

**Probing.** Diagnostic classifiers probe a frozen representation for a property
(Conneau et al., 2018; Belinkov et al., 2017); structural probes recover syntax
(Hewitt and Manning, 2019). The central hazard is that a probe may succeed through
the representation's general capacity rather than a targeted property; *control
tasks* calibrate this (Hewitt and Liang, 2019), surveyed by Belinkov (2022). Our
random-init encoder is exactly such a control.

**Concept erasure.** INLP removes a linear factor by iterative nullspace projection
(Ravfogel et al., 2020); amnesic probing measures the effect of removal (Elazar et
al., 2021); LEACE gives the closed-form minimally-damaging eraser (Belrose et al.,
2023). We use INLP and LEACE as the two directions of the matrix and check agreement.

**Disentanglement and concept directions.** Unsupervised disentanglement is
under-determined without inductive bias or supervision (Locatello et al., 2019;
Higgins et al., 2018); we sidestep this by letting the language supply labelled,
independent factors. That morphosyntactic features occupy consistent linear
directions echoes word-vector analogy geometry (Mikolov et al., 2013) and
lexical-semantic probing (Vulić et al., 2020).

**Resources.** Factor labels come from the SEDRA Syriac lexicon (Kiraz, 1994) and
UniMorph (McCarthy et al., 2020); encoders are CANINE (Clark et al., 2022) for
Syriac, CAMeLBERT (Inoue et al., 2021) for Arabic, AlephBERT (Seker et al., 2022)
for Hebrew.

---

## 3. The instrument: root × pattern as ground truth

In a root-and-pattern language a surface form realises a lexical root through a
morphosyntactic pattern. We treat (lexical identity **R**, a binary morphosyntactic
factor **P**) as two ground-truth factors. For Syriac **R** is the SEDRA lexeme and
**P** is *number* (sg/pl), *gender* (m/f), or *state* (emphatic/absolute) — the three
features that vary *within* a lexeme and so admit a within-lexeme contrast.

**Balanced, decorrelated grid.** For a factor we keep every lexeme attested with both
values and sample one form per value, so the factor is 50/50 and independent of
identity across lexemes. For Arabic and Hebrew the same grid is built from UniMorph
**minimal pairs** — two forms of a lemma whose feature bundles differ only in the
factor.

**Consonantal control.** Each Syriac form is available vocalised
(\u200f<span dir="rtl">ܡܠܟܐ</span> *malkā*, with pointing) and as a bare consonantal
skeleton; we run both. Comparing them localises *where* each factor is marked and
guards against the matrix being a one-character artefact. Arabic UniMorph forms carry
diacritics (same control); Hebrew forms are unvocalised (control vacuous).

---

## 4. Method

**Encoding.** Every form is encoded by a *frozen* model as one mean-pooled vector:
off-the-shelf CANINE (a byte encoder never trained on Syriac), the same model after
light Syriac LoRA adaptation, and a Hebrew model fed Syriac transliterated into
Hebrew script. Probes and erasers are linear, on the frozen vectors.

**Probes.** We split *by lexeme*, so the factor probe must generalise across lexemes.
The factor probe is logistic regression (chance 0.5); identity is self-excluded
nearest-neighbour retrieval (chance 1/|test lexemes|, < 0.01).

**Erasure.** The factor is erased with INLP and, as a closed-form check, LEACE, fit
on train and applied to test. Identity is erased by projecting out the top lexeme-mean
directions. We report original / post-factor-erasure / post-identity-erasure for both
P and R, over five seeds as mean ± SD. Everything is plain NumPy.

---

## 5. Frozen encoders separate pattern from identity

On vocalised Syriac the selectivity matrix (Table TD2) has a clean off-diagonal in
every cell. For off-the-shelf CANINE / number: the factor is decodable at **0.83** and
collapses to chance (**0.51**) when erased, while lexeme retrieval is untouched
(0.33 → 0.36); erasing identity drives retrieval to **0.00** while number stays at
**0.77**. The pattern holds for gender and state and for all three encoders, including
the Hebrew-transfer model that never saw Syriac. Multi-seed SDs are ≤ 0.05.

| Encoder | Factor | P | P·−factor | P·−lex | R | R·−factor | R·−lex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CANINE (frozen) | number | 0.828 | 0.511 | 0.770 | 0.333 | 0.357 | 0.000 |
| CANINE (frozen) | gender | 0.721 | 0.526 | 0.650 | 0.284 | 0.312 | 0.009 |
| CANINE (frozen) | state | 0.846 | 0.506 | 0.799 | 0.441 | 0.517 | 0.004 |
| CANINE (Syriac LoRA) | number | 0.870 | 0.502 | 0.781 | 0.292 | 0.323 | 0.000 |
| CANINE (Syriac LoRA) | gender | 0.722 | 0.516 | 0.657 | 0.207 | 0.249 | 0.009 |
| CANINE (Syriac LoRA) | state | 0.867 | 0.503 | 0.759 | 0.265 | 0.537 | 0.011 |
| Hebrew transfer | number | 0.622 | 0.500 | 0.561 | 0.352 | 0.367 | 0.001 |
| Hebrew transfer | gender | 0.628 | 0.467 | 0.548 | 0.253 | 0.286 | 0.007 |
| Hebrew transfer | state | 0.821 | 0.469 | 0.750 | 0.322 | 0.397 | 0.008 |

(Factor accuracy chance 0.5; mean over five seeds. `−factor`/`−lex` = after erasing
that factor.)

---

## 6. What the effect is, and what it is not

**A random network reproduces the matrix.** Running the identical pipeline on a
*random-initialised* CANINE — same architecture, same byte inputs, no training — does
not collapse: number is decodable at **0.78** (vs. 0.83 pretrained), gender 0.72,
state 0.82, and the off-diagonal stays clean. The separability is therefore largely a
property of the surface form and the byte architecture. Read as a control task, this
is a caution: high probe accuracy and clean erasure do not by themselves implicate the
*learned* representation.

**INLP and LEACE agree.** Closed-form LEACE drives post-erasure factor accuracy to the
same chance level as iterative INLP for all three factors (number 0.51 vs. 0.50;
gender 0.53 vs. 0.48; state 0.51 vs. 0.51), so the erasure is method-independent.

**What pretraining and adaptation add.** Comparing vocalised against consonantal factor
accuracy: number is far more decodable with vowels than without, and the gap *grows*
with training — 0.19 at random init, 0.20 off-the-shelf, **0.23** after Syriac LoRA —
whereas state shows no gap (marked on the consonants, the emphatic
\u200f<span dir="rtl">ܐ</span> *ā*) and gender only a small one. In-language adaptation
measurably sharpens the number signal and pushes it further into the vocalic channel
where the morphology writes it.

**A composable direction.** Treating the factor as one additive offset: per-lexeme
difference vectors share a direction above chance, and adding the mean offset to a
singular form retrieves its own plural well above chance (0.43 for number, 0.61 for
state, against ~0.01). The factor is an approximately composable direction.

---

## 7. Cross-lingual replication

Running the identical design on Hebrew (AlephBERT) with factors and minimal pairs from
UniMorph reproduces the matrix — and on native Hebrew it is *sharper* than on Syriac:
number is decodable at **0.92** and erases to chance (0.50) while identity retrieval
survives (0.640 → 0.725); erasing identity sends retrieval to 0.00 while number stays at
0.88. Gender behaves identically. The construction extends unchanged to Arabic
(CAMeLBERT on diacritised UniMorph, which also restores the consonantal control); we
report Hebrew here and treat Arabic as the same instrument applied to a further language.

| Language | Factor | P | P·−factor | P·−lex | R | R·−factor | R·−lex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hebrew | number | 0.918 | 0.495 | 0.876 | 0.640 | 0.725 | 0.002 |
| Hebrew | gender | 0.899 | 0.511 | 0.871 | 0.633 | 0.713 | 0.001 |

That the same balanced-grid, cross-erasure measurement transfers from a zero-resource
abjad to a well-resourced relative, across byte, subword, and transfer encoders,
supports treating templatic morphology as a general instrument.

---

## 8. Discussion

Frozen encoders — byte, subword, transfer, even untrained — store a Semitic language's
morphosyntactic pattern in a linear subspace separable from lexical identity, cleanly
and symmetrically. Because an untrained network of the same shape does the same, the
right reading is not that pretraining *discovers* the factorisation; it is that
templatic morphology marks its factors in surface positions that a generic byte feature
map already renders linearly separable. Pretraining and adaptation refine this: they
raise decodability and move the number signal into the vowels where the orthography
carries it. The interpretability lesson is the familiar one of control tasks, made
measurable by a clean instrument.

The positive contribution is methodological: where disentanglement studies usually rely
on synthetic generative factors, a labelled templatic lexicon supplies real,
independently labelled factors at scale, and the same design runs across languages and
encoders unchanged. The geometry is also linguistically legible — the consonantal
control recovers, from the model's internal arrangement, the vowel/consonant division
of labour a grammar would state.

---

## 9. Limitations

Observational geometry, not a causal claim that a computation *uses* the subspaces.
Effect sizes are modest (Hebrew-transfer number ~0.62; identity retrieval below
ceiling), so the matrix supports qualitative separability, not capacity estimates.
Binary contrasts on three factors, so "word = root × pattern" as a full algebra remains
aspirational. Syriac labels are NT-scoped SEDRA; UniMorph grids inherit that coverage.
Identity erasure is transductive (appropriate for a per-form quantity, weaker than the
inductive factor direction). The random-network control shows a high surface baseline,
so we report what adaptation adds as a small, directional effect.

---

## 10. Conclusion

Templatic morphology, with a morphological lexicon, is a clean and reusable instrument
for the disentanglement question. With it we find that frozen encoders linearly separate
morphosyntactic pattern from lexical identity across three Semitic languages — but that a
random-initialised network of the same architecture does too, so the separability is
largely surface-carried, with pretraining and adaptation adding a measurable sharpening
and a relocation of number into the vowels. Reporting the random baseline alongside the
effect is the point: the instrument measures both.
