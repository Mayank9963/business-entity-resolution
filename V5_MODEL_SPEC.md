# v5 — Exact Model Specification (for side-by-side comparison)

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution (S1 ↔ S2/S3 record linkage, metric = macro F0.5)
**Model:** `v5` (code in `amazon/v5/src/`, artifacts in `amazon/v5/work/artifacts/`)
**Written:** 2026-09-26, from the source code + run logs (`amazon/work/v5_run*.log`, `loco*.log`) + `model_meta.json`.
**Run status when written:** translit, normalise, blocking, training, and test scoring for France + India are complete. Test scoring for **US was still running**, so `v5/output/*.tsv` has **not** been written yet and v5 has **no leaderboard score yet**.

This document gives the exact rules, constants, sizes and numbers for v5. You can put another pipeline next to it and compare each stage. Section 15 has a ready-made comparison table and checklist.

---

## 0. One-screen summary

| Item | v5 value |
|---|---|
| Problem framing | Each S2/S3 record is a **query**. It links to **at most one** S1 entity: its argmax candidate, if prob ≥ threshold |
| Candidate generation | Per-country inverted index over S1 with 10 key types, IDF-weighted, df-capped, **top K = 10** S1 per query |
| Blocking recall ceiling (val) | **98.08 %** of true pairs are in the candidate set |
| Classifier | **LightGBM binary** (MIT). **Two models**: *seen* (63 feats) for US/India, *unseen / label-free* (61 feats) for other countries (France) |
| Features | 65 defined in total, all pairwise string/number similarities plus within-query context. `country` is **not** a feature |
| Thresholds | seen = **0.75** (tuned on validation); unseen = **0.90** (fixed from leave-one-country-out) |
| Validation macro F0.5 (US+India, seen model) | **0.98253** (P 0.99586, R 0.95773, singleton F 0.98048, non-singleton F 0.98265) |
| Unseen-country proxy (LOCO, label-free model @0.90) | India→US **0.9621**, US→India **0.9358**, average **0.949** |
| Training data used | **450,000** training queries sampled (3.99 M pairs + 0.44 M early-stop pairs). The config default of 650 K ran out of memory on the 16 GB laptop |
| Models/external data | Only LightGBM. No external data, APIs, LLMs or embeddings |
| Output rule | `matching_results` = the linked queries grouped per S1. `candidate_pairs` = all top-10 candidates grouped per S1 |

---

## 1. Problem framing and assumptions v5 relies on

1. **S1 is the reference.** For each S1 entity, output every S2/S3 record that matches it.
2. **At most one S1 entity per S2/S3 record.** This was checked on train: 0 violations over 7.64 M true pairs. v5 therefore flips the direction:
   - Every S2/S3 record acts as a query.
   - It picks its best S1 candidate or nothing.
   - An S1 entity can receive any number of queries. There is no cap and no cardinality constraint on the S1 side.
3. **Matches never cross countries.** Blocking only compares a query with S1 records that have the **same `country` string**. The set of countries is open: whatever labels appear in S1 are used.
4. **S2 and S3 are never compared with each other.** No clustering or transitive closure is used. Each query is decided on its own. The only shared information is the within-query context features.
5. **~26 % of S2/S3 records are distractors** (they match nothing). **5.6 % of train S1 entities are singletons.** The median is 3 matches per S1. The average is ≈ 3.46 true matches per S1 (7.64 M / 2.207 M).
6. **Decoy pattern** (found in train labels): a decoy is a real S1 name plus an added word. Examples: "X Holdings", "X Eastgate", "X Public Limited", French "X Groupe / Participations / Developpement / Holding / SNC". Records containing `formerly / fka / dba / née` are almost always true matches.
7. **Invented brand names** are concatenations of about 40 fragments (belo, halo, onyx, zeta, …).
8. **The Indic vocabulary is closed.** There are 1,518 Indic token types, identical in train and test.

Metric (`src/evaluate.py`), exactly as in the challenge README:
- For each S1 entity, compute precision P = tp / #pred and recall R = tp / #true.
- The entity scores F = 1.25·P·R / (0.25·P + R), with these special cases:
  - both lists empty → F = 1
  - exactly one list empty, or tp = 0 → F = 0
- The final score is the mean of F over **all** S1 entities in the evaluation set. The code also reports micro P/R, singleton F and non-singleton F.

---

## 2. Data sizes (as processed by v5)

| Split | S1 | S2 | S3 | Queries (S2+S3) |
|---|---|---|---|---|
| train | 2,206,821 (India 883,188 · US 1,323,633) | 5,034,616 | 5,285,603 | India 4,133,346 · US 6,186,873 |
| test | 1,732,544 (France 259,452 · India 809,986 · US 663,106) | 4,887,273 | 5,082,316 | France 1,434,993 · India 4,717,565 · US 3,817,031 |

France appears only in test (15 % of test S1).

---

## 3. Pipeline overview

```
run_all.py
 ├─ translit.py        learn Indic→Latin maps from train GT       → work/artifacts/indic_*.json
 ├─ normalize.py       name/address normalisation (6 files)       → work/norm_{train,test}_s{1,2,3}.parquet
 ├─ run_blocking.py    per split × per country, top-10 S1/query   → work/cand_{train,test}/<country>.parquet
 ├─ train.py           token rates + 2 LightGBM models + threshold → work/artifacts/{lgbm_pair_model*.txt, model_meta.json, token_rates.json}
 │                                                                  work/val_scored.parquet
 ├─ predict.py score   per test country: best S1 + prob per query  → work/best_test_<country>.parquet
 ├─ predict.py write   assemble the two TSVs                        → output/matching_results.tsv, output/candidate_pairs.tsv
 └─ validate           official utils/validate_submission.py
```

Stage timings measured on a 16 GB laptop with 16 threads and `--jobs 1`:

| Stage | Time |
|---|---|
| translit | 0.2 min |
| normalise (both splits) | 5.3 min |
| blocking, train (India / US) | 8.4 / 8.9 min |
| blocking, test (France / India / US) | 2.5 / 8.5 / 5.1 min |
| train (both models + validation) | 14.6 min |
| predict France / India | 7.5 / 25.9 min |
| predict US | still running (v4 took ≈ 17.6 min) |

Source files (1,673 lines in total): `config.py, io_utils.py, translit.py, normalize.py, blocking.py, run_blocking.py, token_rates.py, features.py, featurize.py, train.py, predict.py, evaluate.py, loco.py, run_all.py`.

---

## 4. Stage 1 — transliteration learning (`translit.py`)

The maps are learned **only from the training ground truth**. No external dictionary is used.

- **Indic character class:** `[ऀ-෿]`, i.e. U+0900–U+0DFF (Devanagari through Sinhala).
- **Pairs used:** India-only S2/S3 records whose name or address contains Indic characters, joined to the true S1 through the ground truth. This gives **1,108,761 aligned pairs**.

**Name tokens**
- Tokenise both names: lower-case, replace ASCII punctuation and `।॥` with a space, split on spaces.
- Align tokens only when both names have the **same token count**. Align them position by position.
- Count (Indic token → Latin token) whenever the S2/S3 token has Indic characters and the S1 token has none.
- Keep the most frequent Latin token if it was seen at least 2 times **and** accounts for at least 40 % of that Indic token's counts.
- Result: **1,347 tokens** learned.

**Address segments**
- Split the S2/S3 address on commas. For each segment that contains Indic characters, count it against **every** S1 address segment (lower-cased) that has no digits.
- Keep a mapping if it was seen at least 2 times and has at least 20 % share.
- Result: **16 segments** (mostly state names).

When an Indic token is not in the map, it goes through `unidecode` later in normalisation.

---

## 5. Stage 2 — normalisation (`normalize.py`)

This step is pure string processing plus small hand-written tables. It uses `multiprocessing.Pool(N_WORKERS)` with 50 K-row chunks, and files are processed in 2 M-row pieces. Each record produces 12 columns:

| Column | Meaning |
|---|---|
| `name_n` | all name tokens after cleaning (includes legal words, stopwords, digits, invented words; honorifics removed) |
| `name_core` | tokens minus LEGAL, NAME_STOP, NAME_NOISE, pure digits and **synthetic (invented-brand) tokens** |
| `name_sorted` | `name_core` tokens sorted |
| `legal` | sorted unique legal-form tokens |
| `name_alt` | core tokens of the *alternative* name (the part before dba / formerly / …) |
| `synth` | number of synthetic tokens in the main + alt name |
| `altflag` | 1 if an alt-name marker was present |
| `addr_n` | address tokens in their original order (numbers removed, canonicalised, stopwords dropped) |
| `addr_set` | sorted unique `addr_n` tokens |
| `nums` | sorted unique numbers (**string sort**; leading zeros stripped; all-zero numbers become `"0"`) |
| `postcode` | first number in `nums` that has 5 or 6 digits (after zero stripping) |
| `state` | state/region code, found only when a **whole comma segment** is a state |

### 5.1 Name normalisation (exact order)

1. **Indic tokens → Latin.** If the name contains Indic characters, look up each Indic whitespace token (punctuation removed) in the learned map. Unknown tokens are kept as they are.
2. **ASCII and case.** Apply `unidecode` if the name is non-ASCII, then `lower()` and `strip()`.
3. **Alt-name split.** Split on the regex `ALT`:
   - Markers: `doing business as | trading as | formerly known as | previously known as | now known as | also known as | formerly | d.b.a | f.k.a | a.k.a | t/a | nee`. Optional `.`, `/` and spaces are allowed inside dba, fka and aka.
   - Keep the parts that are not empty after stripping ` :-,`.
   - If there are 2 or more parts: `altflag = 1`, **main = the LAST part**, alt = the earlier parts joined.
4. **Tokenise the main name.**
   - Domains: `(https?://)?(www.)?<label>.(com|net|org|co.in|in|co|fr|biz|info|us|io)` is replaced by the label, with hyphens turned into spaces.
   - `&` and `+` become ` and `. Apostrophes are removed.
   - Split on anything that is not `[a-z0-9]`.
   - Leet repair: tokens that mix letters and digits, with more letters than digits, get `0→o 1→l 3→e 4→a 5→s 8→b`.
   - Apply the NAME_CANON map to each token (Appendix A).
   - Merge runs of single letters: `l l c → llc`, `s a s → sas`. The merged token then goes through NAME_CANON again.
5. **Drop honorifics.** Remove NAME_NOISE tokens: `smt shri sri mr mrs ms dr known`.
6. **Legal forms.** `legal` = sorted set of tokens that appear in LEGAL.
7. **Synthetic tokens.** A token is synthetic if it is alphabetic, at least 5 characters long, and can be split into **2 or more** MORPHEMES pieces of 2–5 characters (checked by dynamic programming).
8. **Core name.** `core` = tokens that are not LEGAL, not NAME_STOP, not NAME_NOISE, not pure digits and **not synthetic**.
   - If `core` is empty **and** there are no synthetic tokens, fall back to the non-stopword tokens (or to all tokens).
   - Consequence: a name made only of invented words gets an **empty** core, which sets `q_core_empty = 1`.
9. **Alt name.** The alt part gets the same treatment, giving `name_alt`. Its synthetic tokens are added to `synth`.

### 5.2 Address normalisation (exact order)

1. **Clean each comma segment.**
   - Strip whitespace and quotes.
   - If the segment is Indic, look it up in the segment map.
   - Apply unidecode and lower-case, turn `&` into ` and `, replace non-alphanumerics with spaces.
2. **State detection.** If the whole segment matches `STATE_SEG`, set `state` = code and keep only the rest of the segment (the postcode).
   - `STATE_SEG` = (a state/region phrase, or a US/India state code) optionally followed by a 5–6 digit number.
   - Phrases cover the full names of the 50 US states + DC + PR, 45 Indian state/UT name variants (the `jammu & kashmir` entry can never fire because `&` has already become `and`), and 13 French regions.
   - Codes cover **US and India codes only**. French region codes are not matched.
   - If several segments match, the last one wins.
3. **Ordinals.** Join the segments and remove ordinal suffixes: `(\d+)(st|nd|rd|th|er|eme|e)` becomes the bare number.
4. **Per-token rules.**
   - A pure number goes to `nums`, with leading zeros stripped.
   - A mixed token (e.g. `12b`, `a-12`) sends its digit groups to `nums`. The letters left over are kept only if there are at least 2 of them.
   - Apply ADDR_CANON (Appendix A): `street→st`, `road→rd`, `north→n`, `r→rue`, `ngr→nagar`, `bombay→mumbai`, etc.
   - Drop ADDR_STOP words: `no door h house plot po box near opposite behind unit suite apartment floor building`, articles, `india usa us france greater`, etc.
5. **Postcode.** `postcode` = the first element of `nums` with length 5 or 6.

### 5.3 Normalisation quirks worth knowing when comparing

- Because leading zeros are stripped before the postcode check, **French postcodes 01000–09999 and US ZIPs 0xxxx lose their `postcode`**. They stay in `nums` as a 4-digit number.
- French *départements* are not mapped. Only the full French region names are recognised as `state`.
- `nums` and `addr_set` are sorted **alphabetically**. Blocking takes "the first N" of them, so the choice is alphabetical, not positional.
- `st` in names means `saint`, but `st` in addresses means `street`.
- `co` becomes `company`, which is LEGAL and so is removed from the core.
- New in v5: `et → and` and `frs → freres` (French).

---

## 6. Stage 3 — blocking / candidate generation (`blocking.py`, `run_blocking.py`)

**Direction and scope**
- Queries are S2/S3 records. The index is built over the S1 records of **one country**.
- Countries are processed one at a time, one process each.

**Keys per record** (from `name_core`, `addr_set`, `nums`, `postcode`):

- `toks` = the core tokens with length ≥ 2.
- `words` = the first 8 tokens of `addr_set` with length ≥ 3 (alphabetical order).

| Key | Construction | df cap (`MAX_DF`) |
|---|---|---|
| `n:` | each core token | 500 |
| `k:` | phonetic skeleton of each core token, if the skeleton has length ≥ 2 | 300 |
| `p:` | every sorted pair from the unique sorted set of the **first 6** core tokens | 1500 |
| `c:` | the core name with spaces removed, if length ≥ 4 | 5000 |
| `a:` | `num|word`: take the first 3 nums, drop `"0"` and any longer than 5 digits, then pair each with every `words` entry | 150 |
| `x:` | `tok|word` for the first 4 core tokens × the first 6 `words` | 300 |
| `z:` | postcode | 300 |
| `f:` | the first 7 characters of the concatenated core name, if length ≥ 7 | 300 |
| `b:` | every pair from the first 6 `words` | 150 |
| `m:` | every pair from the first 4 nums with length ≥ 2 | 100 |

**Phonetic skeleton** (`normalize.skeleton`):
1. Replace `ph→f`, `ck→k`, `x→ks`.
2. Translate `c→k, q→k, z→s, w→v, g→j, v→b`. This merges {c, k, q}, {z, s}, {g, j} and {v, b}. `w` becomes `v` and stays separate from `b`.
3. Drop the letters `a e i o u y h`.
4. Collapse repeated letters.

Examples: `shakti`/`shkti` → `shkt`; `latex` → `ltks`.

**Index**
- Build the unique (S1 record, key) postings. Keys are hashed with polars `hash()` into a u64, and the key type is its first character.
- Compute the document frequency `df` of each key over that country's S1.
- **Drop keys with `df > MAX_DF[type]`.**
- Weight = `w = ln(N_S1_country / df)` (float32).

**Query step**
- Process queries in 25 K chunks. Join each query's keys to the index on the hash.
- `bscore` = sum of `w` over the keys the query shares with each S1 record.
- Sort by `bscore` (descending), breaking ties by S1 file order.
- `brank` = 1, 2, …; keep `brank ≤ K = 10`.

**Output:** `(id, s1_id, bscore, brank)` per country parquet.

| Candidate pairs | Count |
|---|---|
| train India | 41,105,914 |
| train US | 60,535,222 |
| test France | 14,273,246 |
| test India | 46,959,571 |
| test US | 37,377,008 |

A query can receive 0 candidates when none of its keys survive the df caps. On test France, 1,720 queries (1,434,993 − 1,433,273 scored) got no candidates, and such queries can never be matched.

**Recall** (this key set, measured on 200 K-query samples, `work/exp_block_v3.log`):

| K | 1 | 3 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| India | 94.86 % | 96.56 % | 97.08 % | **97.55 %** | 98.00 % | 98.37 % |
| US | 96.69 % | 97.93 % | 98.29 % | **98.63 %** | 98.90 % | 99.06 % |

The recall ceiling on the full validation set is **0.98077** (v1–v3 used a smaller key set with a 0.9717 ceiling; v4 = v5 key set).

---

## 7. Statistics the features depend on

| Statistic | Source | Uses labels? | Where it is used |
|---|---|---|---|
| **Token match rates** (`token_rates.py`) | train S2/S3 `name_n` and `addr_set` tokens, **excluding the validation queries**, both countries pooled | **Yes** | `tok_extra_min`, `tok_extra_nlow`, `tok_all_min`, `atok_extra_min` (seen model only) |
| **S1 core-token vocabulary** | S1 `name_core` of the split being scored, all countries | No | `q_df0_frac` |
| **Name frequencies** | count of S1 records per space-free core name, and per `core|legal`, over the split being scored with **all countries pooled** | No | `s_name_freq`, `s_name_legal_freq`, `q_name_freq`, `q_name_legal_freq` |
| **Over-representation log-ratio (`lratio`)** | computed per country on the split being scored (train during training, **test during prediction**) | No | `tok_extra_lratio_max`, `tok_all_lratio_max` (unseen model only) |

Details:

- **Token match rates.** Each token counts once per record.
  - `n` = number of records containing the token; `k` = how many of those have a true match.
  - Only tokens with `n ≥ 30` are kept.
  - `rate = (k + 20·prior) / (n + 20)`, where `prior = 0.740` (the share of S2/S3 records that match).
  - Result: 42,551 name tokens and 52,680 address tokens.
  - Prediction reuses the same file. It is **not** rebuilt on the full training data.
- **`lratio`.** Only tokens that appear in at least 30 S2/S3 names are kept.
  - `lr(t) = ln( (k_q/N_q) / ((k_s+1)/N_s) )`
  - `k_q` and `N_q` are the S2/S3 document frequency of the token and the S2/S3 record count for that country. `k_s` and `N_s` are the same for S1.
  - Why it works without labels: decoy words ("holdings", "groupe", …) are much more frequent in S2/S3 names than in S1 names.

---

## 8. Features (`features.py`) — 65 defined

Notation: q = query (S2/S3), s = S1 candidate.
- `core` = `name_core`; `concat` = core with spaces removed.
- `ratio`, `token_sort_ratio`, `token_set_ratio` and `partial_ratio` are rapidfuzz scores on a 0–100 scale.
- NaN means missing (LightGBM handles it natively).
- Gain% = share of total split gain in the **seen** model. "U" marks features used by the unseen model only; "S" marks features used by the seen model only.

### 8.1 Name similarity (17)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 1 | `nm_ratio` | ratio(core_q, core_s) | – | 0.10 |
| 2 | `nm_tsort` | token_sort_ratio(core) | – | 0.12 |
| 3 | `nm_tset` | token_set_ratio(core) | – | 0.05 |
| 4 | `nm_partial` | partial_ratio(core) | – | 0.16 |
| 5 | `nm_jw` | Jaro-Winkler normalized similarity(core), 0–1 | – | 0.27 |
| 6 | `nm_sorted_lev` | Levenshtein normalized similarity of `name_sorted` with spaces removed | – | 0.06 |
| 7 | `nm_concat_ratio` | ratio(concat_q, concat_s) | – | 0.14 |
| 8 | `nm_concat_partial` | partial_ratio(concat) | – | 0.21 |
| 9 | `nm_full_tset` | token_set_ratio(`name_n`), i.e. including legal words | – | 0.25 |
| 10 | `nm_skel_tset` | token_set_ratio of the per-token skeleton strings of core | – | 0.07 |
| 11 | `nm_alt_tset` | token_set_ratio(q.`name_alt`, core_s) | q has no alt | 0.00 |
| 12 | `nm_jacc` | Jaccard of core token sets | either core empty | 0.16 |
| 13 | `nm_inter` | size of the core token intersection | either core empty | 0.02 |
| 14 | `nm_q_ntok` | number of core tokens in q (an empty core counts as 1) | – | 0.05 |
| 15 | `nm_s_ntok` | number of core tokens in s | – | 0.03 |
| 16 | `nm_len_diff` | abs(len(concat_q) − len(concat_s)) | – | 0.10 |
| 17 | `nm_first_eq` | first core tokens are equal (0 if either is empty) | – | 0.14 |

### 8.2 Legal form (3)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 18 | `lg_q_has` | q has a legal form | – | 0.03 |
| 19 | `lg_s_has` | s has a legal form | – | 0.01 |
| 20 | `lg_jacc` | Jaccard of legal sets | either empty | 0.31 |

### 8.3 Address (8)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 21 | `ad_q_empty` | `addr_n` of q is empty | – | 0.01 |
| 22 | `ad_s_empty` | `addr_n` of s is empty | – | 0.00 |
| 23 | `ad_tset` | token_set_ratio(`addr_n`) | either empty | 0.13 |
| 24 | `ad_tsort` | token_sort_ratio(`addr_n`) | either empty | 0.12 |
| 25 | `ad_partial` | partial_ratio(`addr_n`) | either empty | 0.08 |
| 26 | `ad_jacc` | Jaccard(`addr_set`) | either empty | 0.21 |
| 27 | `ad_inter` | size of the `addr_set` intersection | either empty | 0.19 |
| 28 | `ad_skel_tset` | token_set_ratio of the skeletons of `addr_set` | either address empty | 0.42 |

### 8.4 Numbers, postcode, state (7)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 29 | `num_jacc` | Jaccard(`nums`) | either empty | 1.50 |
| 30 | `num_inter` | size of the `nums` intersection | either empty | 0.16 |
| 31 | `num_q_n` | number of nums in q | – | 0.32 |
| 32 | `num_s_n` | number of nums in s | – | 0.04 |
| 33 | `num_tset` | token_set_ratio(`nums`) | – | **2.99** |
| 34 | `post_eq` | postcodes are equal | either missing | 0.03 |
| 35 | `state_eq` | states are equal | either missing | 0.33 |

### 8.5 Blocking and within-query context (7)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 36 | `bscore` | IDF blocking score | – | 1.76 |
| 37 | `brank` | blocking rank (1–10) | – | **64.18** |
| 38 | `q_src3` | q comes from S3 | – | 0.01 |
| 39 | `bscore_rel` | bscore / max bscore over the query's candidates | – | **14.02** |
| 40 | `nm_tset_gap` | nm_tset − max nm_tset over the query's candidates | – | 0.22 |
| 41 | `ad_tset_gap` | ad_tset (NaN→0) − max over the query's candidates | – | 0.40 |
| 42 | `q_ncand` | number of candidates of the query | – | 0.03 |

### 8.6 v2 additions (11)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 43 | `q_synth` | number of synthetic tokens in q | – | 0.00 |
| 44 | `s_synth` | number of synthetic tokens in s | – | 0.00 |
| 45 | `q_core_empty` | core of q is empty | – | 0.00 |
| 46 | `nm_sub_q` | fraction of s core tokens (length ≥ 3) found as substrings of concat_q | no such tokens | 0.08 |
| 47 | `nm_sub_s` | fraction of q core tokens (length ≥ 3) found as substrings of concat_s | no such tokens | 0.37 |
| 48 | `nm_alt_sub` | fraction of s core tokens found as substrings of q's alt name (spaces removed) | no alt | 0.00 |
| 49 | `nm_initials` | acronym match: one side's initials equal the other side's concat (the side giving initials needs ≥ 2 tokens) | either empty | 0.02 |
| 50 | `q_df0_frac` | fraction of q core tokens missing from the S1 vocabulary | q core empty | 0.05 |
| 51 | `num_near` | number of q nums (length ≥ 2) that are not in s but differ from some s num (length ≥ 2) by exactly one digit | either empty | 0.05 |
| 52 | `ad_tset_rank` | rank of ad_tset within the query (min method, descending, NaN→−1) | – | 0.16 |
| 53 | `nm_tset_rank` | rank of nm_tset within the query | – | 0.05 |

### 8.7 v3 additions (5): learned decoy-word rates, **seen model only** (except `q_altflag`)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 54 | `q_altflag` | q had an alt-name marker (used by both models) | – | 0.00 |
| 55 | `tok_extra_min` S | minimum match-rate over the q `name_n` tokens **not present** in s `name_n` | no such known token | **8.62** |
| 56 | `tok_extra_nlow` S | number of those extra tokens with rate < 0.3 | – | 0.02 |
| 57 | `tok_all_min` S | minimum match-rate over all q `name_n` tokens | – | 0.26 |
| 58 | `atok_extra_min` S | minimum address-token rate over the q `addr_set` tokens not in s | – | 0.05 |

### 8.8 v5 additions (7)

| # | Feature | Definition | NaN rule | Gain% |
|---|---|---|---|---|
| 59 | `s_name_freq` | number of S1 records with the same concat core as s | – | 0.29 |
| 60 | `s_name_legal_freq` | same, keyed by concat core + legal | – | 0.16 |
| 61 | `q_name_freq` | number of S1 records with q's concat core | q core empty | 0.10 |
| 62 | `q_name_legal_freq` | same, keyed by q's concat core + legal | q core empty | 0.09 |
| 63 | `legal_eq` | legal strings are identical | both empty | 0.22 |
| 64 | `tok_extra_lratio_max` U | maximum `lratio` over the q `name_n` tokens not in s | none known | (U: 2.77) |
| 65 | `tok_all_lratio_max` U | maximum `lratio` over all q tokens (unknown token = −9) | empty name | (U: 0.37) |

- **Seen model** = all features except #64 and #65 → **63 features**.
- **Unseen model** = all features except #55–#58 → **61 features**.

Top gains in the unseen model: `brank` 59.5 %, `bscore_rel` 18.3 %, `num_tset` 5.9 %, `num_jacc` 3.3 %, `tok_extra_lratio_max` 2.8 %, `num_q_n` 1.1 %, `bscore` 0.95 %.

Features with **zero gain** in the seen model: `nm_alt_tset`, `ad_s_empty`, `s_synth`, `nm_alt_sub`, `q_altflag` (and roughly zero: `q_synth`, `q_core_empty`).

---

## 9. Training and validation (`train.py`)

### 9.1 Split design (leakage-free, precision-honest)

- **V (validation entities)** = S1 entities where `hash(entity_id, seed=42) % 100 < 1`, using the **polars `hash`** (see §14). This gives **22,056** S1 entities from US + India together.
- **Validation queries** = **every** S2/S3 query that has any V entity among its top-10 candidates. That is 959,683 queries and **9,551,852 pairs**, of which 698,729 are positive.
  - Because all of these queries are scored, every possible false merge onto a V entity is counted.
  - Only predictions whose `s1_id ∈ V` are then scored.
- **Training pool** = all remaining queries that are (a) not validation queries and (b) whose true S1 entity is not in V. The pool includes distractor queries, which have no true match.
- **Sample** = `N_TRAIN_Q` queries (seed 42).
  - The config default is 650,000, but runs at 650 K and 550 K hit `MemoryError` on the 16 GB laptop.
  - **The final model was trained with 450,000 queries.**
- **Early-stopping split** = the queries with `hash(id, seed=42) % 10 == 0`.
  - Train: **3,989,892 pairs** (294,608 positives).
  - Early-stop: **444,360 pairs** (32,840 positives).
- **Labels:** a pair is positive if (id, s1_id) is in the ground truth.
- The model is **not** refit on all data after validation.

### 9.2 LightGBM configuration (identical for both models)

```
objective=binary, learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
max_bin=255, seed=42, num_threads=0 (all), num_boost_round=2000,
early_stopping(50) on binary_logloss of the early-stop set
```

| Model | Features | Trees (best_iteration) | Best early-stop logloss |
|---|---|---|---|
| seen | 63 | **629** | 0.005224 |
| unseen | 61 | **610** | 0.005392 |

Model files: `work/artifacts/lgbm_pair_model.txt` (8.8 MB) and `lgbm_pair_model_unseen.txt` (8.6 MB).

### 9.3 Threshold selection

The full decision rule (§10) is run on the validation pairs for thresholds 0.30, 0.35, …, 0.95. Macro F0.5 is computed over V.

**Seen model:**

| thr | F0.5 | P | R | singleton F |
|---|---|---|---|---|
| 0.30 | 0.9775 | 0.9871 | 0.9663 | 0.9431 |
| 0.50 | 0.9813 | 0.9925 | 0.9631 | 0.9669 |
| 0.60 | 0.9820 | 0.9942 | 0.9610 | 0.9711 |
| 0.70 | 0.9824 | 0.9953 | 0.9591 | 0.9762 |
| **0.75** | **0.9825** | **0.9959** | **0.9577** | **0.9805** |
| 0.80 | 0.9821 | 0.9962 | 0.9557 | 0.9813 |
| 0.85 | 0.9815 | 0.9968 | 0.9528 | 0.9839 |
| 0.90 | 0.9801 | 0.9973 | 0.9475 | 0.9864 |
| 0.95 | 0.9757 | 0.9983 | 0.9332 | 0.9915 |

**Unseen model, evaluated in-country** (for reference only; its threshold is not taken from here):

| thr | F0.5 | P | R | singleton F |
|---|---|---|---|---|
| 0.70 | 0.9817 (best) | 0.9954 | 0.9580 | 0.9737 |
| 0.80 | 0.9814 | 0.9962 | 0.9541 | 0.9813 |
| **0.90 (used)** | **0.9795** | 0.9974 | 0.9456 | 0.9847 |

Selected: seen threshold = **0.75** (val F0.5 = 0.98253, non-singleton F 0.98265). Unseen threshold = **0.90**, fixed by LOCO (§11).

Validation macro F0.5 is **not split by country** in `model_meta.json`. Appendix C has a snippet to compute it.

---

## 10. Inference and decision rule (`predict.py`)

1. **Choose the model per country.**
   - If the country is in `train_countries = ["India", "US"]`, use the seen model (629 iterations, threshold 0.75).
   - Otherwise (France), use the unseen model (610 iterations, threshold 0.90).
2. **Stream the candidates.** Read the candidate parquet by row group and buffer at least 1.5 M rows.
3. **Score.** Featurise in chunks of about 250 K pairs, extended so that a query is never split across chunks, and predict.
4. **Assign.** For each query keep the argmax-probability candidate. Link it if `prob ≥ threshold`, otherwise leave the query unmatched.
   - A query never links to more than one S1.
   - No second-best fallback, no S1-side constraints, no reconciliation between S2 and S3.
5. **Features at test time** use the **test** split's S1 vocabulary, name frequencies and `lratio` (label-free and transductive on test inputs). They also use the training `token_rates.json`.
6. **Write the outputs.**
   - `matching_results.tsv`: one row per test S1. The value is the unique, sorted, comma-joined linked ids, or empty.
   - `candidate_pairs.tsv`: one row per test S1. The value is every query that has this S1 in its top 10.
   - Rows are ordered by country (sorted) and then by 8 hash buckets, **not** in input file order.
   - Matches are always a subset of candidates.
7. **Validate** with the official `validate_submission.py`.

**Test link counts so far:**

| Country | Model | Threshold | Queries scored | Records linked | Links per S1 | v4 links (thr 0.70, one model) | v4 links per S1 |
|---|---|---|---|---|---|---|---|
| France | unseen | 0.90 | 1,433,273 | **848,983** | 3.27 | 977,629 | 3.77 |
| India | seen | 0.75 | 4,711,817 | **2,690,966** | 3.32 | 2,697,475 | 3.33 |
| US | seen | 0.75 | (running) | – | – | 2,231,903 | 3.37 |

Sanity check on these counts (an estimate, not a measurement):
- Train has ≈ 3.46 true matches per S1. At validation precision 0.996 and recall 0.958, the expected number of links per S1 is ≈ 3.46 × 0.958 / 0.996 ≈ **3.33**.
- India's 3.32 lands right on that.
- v4's France value of 3.77 suggests it over-linked France, which fits its weaker leaderboard results. v5's France value of 3.27 is close to the expectation.

---

## 11. Unseen-country strategy and leave-one-country-out (LOCO) evidence

Why a second model was built:
- The leaderboard gains were much smaller than the validation gains (v1 → v3: validation +0.006, leaderboard +0.002).
- The explanation: France is unseen. Features built from label-derived decoy-word rates teach the model that an *unknown* extra word is harmless. In a new country every decoy word is unknown, so false merges follow.

**LOCO protocol** (`loco.py`):
- Train on country A only; the token rates are also built from A only.
- 400 K training queries.
- Evaluate on 0.8 % of country B's S1 entities (`hash(id, 42) % 1000 < 8`), counting every query that touches them. That is 10,501 S1 entities for US and 6,908 for India.
- Scoring uses the full decision rule and macro F0.5.

| Variant (feature set) | India→US best | US→India best | Avg @0.90 |
|---|---|---|---|
| `all` (65 feats) | 0.9356 @0.9 | 0.9333 @0.9 | 0.9345 |
| `no_lratio` (= **seen** set, 63) | 0.9469 @0.98 | 0.9281 @0.9 | 0.9316–0.9348 (two runs) |
| `v4_features` (58) | 0.9339 @0.9 | 0.9272 @0.9 | 0.9306 |
| **`label_free` (= unseen set, 61)** | **0.9621 @0.9** (P 0.992) | **0.9358 @0.9** (P 0.961) | **0.9490** |
| `label_free_no_lratio` (59) | 0.9666 @0.8 | 0.9252 @0.93 | 0.9443 |

On the unseen-country proxy, the label-free model at 0.90 scores **0.949** vs **0.937** for the seen-type model (each direction at its own best threshold). Almost all of the loss on an unseen country is **precision**.

The `label_free*` variants in `work/loco2.log` were run with an edited `VARIANTS` dict. The `loco.py` shipped in `v5/src` defines only `all`, `no_lratio` and `v4_features`.

---

## 12. Results summary and version history

| Version | Val F0.5 | Leaderboard | Blocking ceiling | Features | Threshold | Key change |
|---|---|---|---|---|---|---|
| v1 | 0.9712 | 0.9623 | 0.9717 | 42 | 0.70 | baseline (7 key types, 42 features) |
| v3 | 0.9771 | 0.9642 | 0.9717 | 58 | 0.75 | decoy-word rates, alt-name split, invented names |
| v4 | 0.9797 | not submitted | 0.9807 | 58 | 0.70 | added keys `b`, `m`, `f` (recall 97.2 % → 98.1 %) |
| **v5** | **0.9825** (seen) | pending | 0.9808 | 63 seen / 61 unseen | 0.75 / 0.90 | two models; label-free `lratio` for France; name-ambiguity features; `et`/`frs` |

Validation → leaderboard gap so far: v1 −0.0089, v3 −0.0129. The gap is attributed to France.

Top seen-model gains per version:

| Version | brank | bscore_rel | tok_extra_min |
|---|---|---|---|
| v1 | 59.8 % | 16.7 % | – |
| v3 | 58.4 % | 20.2 % | 8.2 % |
| v5 | 64.2 % | 14.0 % | 8.6 % |

The model leans heavily on the blocking rank and score.

---

## 13. Known limitations and error modes (useful when comparing)

1. **Blocking misses about 1.9 % of true pairs.** These are mostly records with no address and a generic name (every name key is over the df caps), or heavy typos combined with reordering. None of these can be recovered later.
2. **Identical names without addresses.** A query with no address, whose name exactly matches several S1 entities, has no disambiguating evidence. The v5 name-frequency features are meant to help here.
3. **Same address, shared first token** ("real systems" vs "real industries"), or house numbers that differ by one digit.
4. **Word shuffles combined with typos** get out-scored by other candidates.
5. **Leading-zero stripping** removes French and US-Northeast postcodes (§5.3). No French département mapping exists.
6. **The unseen threshold (0.90) is not tuned on France,** because that is impossible without labels. It comes from a US ↔ India proxy.
7. **Only about 4.4 % of training queries are used** (450 K of ≈ 10.3 M). There is no refit on the full data, and the token rates exclude the validation queries.
8. **Per-query greedy argmax.** No global assignment, no S1-side cap, no S2 ↔ S3 consistency, no clustering.
9. **Name frequency and vocabulary statistics pool all countries** in a split rather than being computed per country.
10. **Zero-gain features** remain in the model (listed in §8).

---

## 14. Reproducibility

- **Environment:**
  - Python 3.13.9, Windows 11, 16 GB RAM, CPU only.
  - `polars==1.44.2 pyarrow==25.0.1 numpy==2.5.2 pandas==3.0.6 scikit-learn==1.9.1 scipy==1.18.0 rapidfuzz==3.14.6 lightgbm==4.7.0 Unidecode==1.4.0`.
- **Seeds:** `SEED = 42` is used for the validation hash, sampling, the early-stop hash and LightGBM. The candidate-file hash buckets use seed 7.
  - ⚠ The V split depends on **polars' `hash`**, which is not guaranteed to be stable across polars versions.
  - To compare on the *same* V, use polars 1.44.2, or read the V entity list out of `work/val_scored.parquet` (see Appendix C).
- **Run commands:**
  - Full run: `cd v5/src && python run_all.py` (resume with `--from block|train|predict|validate`).
  - Set `ER_TRAIN_Q=450000` to reproduce the shipped model on a 16 GB machine.
- **Other environment variables:**
  - `ER_DATA`: path to the dataset.
  - `ER_WORKERS`: number of normalisation processes (default: CPUs − 2, at most 16).
  - `ER_RUN_DIR`: where `work/` and `output/` go.
- **Artifacts:**
  - `model_meta.json`: thresholds, feature lists, validation results and gains.
  - `token_rates.json` (1.8 MB).
  - `indic_name_tokens.json`, `indic_addr_segments.json`.
  - `val_scored.parquet`: the validation pairs with columns `id, s1_id, prob, prob_unseen, y`.

---

## 15. How to compare v5 with another model

### 15.1 Checklist: questions to answer for the other model

1. **Direction.** Does it decide per S2/S3 record (argmax) or per S1 entity? Does it enforce "at most one S1 per record"?
2. **Blocking.** Which key types, what K, which df caps? What is the **recall ceiling** on the same validation set? (v5: 98.08 %, K = 10.)
3. **Candidate volume.** How many test pairs? (v5: 98.6 M.)
4. **Normalisation.** Does it handle Indic scripts, alt names (dba/formerly), invented brand names, legal forms, leet, domains, state codes and postcodes? Diff it against the tables in Appendix A.
5. **Features.** Does it use country or any label-derived statistics? These hurt on France.
6. **Model.** Which learner, how many training queries, and are the hyperparameters the same?
7. **Threshold.** How was it chosen? Is there a separate treatment for unseen countries?
8. **Validation honesty.** Does validation score *every* query that could merge into a held-out entity? If not, its precision is optimistic and the numbers are **not** comparable to v5's 0.9825.
9. **Unseen-country proxy.** What are its LOCO numbers (India→US, US→India)? (v5 label-free: 0.9621 / 0.9358.)
10. **Test link counts per country** vs the ≈ 3.33 links per S1 expectation. (v5: France 3.27, India 3.32.)
11. **Leaderboard score,** if one has been submitted.

### 15.2 Comparison table (fill in the right column)

| Dimension | v5 | Other model |
|---|---|---|
| Decision unit | per S2/S3 query, argmax over S1, prob ≥ thr | |
| Blocking keys | n, k, p, c, a, x, z, f, b, m (IDF, df-capped) | |
| K per query | 10 | |
| Blocking recall ceiling (val) | 0.9808 | |
| Test candidate pairs | 98,609,825 | |
| Normalisation extras | Indic map (1,347 tokens + 16 segments), alt-name split, synthetic brand detection, leet, domain, state codes | |
| Number of features | 63 seen / 61 unseen (65 defined) | |
| Label-derived features | yes (4, seen model only) | |
| Country-specific handling | separate label-free model for unseen countries | |
| Learner | LightGBM, 127 leaves, lr 0.05, ~620 trees | |
| Training queries / pairs | 450 K / 3.99 M | |
| Validation set | 22,056 S1 (1 % hash) + all 959,683 touching queries | |
| Val macro F0.5 | 0.98253 | |
| Val precision / recall (micro) | 0.99586 / 0.95773 | |
| Val singleton F / non-singleton F | 0.98048 / 0.98265 | |
| Threshold(s) | 0.75 seen, 0.90 unseen | |
| LOCO India→US / US→India | 0.9621 / 0.9358 (label-free @0.9) | |
| Test links France / India / US | 848,983 / 2,690,966 / pending | |
| Links per S1 (France / India) | 3.27 / 3.32 | |
| Leaderboard F0.5 | pending | |
| Train + predict wall time (16 GB laptop) | ≈ 2 h end-to-end | |

---

## Appendix A — exact normalisation tables

**NAME_CANON**
```
pvt,pvte,prv,priv→private; ltd,ltda,lim,limted→limited; inc,incorp,lnc→incorporated;
corp→corporation; co,cos,comp→company; llc→llc; llp→llp; intl→international;
natl→national; mfg→manufacturing; svc,svcs,srvcs→services; mgmt→management;
assoc→associates; bros→brothers; ent,ents→enterprises; tech,techs,technology→technologies;
grp→group; hldgs→holdings; ctr,centre→center; dept→department; univ→university;
ste,soc→societe; st→saint; lndia→india; et→and; frs→freres
```
**NAME_NOISE**
```
smt shri sri mr mrs ms dr known
```
**LEGAL**
```
private limited incorporated corporation company llc llp plc lp pllc pc opc sarl sas sasu eurl sa sci snc scop selarl gmbh ag bv nv pte pty
```
**NAME_STOP**
```
the and of a an de la le les du des et l d en for m s
```
**MORPHEMES (42)**
```
aria avi belo brix calo delta drex ecto evo faye flux gild halo iri jax kelo kor lum lyra mira nex novi nyla onyx orbi pyra quo riza syn tavo umbra vera vio xylo yuma zeph zeta veo arc solio sol dova
```
**ADDR_CANON**
- Street types: street/str/strt→st, road→rd, avenue/av/avn→ave, drive/drv→dr, boulevard/bd/bld/boul→blvd, lane→ln, court→ct, place→pl, circle→cir, highway→hwy, parkway→pkwy, terrace→ter, trail→trl, square→sq, expressway→expy, freeway→fwy, turnpike→tpke, crossing→xing, point→pt, center/centre→ctr, mount→mt, fort→ft, heights→hts.
- Directions: north/south/east/west(/so)→n/s/e/w; northeast/northwest/southeast/southwest→ne/nw/se/sw.
- French: r→rue, all→allee, ch/chem→chemin, imp→impasse, rte→route, qu→quai, fbg→faubourg.
- Indian: ngr→nagar, col→colony, sec/sect→sector, dist/distt→district, mkt→market.
- Building parts: opp→opposite, nr→near, bldg→building, apt/apts/appt→apartment, flr/fl→floor, ste→suite.
- Cities: bombay→mumbai, calcutta→kolkata, madras→chennai, bangalore→bengaluru, gurgaon→gurugram.

**ADDR_STOP** (dropped after canonicalisation)
```
no door h house plot the of de la le les du des d l et po box near opposite behind unit suite apartment floor building and at via bis ter a b c none nan india usa us france greater
```

## Appendix B — full LOCO grid (`work/loco.log`, `work/loco2.log`)

Each cell is macro F0.5 at the given threshold; `(P…)` is micro precision.

India → US:

| Variant | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.93 | 0.95 | 0.97 | 0.98 |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.8974 | 0.9054 | 0.9143 | 0.9234 | 0.9356 | – | – | – | – |
| no_lratio | 0.9169 | 0.9225 | 0.9274/0.9241 | 0.9341/0.9294 | 0.9415/0.9357 | 0.9398 | 0.9422 | 0.9462 | 0.9469 |
| v4_features | 0.9102 | 0.9150 | 0.9203 | 0.9266 | 0.9339 | – | – | – | – |
| label_free | – | – | 0.9589 | 0.9612 | **0.9621** (P0.992) | 0.9598 | 0.9578 | 0.9506 | 0.9417 |
| label_free_no_lratio | – | – | 0.9665 | 0.9666 | 0.9638 | 0.9612 | 0.9577 | 0.9524 | 0.9459 |

US → India:

| Variant | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.93 | 0.95 | 0.97 | 0.98 |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.9097 | 0.9177 | 0.9245 | 0.9309 | 0.9333 | – | – | – | – |
| no_lratio | 0.8921 | 0.9018 | 0.9112/0.9140 | 0.9191/0.9229 | 0.9281/0.9275 | 0.9267 | 0.9264 | 0.9264 | 0.9254 |
| v4_features | 0.8919 | 0.9002 | 0.9103 | 0.9197 | 0.9272 | – | – | – | – |
| label_free | – | – | 0.9299 | 0.9339 | **0.9358** (P0.961) | 0.9360 | 0.9347 | 0.9335 | 0.9309 |
| label_free_no_lratio | – | – | 0.9195 | 0.9231 | 0.9247 | 0.9252 | 0.9252 | 0.9245 | 0.9218 |

Where a cell shows two values (x/y), they come from the two runs, `loco.log` and `loco2.log` (different LightGBM random draws).

## Appendix C — per-country validation breakdown (run after the US scoring finishes; uses about 1.5 GB RAM)

```python
# from v5/src
import polars as pl
from config import WORK, SEED
from evaluate import macro_f05
from io_utils import read_ground_truth
from train import assign
s1 = pl.read_parquet(WORK/"norm_train_s1.parquet", columns=["entity_id","country"])
V = s1.filter(pl.col("entity_id").hash(seed=SEED) % 100 < 1)        # same V as train.py (polars 1.44.2)
vs = pl.read_parquet(WORK/"val_scored.parquet")                     # id, s1_id, prob, prob_unseen, y
gt = read_ground_truth()
for c in ["India", "US"]:
    ids = V.filter(pl.col("country") == c)["entity_id"]
    for col, thr in [("prob", 0.75), ("prob_unseen", 0.90)]:
        pred = assign(vs.select("id","s1_id"), vs[col].to_numpy(), thr).filter(pl.col("s1_id").is_in(ids.implode()))
        r = macro_f05(pred, gt.filter(pl.col("s1_id").is_in(ids.implode())), ids.to_list())
        print(c, col, thr, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
```

