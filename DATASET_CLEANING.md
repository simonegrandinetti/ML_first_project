# DATASET_CLEANING Report

## Scope
This report reviews Task 1 in the notebook and proposes where to replace long notebook cells with cleaner, reusable data-transformation stages.

## Where To Integrate Cleaner Transformations

1. Cell 12 (Cleaning, Typing, And Derived Dates)
- Current issue: one very long mixed-responsibility block (missing handling, numeric coercion, datetime parsing, boolean normalization, release-year derivation).
- Suggested cleaner integration: split into a staged transformation pipeline with explicit input/output at each stage.
- Stage A: normalize blanks and known placeholder strings to missing.
- Stage B: enforce column typing (numeric, datetime, boolean).
- Stage C: apply domain plausibility masks.
- Stage D: derive canonical date fields and final release year.
- Stage E: produce a compact quality report table after transformation.

2. Cell 14 (Missing Values)
- Current issue: profiling logic mixed with plotting and semantic joins.
- Suggested cleaner integration: keep one transformation output table for missingness and one separate visualization stage.
- Transformation output should be a stable missingness dataset reused by later tasks.

3. Cell 16 and Cell 17 (Duplicates and Inconsistencies)
- Current issue: duplicate checks and categorical consistency checks are scattered.
- Suggested cleaner integration: one consolidated data integrity stage.
- Add standardized flags for each integrity rule, then aggregate once.

4. Cell 19 (Incorrect Dates and OOD)
- Current issue: date checks and numeric OOD checks are combined with reporting.
- Suggested cleaner integration: centralize all domain rules into one rule registry and apply uniformly.
- Keep separate outputs for invalid dates, out-of-domain numeric values, and suspicious edge cases.

5. Cell 21 (Feature Engineering)
- Current issue: business transformations are interleaved with documentation tables.
- Suggested cleaner integration: separate feature generation from feature catalog/reporting.
- First build engineered feature matrix, then run a dedicated feature documentation and validation step.

## Feature-Level Transformations To Prioritize

### Numeric Coercion and Validation
- stats_pageviews: numeric coercion, non-negative rule, high-tail treatment candidate.
- swear_IT, swear_EN: numeric coercion, non-negative rule.
- year, month, day: numeric coercion, plausibility ranges.
- n_sentences, n_tokens: numeric coercion, positive-only rule.
- tokens_per_sent, avg_token_per_clause: numeric coercion, positive-only rule.
- char_per_tok: numeric coercion, positive-only rule.
- lexical_density: numeric coercion, bounded in [0, 1].
- bpm: numeric coercion, plausible music range.
- centroid, rolloff, flux, rms, zcr, flatness, spectral_complexity, pitch, loudness: numeric coercion and feature-family consistency checks.
- disc_number, track_number: integer-like coercion and positive checks.
- duration_ms: numeric coercion, plausible duration range.
- popularity: bounded in [0, 100].
- modified_popularity: non-negative check.
- latitude, longitude: geographic bounds checks.

### Datetime Standardization
- birth_date, active_start, active_end, album_release_date: parse with coercion and enforce chronology rules.
- track_release_date: construct from parts with invalid-part detection.
- release_year: canonical target year using fallback logic with clear priority order.

### Categorical and Boolean Standardization
- explicit: strict mapping to boolean, then optional numeric representation.
- language, album_type, region, country, nationality: trim/case normalization and canonical category mapping.
- artist_macroarea: standardized mapping from region with explicit missing category policy.

### Duplicate and Entity Consistency
- id_song uniqueness and duplicate-row rules.
- name versus name_artist normalized-text consistency.
- artist-title duplicate pair checks.

## Cleaner Architecture Suggested For Task 1

1. Schema contract stage
- Define required columns, expected types, allowed ranges, and nullable policy.

2. Standardization stage
- Normalize null-like values, text casing, booleans, and datetime formats.

3. Validation stage
- Apply rule registry to produce row-level flags and dataset-level summary.

4. Canonical feature stage
- Build release_year and other canonical fields only after validation.

5. Engineering stage
- Generate engineered features from validated canonical inputs.

6. Reporting stage
- Produce compact artifacts: typing summary, missingness summary, integrity summary, OOD summary, engineered-feature coverage.

## High-Impact Simplifications
- Move all column lists into one centralized schema object instead of repeating inside cells.
- Move all domain rules into one centralized rule table for reuse in Task 2, Task 3, and Task 4.
- Create one canonical cleaned dataset artifact at end of Task 1 and avoid re-cleaning ad hoc in later tasks.
- Keep visual analysis separate from data transformation logic to reduce long-cell complexity.

## Expected Benefit For Task 3 and Task 4
- More stable clustering because scaling and outlier handling start from a consistent validated base.
- Lower leakage risk in supervised learning because transformation intent is explicit and reproducible.
- Faster iteration because diagnostics are generated from the same standardized transformation outputs.
