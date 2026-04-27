"""
relabel_and_retrain.py
----------------------
One-time script to:
  1. Diagnose the current label distribution (shows why CV=1.0 happened)
  2. Re-label all 547 existing AlertOutcome records using the corrected
     direction-aware labeler (compares actual return to causal signal direction)
  3. Delete the stale ml_model.pkl trained on the biased labels
  4. Retrain the ML model on the corrected dataset
  5. Print a final report

Run once from your project root:
    python3 relabel_and_retrain.py

Safe to re-run — it will skip outcomes that have no return data and
report what changed.
"""

import json
import os
import pickle
from collections import Counter

from database import SessionLocal
import database as db
from services.outcome_tracker import _auto_label, _get_expected_direction

session = SessionLocal()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Diagnose current state
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Current label distribution (before fix)")
print("=" * 60)

all_labeled = session.query(db.AlertOutcome).filter(
    db.AlertOutcome.outcome_label.notin_(["pending"])
).all()

old_dist = Counter(o.outcome_label for o in all_labeled)
total = sum(old_dist.values())
for label, count in sorted(old_dist.items(), key=lambda x: -x[1]):
    bar = "█" * int(40 * count / total)
    print(f"  {label:20s}: {count:4d}  ({100*count/total:5.1f}%)  {bar}")

print(f"\n  Total labeled: {total}")
print()

# Check for the bull-market bias: if >70% are 'profitable', the model is broken
profitable_pct = old_dist.get("profitable", 0) / max(total, 1)
if profitable_pct > 0.70:
    print(f"  ⚠  CONFIRMED: {profitable_pct:.0%} of labels are 'profitable'.")
    print(f"     This is the bull-market bias. The model learned 'always say profitable'.")
    print(f"     CV accuracy 1.0 is a sign of a broken labeler, not a good model.")
elif profitable_pct > 0.55:
    print(f"  ⚠  Moderate bias: {profitable_pct:.0%} profitable. Direction-aware fix will improve balance.")
else:
    print(f"  ✓  Label distribution looks reasonable ({profitable_pct:.0%} profitable).")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Re-label all outcomes with direction-aware logic
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2: Re-labeling with direction-aware logic")
print("=" * 60)

relabeled  = 0
unchanged  = 0
skipped    = 0
changes    = Counter()

all_outcomes = session.query(db.AlertOutcome).filter(
    db.AlertOutcome.outcome_label.notin_(["pending"])
).all()

for outcome in all_outcomes:
    # Skip if no return data — can't relabel without returns
    if outcome.forward_return_1d is None and outcome.forward_return_1w is None:
        skipped += 1
        continue

    alert = session.get(db.Alert, outcome.alert_id)
    if alert is None:
        skipped += 1
        continue

    old_label = outcome.outcome_label
    new_label = _auto_label(outcome, alert, session=session)

    if new_label != old_label:
        changes[f"{old_label} → {new_label}"] += 1
        outcome.outcome_label = new_label
        relabeled += 1
    else:
        unchanged += 1

session.commit()

print(f"  Re-labeled:  {relabeled}")
print(f"  Unchanged:   {unchanged}")
print(f"  Skipped:     {skipped} (no return data)")
print()

if changes:
    print("  Label changes:")
    for change, count in sorted(changes.items(), key=lambda x: -x[1]):
        print(f"    {change}: {count}")
    print()

# Show new distribution
print("  New label distribution after fix:")
new_labels = session.query(db.AlertOutcome).filter(
    db.AlertOutcome.outcome_label.notin_(["pending"])
).all()
new_dist = Counter(o.outcome_label for o in new_labels)
new_total = sum(new_dist.values())
for label, count in sorted(new_dist.items(), key=lambda x: -x[1]):
    bar = "█" * int(40 * count / new_total)
    print(f"    {label:20s}: {count:4d}  ({100*count/new_total:5.1f}%)  {bar}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Delete the stale model
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3: Removing stale model")
print("=" * 60)

from services.ml_predictor import FEATURE_NAMES as CURRENT_FEATURES

model_path = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        old_model = pickle.load(f)
    old_features = old_model.get("feature_names", [])
    feature_mismatch = old_features != CURRENT_FEATURES
    print(f"  Trained on:    {old_model.get('n_samples')} samples at {old_model.get('trained_at')}")
    print(f"  Old CV:        {old_model.get('outcome_cv_accuracy')}")
    print(f"  Old features:  {len(old_features)}  →  New features: {len(CURRENT_FEATURES)}")
    if feature_mismatch:
        print(f"  ⚠  Feature vector changed ({len(old_features)} → {len(CURRENT_FEATURES)}) — must retrain.")
    os.remove(model_path)
    print("  ✓ Deleted ml_model.pkl")
else:
    print("  No model file found — nothing to delete.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Retrain on corrected labels
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4: Retraining on corrected labels")
print("=" * 60)

from services.ml_predictor import train_model, MIN_TRAINING_SAMPLES

labeled_count = session.query(db.AlertOutcome).filter(
    db.AlertOutcome.outcome_label.notin_(["pending"])
).count()

print(f"  Available labeled samples: {labeled_count}")

if labeled_count < MIN_TRAINING_SAMPLES:
    print(f"  ✗ Need at least {MIN_TRAINING_SAMPLES} labeled samples. Run outcome sweep first.")
else:
    print(f"  Training model...")
    result = train_model(session)

    if result.get("trained"):
        print(f"  ✓ Model trained successfully")
        print(f"    Samples used:          {result['n_samples']}")
        print(f"    Outcome CV accuracy:   {result.get('outcome_cv_accuracy')}")
        print(f"    Direction CV accuracy: {result.get('direction_cv_accuracy')}")
        print()

        outcome_acc = result.get("outcome_cv_accuracy") or 0
        # CV now uses balanced_accuracy (mean recall per class), NOT raw accuracy.
        # A random guesser on 3 classes scores ~0.33.  Dominant-class always-predict
        # also scores ~0.33 (not 0.71).  Healthy range is 0.45–0.75.
        if outcome_acc >= 0.80:
            print("  ⚠  Balanced CV accuracy is very high (≥0.80).")
            print("     This may indicate overfitting. The model will still work correctly;")
            print("     accumulate more data and retrain to confirm generalisation.")
        elif 0.45 <= outcome_acc <= 0.80:
            print("  ✓ Balanced CV accuracy in healthy range (0.45–0.80).")
            print("    Model is learning genuine signal patterns across all outcome classes.")
        elif outcome_acc < 0.45:
            print("  ⚠  Balanced CV accuracy below 0.45 — close to random guessing.")
            print("     Accumulate more outcome data (target: 200+ labeled outcomes) and retrain.")
    else:
        print(f"  ✗ Training failed: {result.get('reason')}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Labels corrected:   {relabeled} of {total}")
print(f"  Old dominant class: profitable at {profitable_pct:.0%} (broken)")
new_profitable_pct = new_dist.get("profitable", 0) / max(new_total, 1)
print(f"  New profitable pct: {new_profitable_pct:.0%} (corrected)")
print()
print("  Next steps:")
print("  1. Let the server run — new alerts will use direction-aware labeling automatically.")
print("  2. Retrain again after 30+ more outcomes accumulate (API: POST /ml/train).")
print("  3. A healthy CV accuracy target is 0.60–0.75, not 1.0.")
print("  4. Check the Alerts History view for labeling quality.")
print()

session.close()
