#!/usr/bin/env python3
"""
Learn optimal decision tree from benchmark data to predict best FlashInfer approach.
Updated for latest benchmark results with official_fa3 and batch_attention.
"""

import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def extract_features_from_result(result):
    """Extract features from a benchmark result.

    Returns 7 features (4 base + 3 derived):
    - prefill_q: Query tokens per prefill request (0 if decode-only)
    - prefill_kv: KV cache length for prefill requests (0 if decode-only)
    - decode_kv: KV cache length for decode requests (0 if prefill-only)
    - decode_batch: Number of decode requests (0 if prefill-only)
    - total_decode_tokens: decode_batch * decode_kv
    - total_prefill_tokens: prefill_q * prefill_kv
    - num_prefills: Number of prefill requests (0 or 1)
    """
    scenario_name = result['scenario_name']

    # Initialize features
    prefill_q = 0
    prefill_kv = 0
    decode_kv = 0
    decode_batch = 0

    # Check if this has prefill component using num_prefills field
    has_prefill = result.get('num_prefills', 0) > 0

    if has_prefill:
        # Extract prefill query tokens (per request, not total)
        if 'batch128_prefill_' in scenario_name:
            # Format: batch128_prefill_64 means 64 prefills with 2K tokens each
            prefill_q = 2048
            prefill_kv = 2048  # Default KV for these scenarios
        elif 'prefill_b' in scenario_name and '_q' in scenario_name:
            # Format: prefill_b1_q128_kv1k or prefill_b8_q2k_kv2k
            parts = scenario_name.split('_')
            for i, part in enumerate(parts):
                if part.startswith('q'):
                    q_str = part[1:]
                    try:
                        if q_str.endswith('k'):
                            prefill_q = int(q_str[:-1]) * 1024
                        else:
                            prefill_q = int(q_str)
                        break
                    except:
                        prefill_q = -1
        elif '_pref_q' in scenario_name:
            # Format: mixed_dec_b128_kv16k_pref_q128_kv128
            pref_part = scenario_name.split('_pref_')[1]
            parts = pref_part.split('_')
            q_str = parts[0][1:]  # Remove 'q'
            try:
                if q_str.endswith('k'):
                    prefill_q = int(q_str[:-1]) * 1024
                else:
                    prefill_q = int(q_str)
            except:
                prefill_q = -1
        else:
            prefill_q = -1

        # Extract prefill KV length
        if '_pref_' in scenario_name and '_kv' in scenario_name.split('_pref_')[1]:
            # Format: mixed_dec_b128_kv16k_pref_q128_kv128
            pref_part = scenario_name.split('_pref_')[1]
            parts = pref_part.split('_')
            for i, part in enumerate(parts):
                if part.startswith('kv') and i > 0:
                    kv_str = part[2:]  # Remove 'kv'
                    try:
                        if kv_str.endswith('k'):
                            prefill_kv = int(kv_str[:-1]) * 1024
                        else:
                            prefill_kv = int(kv_str)
                        break
                    except:
                        pass
        elif '_kv' in scenario_name and not '_pref_' in scenario_name:
            # Pure prefill scenario: prefill_b1_q128_kv1k
            parts = scenario_name.split('_kv')
            if len(parts) > 1:
                kv_str = parts[-1].split('_')[0]
                try:
                    if kv_str.endswith('k'):
                        prefill_kv = int(kv_str[:-1]) * 1024
                    elif kv_str.isdigit():
                        prefill_kv = int(kv_str)
                except:
                    pass

        # Default if not extracted
        if prefill_kv == 0 and prefill_q > 0:
            prefill_kv = 2048  # Reasonable default

    # Extract decode information
    if scenario_name.startswith('decode_b'):
        # Format: decode_b128_kv16k
        parts = scenario_name.split('_')
        decode_batch = int(parts[1][1:])
    elif 'mixed_dec_b' in scenario_name or '_dec_b' in scenario_name:
        # Format: mixed_dec_b128_kv16k_pref_q128_kv128
        parts = scenario_name.split('_')
        for part in parts:
            if part.startswith('b') and part[1:].isdigit():
                decode_batch = int(part[1:])
                break
    elif 'batch128_prefill_' in scenario_name:
        # Format: batch128_prefill_64 means 64 prefills, 64 decodes
        num_prefills = int(scenario_name.split('_')[-1])
        decode_batch = 128 - num_prefills

    if decode_batch > 0:
        # Extract decode KV length
        if '_dec_' in scenario_name or scenario_name.startswith('decode_'):
            # Find the first _kv that's NOT part of _pref_
            before_pref = scenario_name.split('_pref_')[0] if '_pref_' in scenario_name else scenario_name
            if '_kv' in before_pref:
                parts = before_pref.split('_kv')
                if len(parts) > 1:
                    kv_str = parts[-1].split('_')[0]
                    try:
                        if kv_str.endswith('k'):
                            decode_kv = int(kv_str[:-1]) * 1024
                        elif kv_str.isdigit():
                            decode_kv = int(kv_str)
                    except:
                        pass

        # Default if not extracted
        if decode_kv == 0 and decode_batch > 0:
            decode_kv = 2048  # Reasonable default

    # Calculate derived features: token counts
    # These capture the computational load more transparently than a ratio
    total_decode_tokens = decode_batch * decode_kv
    total_prefill_tokens = prefill_q * prefill_kv if prefill_q > 0 else 0
    num_prefills = result.get('num_prefills', 1 if prefill_q > 0 else 0)

    return {
        'prefill_q': prefill_q,
        'prefill_kv': prefill_kv,
        'decode_kv': decode_kv,
        'decode_batch': decode_batch,
        'total_decode_tokens': total_decode_tokens,
        'total_prefill_tokens': total_prefill_tokens,
        'num_prefills': num_prefills,
    }

def get_winner(result, tolerance=0.10):
    """Determine which approach won with tolerance-based tie-breaking.

    If multiple approaches are within tolerance of the best time,
    choose the one with the highest preference order.

    Args:
        result: Benchmark result dictionary
        tolerance: Performance tolerance (default 10%)

    Returns:
        Name of winning approach
    """
    approach_times = result['approach_times']

    # Filter out inf values
    times = {name: time for name, time in approach_times.items()
             if time != float('inf') and not np.isnan(time)}

    if not times:
        return None

    # Find the best time
    best_time = min(times.values())

    # Find all approaches within tolerance of best
    threshold = best_time * (1 + tolerance)
    candidates = {name: time for name, time in times.items() if time <= threshold}

    # Preference order (updated for new approaches)
    # Prioritize official_fa3 and batch_attention
    preference_order = [
        'official_fa3',                 # Official FA3 implementation
        'flashinfer_batch_attention',   # FlashInfer unified wrapper
        'flashinfer_separated_fa3',     # Separated FA3
        'flashinfer_mixed_fa3',         # Mixed FA3
        'flashinfer_separated_fa2',     # Separated FA2
        'flashinfer_mixed_fa2',         # Mixed FA2
    ]

    # Choose the highest-preference candidate
    for approach in preference_order:
        if approach in candidates:
            return approach

    # Fallback: return the absolute best (should not happen)
    return min(times.items(), key=lambda x: x[1])[0]

def load_data():
    """Load and prepare data for training."""
    results_file = 'results/tolerance_clean_eager_20251117_152225.json'

    X = []  # Features
    y = []  # Labels (winner)
    scenarios = []

    print(f"Loading data from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    for result in data:
        features = extract_features_from_result(result)

        # Skip unparseable scenarios
        if features['prefill_q'] < 0:
            continue

        winner = get_winner(result)

        if winner is None:
            continue

        # Feature vector: 7 features (4 base + 3 derived token counts)
        feature_vec = [
            features['prefill_q'],              # Query tokens per prefill request
            features['prefill_kv'],             # KV cache length for prefill
            features['decode_kv'],              # KV cache length for decode
            features['decode_batch'],           # Number of decode requests
            features['total_decode_tokens'],    # Total decode tokens (decode_batch * decode_kv)
            features['total_prefill_tokens'],   # Total prefill tokens (prefill_q * prefill_kv)
            features['num_prefills'],           # Number of prefill requests
        ]

        X.append(feature_vec)
        y.append(winner)
        scenarios.append(result['scenario_name'])

    return np.array(X), np.array(y), scenarios

def train_decision_tree(X, y):
    """Train decision tree and return the model."""
    print("\nTraining decision tree...")

    # Try different max_depths to find best
    best_score = 0
    best_depth = 0

    for depth in range(3, 11):
        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        scores = cross_val_score(clf, X, y, cv=5)
        mean_score = scores.mean()
        print(f"  max_depth={depth}: CV accuracy = {mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    print(f"\nBest max_depth: {best_depth} (CV accuracy: {best_score:.3f})")

    # Train final model
    clf = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    clf.fit(X, y)

    return clf

def print_decision_rules(clf, feature_names, class_names, output_dir):
    """Print human-readable decision rules."""
    print("\n" + "=" * 80)
    print("Decision Tree Rules")
    print("=" * 80)

    tree_rules = export_text(clf, feature_names=feature_names)
    print(tree_rules)

    # Save to file
    with open(os.path.join(output_dir, 'decision_tree_rules.txt'), 'w') as f:
        f.write("Decision Tree Rules\n")
        f.write("=" * 80 + "\n\n")
        f.write(tree_rules)
    print(f"\nRules saved to: {output_dir}/decision_tree_rules.txt")

def generate_python_code(clf, feature_names, class_names, output_dir):
    """Generate Python function from decision tree."""
    print("\n" + "=" * 80)
    print("Generated Python Code")
    print("=" * 80)

    tree = clf.tree_

    output_file = os.path.join(output_dir, 'predict_best_approach.py')
    with open(output_file, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Auto-generated decision tree for selecting optimal FlashInfer approach.\n')
        f.write('Generated from benchmark results.\n')
        f.write('"""\n\n')

        f.write('def select_best_approach(prefill_q, prefill_kv, decode_kv, decode_batch, total_decode_tokens, total_prefill_tokens, num_prefills):\n')
        f.write('    """\n')
        f.write('    Select the best FlashInfer approach based on workload characteristics.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        prefill_q: Query tokens per prefill request (0 if decode-only)\n')
        f.write('        prefill_kv: KV cache length for prefill requests (0 if decode-only)\n')
        f.write('        decode_kv: KV cache length for decode requests (0 if prefill-only)\n')
        f.write('        decode_batch: Number of decode requests (0 if prefill-only)\n')
        f.write('        total_decode_tokens: Total decode tokens = decode_batch * decode_kv\n')
        f.write('        total_prefill_tokens: Total prefill tokens = prefill_q * prefill_kv\n')
        f.write('        num_prefills: Number of prefill requests\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        str: Name of best approach\n')
        f.write('    """\n')

        def recurse(node, depth=0, file=f):
            indent = "    " * (depth + 1)

            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                file.write(f"{indent}if {feature} <= {threshold:.2f}:\n")
                recurse(tree.children_left[node], depth + 1, file)
                file.write(f"{indent}else:  # {feature} > {threshold:.2f}\n")
                recurse(tree.children_right[node], depth + 1, file)
            else:  # Leaf node
                class_idx = np.argmax(tree.value[node])
                class_name = class_names[class_idx]
                file.write(f"{indent}return '{class_name}'\n")

        recurse(0, 0, f)

    print(f"Python code saved to: {output_file}")

    # Also print to console
    with open(output_file, 'r') as f:
        print(f.read())

def visualize_tree(clf, feature_names, class_names, output_dir):
    """Create visual representation of decision tree."""
    print("\n" + "=" * 80)
    print("Creating Tree Visualization")
    print("=" * 80)

    plt.figure(figsize=(25, 15))
    plot_tree(clf,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)

    output_file = os.path.join(output_dir, 'decision_tree.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Tree visualization saved to: {output_file}")

def evaluate_model(clf, X, y, scenarios, output_dir):
    """Evaluate model performance."""
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    # Training accuracy
    train_acc = clf.score(X, y)
    print(f"\nTraining Accuracy: {train_acc:.3f} ({int(train_acc * len(y))}/{len(y)})")

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Feature importance
    print("\nFeature Importance:")
    feature_names_eval = ['prefill_q', 'prefill_kv', 'decode_kv', 'decode_batch',
                          'total_decode_tokens', 'total_prefill_tokens', 'num_prefills']
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i, idx in enumerate(indices):
        if importances[idx] > 0.01:
            print(f"  {i+1}. {feature_names_eval[idx]}: {importances[idx]:.3f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names_eval[i] for i in indices])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved to: {output_file}")

    # Predictions
    y_pred = clf.predict(X)

    # Confusion matrix
    print("\nConfusion Matrix:")
    classes = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=classes)

    # Print header
    print(f"\n{'Actual \\ Predicted':<30}", end='')
    for cls in classes:
        print(f"{cls[:20]:<22}", end='')
    print()
    print("-" * (30 + 22 * len(classes)))

    # Print rows
    for i, cls in enumerate(classes):
        print(f"{cls[:30]:<30}", end='')
        for j in range(len(classes)):
            print(f"{cm[i][j]:<22}", end='')
        print()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_file}")

    # Classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y, y_pred, zero_division=0)
    print(report)

    # Save report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Accuracy: {train_acc:.3f} ({int(train_acc * len(y))}/{len(y)})\n")
        f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n\n")
        f.write(report)

    # Show some errors
    print("\nSample Misclassifications (first 20):")
    errors = [(scenarios[i], y[i], y_pred[i]) for i in range(len(y)) if y[i] != y_pred[i]]
    for i, (scenario, actual, predicted) in enumerate(errors[:20]):
        print(f"  {i+1}. {scenario}")
        print(f"     Actual: {actual}, Predicted: {predicted}")

    # Save misclassifications
    with open(os.path.join(output_dir, 'misclassifications.txt'), 'w') as f:
        f.write("Misclassifications\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total misclassifications: {len(errors)}/{len(y)} ({len(errors)/len(y)*100:.1f}%)\n\n")
        for i, (scenario, actual, predicted) in enumerate(errors):
            f.write(f"{i+1}. {scenario}\n")
            f.write(f"   Actual: {actual}\n")
            f.write(f"   Predicted: {predicted}\n\n")
    print(f"\nAll misclassifications saved to: {output_dir}/misclassifications.txt")

def main():
    print("=" * 80)
    print("Decision Tree Learning for FlashInfer Approach Selection")
    print("=" * 80)

    # Create output directory
    output_dir = 'decision_tree_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Load data
    print("\nLoading data...")
    X, y, scenarios = load_data()
    print(f"Loaded {len(X)} scenarios with {X.shape[1]} features")
    print(f"Classes: {sorted(set(y))}")

    # Print class distribution
    print("\nClass distribution:")
    for cls in sorted(set(y)):
        count = sum(1 for label in y if label == cls)
        print(f"  {cls}: {count} ({count/len(y)*100:.1f}%)")

    # Train decision tree
    clf = train_decision_tree(X, y)

    # Feature names
    feature_names = ['prefill_q', 'prefill_kv', 'decode_kv', 'decode_batch',
                     'total_decode_tokens', 'total_prefill_tokens', 'num_prefills']
    class_names = sorted(set(y))

    # Print rules
    print_decision_rules(clf, feature_names, class_names, output_dir)

    # Generate Python code
    generate_python_code(clf, feature_names, class_names, output_dir)

    # Visualize tree
    visualize_tree(clf, feature_names, class_names, output_dir)

    # Evaluate
    evaluate_model(clf, X, y, scenarios, output_dir)

    print("\n" + "=" * 80)
    print("All outputs saved to:", output_dir + "/")
    print("=" * 80)

if __name__ == '__main__':
    main()
