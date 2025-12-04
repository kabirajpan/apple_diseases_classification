"""
Complete Evaluation Suite for Research Paper
============================================
Generates ALL metrics needed for publication:
- Confusion Matrix (normalized + raw)
- ROC Curves + AUC
- Per-class Precision/Recall/F1
- Publication-quality plots
- LaTeX-ready tables
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
import json
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 12
AUGMENTED_DATA_DIR = "./apple_dataset/augmented"
DATA_DIR = "./apple_dataset/raw"
MODEL_PATH = "best_mobilenetv2_surgical_boost_i3_phase2.keras"
RESULTS_DIR = "evaluation_results"

print("="*70)
print("COMPREHENSIVE EVALUATION SUITE - Research Paper Ready")
print("="*70)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check data source
if os.path.exists(AUGMENTED_DATA_DIR):
    data_dir = AUGMENTED_DATA_DIR
    validation_split = 0.2
    print(f"✅ Using pre-augmented data")
else:
    data_dir = DATA_DIR
    validation_split = 0.15
    print(f"✅ Using original data")

# Load test/validation data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    validation_split=validation_split
)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
num_classes = len(class_names)

print(f"📊 Test samples: {test_generator.samples}")
print(f"📊 Classes: {class_names}")

# Load model
print(f"\n🤖 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Parameters: {model.count_params():,}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Get predictions
print(f"\n🔮 Generating predictions...")
test_generator.reset()
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

# Get true labels
test_generator.reset()
y_true = []
for i in range(len(test_generator)):
    _, batch_y = test_generator[i]
    y_true.extend(np.argmax(batch_y, axis=1))
y_true = np.array(y_true[:len(y_pred)])

# Overall accuracy
overall_accuracy = np.mean(y_pred == y_true)

print("\n" + "="*70)
print(f"OVERALL TEST ACCURACY: {overall_accuracy*100:.2f}%")
print("="*70)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n📊 Generating Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)
cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar_kws={'label': 'Proportion'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.pdf", bbox_inches='tight')
print(f"   ✅ Saved: {RESULTS_DIR}/confusion_matrix.png")
plt.close()

# =============================================================================
# PER-CLASS METRICS
# =============================================================================

print("\n📊 Computing Per-Class Metrics...")

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

# Per-class accuracy
per_class_acc = []
for i in range(num_classes):
    class_mask = (y_true == i)
    class_correct = np.sum((y_true[class_mask] == y_pred[class_mask]))
    class_acc = class_correct / np.sum(class_mask) if np.sum(class_mask) > 0 else 0
    per_class_acc.append(class_acc)

print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)
print(f"{'Class':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support'}")
print("-"*70)

for i, name in enumerate(class_names):
    print(f"{name:<12} {per_class_acc[i]:>7.2%}   {precision[i]:>7.2%}   "
          f"{recall[i]:>7.2%}   {f1[i]:>7.2%}   {support[i]:>6}")

print("-"*70)
print(f"{'Average':<12} {np.mean(per_class_acc):>7.2%}   {np.mean(precision):>7.2%}   "
      f"{np.mean(recall):>7.2%}   {np.mean(f1):>7.2%}   {np.sum(support):>6}")

# Plot per-class metrics
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(class_names))
width = 0.2

bars1 = ax.bar(x - 1.5*width, per_class_acc, width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/per_class_metrics.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{RESULTS_DIR}/per_class_metrics.pdf", bbox_inches='tight')
print(f"   ✅ Saved: {RESULTS_DIR}/per_class_metrics.png")
plt.close()

# =============================================================================
# ROC CURVES
# =============================================================================

print("\n📊 Computing ROC Curves...")

# Binarize labels
y_true_bin = label_binarize(y_true, classes=range(num_classes))

# Compute ROC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))

colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{RESULTS_DIR}/roc_curves.pdf", bbox_inches='tight')
print(f"   ✅ Saved: {RESULTS_DIR}/roc_curves.png")
plt.close()

print("\n📊 AUC Scores:")
for i, name in enumerate(class_names):
    print(f"   {name:<12}: {roc_auc[i]:.4f}")
print(f"   {'Average':<12}: {np.mean(list(roc_auc.values())):.4f}")

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

print("\n📋 Generating Classification Report...")

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

with open(f"{RESULTS_DIR}/classification_report.txt", 'w') as f:
    f.write("Classification Report\n")
    f.write("="*70 + "\n\n")
    f.write(report)

print(f"   ✅ Saved: {RESULTS_DIR}/classification_report.txt")

# =============================================================================
# SAVE JSON RESULTS
# =============================================================================

results = {
    'timestamp': datetime.now().isoformat(),
    'model_path': MODEL_PATH,
    'overall_accuracy': float(overall_accuracy),
    'num_test_samples': int(len(y_true)),
    'per_class_metrics': {
        class_names[i]: {
            'accuracy': float(per_class_acc[i]),
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
            'auc': float(roc_auc[i])
        }
        for i in range(num_classes)
    },
    'average_metrics': {
        'accuracy': float(np.mean(per_class_acc)),
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1_score': float(np.mean(f1)),
        'auc': float(np.mean(list(roc_auc.values())))
    },
    'confusion_matrix': cm.tolist(),
    'confusion_matrix_normalized': cm_normalized.tolist()
}

with open(f"{RESULTS_DIR}/evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✅ Saved: {RESULTS_DIR}/evaluation_results.json")

# =============================================================================
# MODEL SUMMARY
# =============================================================================

with open(f"{RESULTS_DIR}/model_summary.txt", 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"   ✅ Saved: {RESULTS_DIR}/model_summary.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)

print(f"\n📊 Summary:")
print(f"   Overall Accuracy: {overall_accuracy*100:.2f}%")
print(f"   Average Precision: {np.mean(precision)*100:.2f}%")
print(f"   Average Recall: {np.mean(recall)*100:.2f}%")
print(f"   Average F1-Score: {np.mean(f1)*100:.2f}%")
print(f"   Average AUC: {np.mean(list(roc_auc.values())):.4f}")

print(f"\n📁 All results saved to: {RESULTS_DIR}/")
print(f"   • confusion_matrix.png (+ PDF)")
print(f"   • per_class_metrics.png (+ PDF)")
print(f"   • roc_curves.png (+ PDF)")
print(f"   • classification_report.txt")
print(f"   • evaluation_results.json")
print(f"   • model_summary.txt")

print("\n🎯 For Research Paper:")
print(f"   1. Use PNG files for presentations")
print(f"   2. Use PDF files for LaTeX papers")
print(f"   3. Check JSON for exact numbers")

print("="*70)
