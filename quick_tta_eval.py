"""
Quick Test-Time Augmentation Evaluation - FIXED VERSION
========================================================
Fixes:
1. ✅ Removed double rescaling bug (was causing 66% accuracy)
2. ✅ Optimized augmentation parameters
3. ✅ Added proper error handling
4. ✅ Better progress tracking
5. ✅ Comprehensive metrics reporting

Expected: 92-94% TTA accuracy (up from broken 66%)
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 12
AUGMENTED_DATA_DIR = "./apple_dataset/augmented"
DATA_DIR = "./apple_dataset/raw"
MODEL_PATH = "best_mobilenetv2_surgical_boost_i3_phase2.keras"

# TTA Configuration
N_TTA_AUGMENTATIONS = 5  # More augmentations = better accuracy (but slower)

print("🚀 Enhanced TTA Evaluation - Fixed & Optimized")
print("="*60)

# Check data source
if os.path.exists(AUGMENTED_DATA_DIR):
    data_dir = AUGMENTED_DATA_DIR
    validation_split = 0.2
    print(f"✅ Using pre-augmented data: {data_dir}")
else:
    data_dir = DATA_DIR
    validation_split = 0.15
    print(f"✅ Using original data: {data_dir}")

# Load validation data (with proper rescaling)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,  # Only rescale ONCE here
    validation_split=validation_split
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    interpolation='bilinear'
)

class_names = list(val_generator.class_indices.keys())
print(f"📊 Validation samples: {val_generator.samples}")
print(f"📊 Classes: {class_names}")

# Load trained model
try:
    print("\n📥 Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {model.count_params():,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Make sure {MODEL_PATH} exists!")
    exit(1)

# Standard evaluation baseline
print("\n📊 Baseline Evaluation (No TTA):")
val_generator.reset()
standard_loss, standard_acc = model.evaluate(val_generator, verbose=1)
print(f"✅ Standard Accuracy: {standard_acc:.4f} ({standard_acc*100:.2f}%)")

# Enhanced TTA function
def enhanced_tta_evaluation(model, val_gen, n_augmentations=5):
    """
    Enhanced Test-Time Augmentation with proper normalization
    
    KEY FIX: No rescaling in TTA datagen (already normalized!)
    """
    print(f"\n🔄 Running Enhanced TTA ({n_augmentations} augmentations)...")
    print(f"⏱️  Estimated time: ~{n_augmentations * 0.3:.1f} minutes")
    
    start_time = time.time()
    
    # Get original predictions
    val_gen.reset()
    print("   Step 1/3: Getting baseline predictions...")
    original_preds = model.predict(val_gen, verbose=0)
    all_predictions = [original_preds]
    
    # ✅ FIXED: NO rescale parameter (images already normalized)
    tta_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1.0/255,  ❌ REMOVED - This was the bug!
        rotation_range=12,      # Reduced from 15 (less aggressive)
        width_shift_range=0.08,  # Reduced from 0.1
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,      # Added for more variety
        zoom_range=0.05,         # Reduced from 0.08
        brightness_range=[0.92, 1.08],  # Tighter range
        fill_mode='nearest'
    )
    
    # Load all validation images
    print("   Step 2/3: Loading validation images...")
    val_gen.reset()
    x_images = []
    y_labels = []
    
    for i in range(len(val_gen)):
        batch_x, batch_y = val_gen[i]
        x_images.append(batch_x)
        y_labels.append(batch_y)
    
    x_all = np.concatenate(x_images, axis=0)[:len(original_preds)]
    y_all = np.concatenate(y_labels, axis=0)[:len(original_preds)]
    
    print(f"   ✅ Loaded {len(x_all)} validation images")
    
    # Apply TTA augmentations
    print("   Step 3/3: Applying augmentations...")
    
    for aug_idx in range(n_augmentations):
        print(f"      TTA round {aug_idx + 1}/{n_augmentations}...", end='\r')
        
        augmented_predictions = []
        
        # Process in batches to manage memory
        for batch_start in range(0, len(x_all), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(x_all))
            batch = x_all[batch_start:batch_end]
            
            # Apply augmentation (NO RESCALING - already done!)
            aug_flow = tta_datagen.flow(
                batch,
                batch_size=len(batch),
                shuffle=False
            )
            augmented_batch = next(aug_flow)
            
            # Get predictions
            batch_preds = model.predict(augmented_batch, verbose=0)
            augmented_predictions.extend(batch_preds)
        
        all_predictions.append(np.array(augmented_predictions))
    
    print(f"\n   ✅ Completed {n_augmentations} augmentations")
    
    # Average all predictions (ensemble)
    final_predictions = np.mean(all_predictions, axis=0)
    predicted_classes = np.argmax(final_predictions, axis=1)
    
    # Get true labels
    true_classes = np.argmax(y_all, axis=1)
    
    # Calculate metrics
    tta_accuracy = np.mean(predicted_classes == true_classes)
    tta_time = time.time() - start_time
    
    return tta_accuracy, predicted_classes, true_classes, final_predictions, tta_time

# Run Enhanced TTA
print("\n" + "="*60)
print("RUNNING ENHANCED TTA")
print("="*60)

tta_acc, pred_classes, true_classes, pred_probs, tta_time = enhanced_tta_evaluation(
    model, 
    val_generator, 
    n_augmentations=N_TTA_AUGMENTATIONS
)

# Calculate improvement
improvement = tta_acc - standard_acc
improvement_pct = improvement * 100

# Results Summary
print("\n" + "="*60)
print("🎉 FINAL RESULTS")
print("="*60)
print(f"📊 Standard Accuracy:  {standard_acc:.4f} ({standard_acc*100:.2f}%)")
print(f"🚀 TTA Accuracy:       {tta_acc:.4f} ({tta_acc*100:.2f}%)")
print(f"📈 Improvement:        {improvement:+.4f} ({improvement_pct:+.2f}%)")
print(f"⏱️  TTA Time:          {tta_time/60:.2f} minutes")
print("="*60)

# Success indicators
if tta_acc >= 0.95:
    print("🏆 OUTSTANDING: 95%+ accuracy achieved!")
    print("   Ready for publication!")
elif tta_acc >= 0.93:
    print("✅ EXCELLENT: 93%+ accuracy!")
    print("   Very strong performance!")
elif tta_acc >= 0.90:
    print("✅ GOOD: 90%+ accuracy!")
    print("   Solid model performance!")
else:
    print("⚠️  FAIR: Model needs improvement")
    print("   Consider more training data or different architecture")

# Detailed Metrics
print("\n" + "="*60)
print("📋 DETAILED METRICS")
print("="*60)

# Per-class metrics
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    true_classes, pred_classes, average=None
)

print("\n📊 Per-Class Performance:")
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
print("-"*60)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<12} {precision[i]:>6.2%}      {recall[i]:>6.2%}      "
          f"{f1[i]:>6.2%}      {support[i]:>4}")

# Overall metrics
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print("-"*60)
print(f"{'Average':<12} {macro_precision:>6.2%}      {macro_recall:>6.2%}      "
      f"{macro_f1:>6.2%}      {sum(support):>4}")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(true_classes, pred_classes)
print(f"\n{'':>12}", end='')
for name in class_names:
    print(f"{name[:10]:>12}", end='')
print()
for i, name in enumerate(class_names):
    print(f"{name[:10]:>12}", end='')
    for j in range(len(class_names)):
        print(f"{cm[i,j]:>12}", end='')
    print()

# Classification Report
print("\n📋 Full Classification Report:")
print(classification_report(
    true_classes, 
    pred_classes,
    target_names=class_names, 
    digits=4
))

# Additional Analysis
print("\n" + "="*60)
print("📈 PERFORMANCE ANALYSIS")
print("="*60)

# Find best and worst performing classes
best_class_idx = np.argmax(f1)
worst_class_idx = np.argmin(f1)

print(f"\n✅ Best Performing Class:")
print(f"   {class_names[best_class_idx]}: F1={f1[best_class_idx]:.2%}")

print(f"\n⚠️  Worst Performing Class:")
print(f"   {class_names[worst_class_idx]}: F1={f1[worst_class_idx]:.2%}")

# TTA effectiveness analysis
if improvement > 0.01:
    print(f"\n✅ TTA Effectiveness: EXCELLENT")
    print(f"   +{improvement_pct:.2f}% improvement shows good augmentation strategy")
elif improvement > 0:
    print(f"\n✅ TTA Effectiveness: GOOD")
    print(f"   +{improvement_pct:.2f}% improvement shows TTA is helpful")
else:
    print(f"\n⚠️  TTA Effectiveness: MINIMAL")
    print(f"   Consider adjusting augmentation parameters")

# Model confidence analysis
avg_confidence = np.mean(np.max(pred_probs, axis=1))
print(f"\n📊 Model Confidence:")
print(f"   Average: {avg_confidence:.2%}")

if avg_confidence > 0.9:
    print(f"   Status: Very confident predictions")
elif avg_confidence > 0.75:
    print(f"   Status: Confident predictions")
else:
    print(f"   Status: Low confidence - model uncertainty detected")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE")
print("="*60)

# Save results
results_dict = {
    'standard_accuracy': float(standard_acc),
    'tta_accuracy': float(tta_acc),
    'improvement': float(improvement),
    'tta_time_minutes': float(tta_time/60),
    'n_augmentations': N_TTA_AUGMENTATIONS,
    'per_class_metrics': {
        class_names[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(len(class_names))
    }
}

import json
with open('tta_evaluation_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n💾 Results saved to: tta_evaluation_results.json")
print("\n🎯 Next steps:")
print("   1. Review per-class metrics above")
print("   2. Check confusion matrix for common errors")
print("   3. Run: python surgical_boost_plots.py for visualizations")
print("="*60)
