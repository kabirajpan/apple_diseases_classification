#!/bin/bash

echo "================================"
echo "STARTING COMPLETE PIPELINE"
echo "================================"
echo "Start time: $(date)"
echo ""

pipeline_start=$(date +%s)

# ----------------------------------------
# STEP 1 — TRAINING
# ----------------------------------------
echo "Step 1/3: Training model..."
step_start=$(date +%s)

python surgical_boost_version.py
if [ $? -eq 0 ]; then
	echo "✅ Training complete!"
else
	echo "❌ Training failed!"
	exit 1
fi

step_end=$(date +%s)
echo "⏱ Step 1 time: $((step_end - step_start)) seconds"
echo ""
echo "================================"
echo ""

# ----------------------------------------
# STEP 2 — TTA EVALUATION
# ----------------------------------------
echo "Step 2/3: Running TTA evaluation..."
step_start=$(date +%s)

python quick_tta_eval.py
if [ $? -eq 0 ]; then
	echo "✅ Evaluation complete!"
else
	echo "❌ Evaluation failed!"
	exit 1
fi

step_end=$(date +%s)
echo "⏱ Step 2 time: $((step_end - step_start)) seconds"
echo ""
echo "================================"
echo ""

# ----------------------------------------
# STEP 3 — FULL EVALUATION
# ----------------------------------------
echo "Step 3/3: Running full evaluation..."
step_start=$(date +%s)

python evaluate_complete.py
if [ $? -eq 0 ]; then
	echo "✅ Full evaluation complete!"
else
	echo "❌ Full evaluation failed!"
	exit 1
fi

step_end=$(date +%s)
echo "⏱ Step 3 time: $((step_end - step_start)) seconds"
echo ""
echo "================================"
echo ""

# ----------------------------------------
# 🔚 END — TOTAL TIME
# ----------------------------------------
pipeline_end=$(date +%s)
total=$((pipeline_end - pipeline_start))

echo "ALL DONE!"
echo "End time: $(date)"
echo "Total pipeline time: ${total} seconds"
echo "================================"
