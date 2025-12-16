#!/bin/bash
# Quick script to check all results

echo "=========================================="
echo "STAGE 2 RESULTS CHECKER"
echo "=========================================="
echo ""

echo "1. Zero-Shot Results:"
if [ -f "experiments/zeroshot_results.json" ]; then
    echo "   ✅ Zero-shot results found!"
    echo "   Accuracy: $(cat experiments/zeroshot_results.json | grep -o '"overall_accuracy":[^,]*' | cut -d: -f2)"
else
    echo "   ⏳ Zero-shot results not ready yet"
fi
echo ""

echo "2. Training Info:"
if [ -f "experiments/training_info_enhanced.json" ]; then
    echo "   ✅ Training info found!"
    echo "   Model path: $(cat experiments/training_info_enhanced.json | grep -o '"final_model_path":"[^"]*' | cut -d'"' -f4)"
else
    echo "   ⏳ Training info not ready yet"
fi
echo ""

echo "3. Trained Model:"
if [ -d "models/qwen2vl_lora_enhanced_final" ]; then
    echo "   ✅ Trained model found!"
    echo "   Files:"
    ls -lh models/qwen2vl_lora_enhanced_final/ | tail -n +2 | awk '{print "      " $9 " (" $5 ")"}'
else
    echo "   ⏳ Trained model not ready yet"
fi
echo ""

echo "=========================================="
echo "Status Summary:"
echo "=========================================="

if [ -f "experiments/zeroshot_results.json" ] && [ -f "experiments/training_info_enhanced.json" ] && [ -d "models/qwen2vl_lora_enhanced_final" ]; then
    echo "✅ ALL COMPLETE! Stage 2 training successful!"
else
    echo "⏳ Still in progress..."
fi
