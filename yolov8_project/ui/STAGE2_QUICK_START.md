# 🚀 Stage 2 UI Quick Start

## One-Command Launch

```bash
cd /Users/prathamprabhu/Desktop/CLIP\ model/yolov8_project/ui
./run_stage2.sh
```

Then open: **http://localhost:8501**

---

## 🎯 What You'll See

### Sidebar (Model Selection)

**Stage 1: Object Detection**
- ☑️ YOLOv11-Best (82.3% mAP) ⭐ **Recommended**
- ☐ YOLOv11
- ☐ YOLOv8

**Stage 2: Classification**
- ☑️ CLIP-B32-Binary (80.47% accuracy) ⭐ **Available**
- ☐ None (Stage 1 only)

### Main Page

1. **Upload Image** (left side)
2. **Click "Run Analysis"**
3. **View Results** (right side)
   - Green boxes 🟢 = NORMAL objects
   - Orange boxes 🟠 = CONDITIONAL objects

---

## 📊 Example Output

```
Detection Results:
1. slope drain (NORMAL) - Confidence: 0.92
2. toe drain (CONDITIONAL) - Confidence: 0.87
3. rock toe (NORMAL) - Confidence: 0.95
4. vegetation (NORMAL) - Confidence: 0.89

Summary:
🟢 NORMAL: 3 objects
🟠 CONDITIONAL: 1 object
⏱️ Total Time: 1.2s
```

---

## 🛠️ Troubleshooting

### Stage 2 Model Not Found

If you see: "Stage 2: ❌ Model not found"

**Solution:**
```bash
cd /Users/prathamprabhu/Desktop/CLIP\ model/stage2_conditional/scripts
python3 train_binary_clip.py --epochs 8 --batch_size 32
```

Wait ~8 minutes for training to complete.

### Dependencies Missing

```bash
pip install streamlit torch ultralytics transformers pillow
```

### Port Already in Use

If 8501 is busy:
```bash
streamlit run app_stage2.py --server.port 8502
```

---

## 🎯 Testing Tips

### Test Images Location
```
/Users/prathamprabhu/Desktop/CLIP model/quen2-vl.yolov11/test/images/
```

### Good Test Images
- Images with multiple objects (3-5)
- Mix of conditions (normal + damaged/blocked)
- Clear, well-lit photos

### What to Check
- ✅ Both models load successfully
- ✅ Detections appear with green/orange boxes
- ✅ NORMAL vs CONDITIONAL labels are accurate
- ✅ Inference time < 5 seconds
- ✅ UI is responsive

---

## 📸 Expected Behavior

| Object Type | Condition | Box Color | Label Example |
|------------|-----------|-----------|---------------|
| slope drain | Normal | 🟢 Green | slope drain (NORMAL) |
| toe drain | Blocked | 🟠 Orange | toe drain (CONDITIONAL) |
| rock toe | Damaged | 🟠 Orange | rock toe (CONDITIONAL) |
| vegetation | Normal | 🟢 Green | vegetation (NORMAL) |

---

## 🎉 Success Criteria

✅ UI loads without errors  
✅ Stage 1 detects objects correctly  
✅ Stage 2 classifies as NORMAL or CONDITIONAL  
✅ Results display with color-coded boxes  
✅ Performance is reasonable (< 5 seconds)  

---

## 📞 Need Help?

See full documentation:
- **Complete Guide**: `STAGE2_UI_DEPLOYMENT_GUIDE.md`
- **Model Summary**: `../../stage2_conditional/BINARY_CLASSIFIER_SUMMARY.md`
- **Implementation Plan**: `../../plan.plan.md`

---

**Ready to test!** Just run `./run_stage2.sh` and upload an image.

