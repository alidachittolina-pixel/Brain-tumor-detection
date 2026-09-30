<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b1e3d,50:1466a8,100:2ec6d6&height=220&section=header&text=Brain%20Tumour%20Detection&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=YOLOv7-tiny%20vs%20YOLOv8n%20on%20T1%20MRI%20Slices&descAlignY=58&descSize=17&descColor=bfeaf5" width="100%"/>

<img src="https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif" width="260"/>

### 🧠 A controlled comparison of two compact CNN detectors for localizing and classifying brain tumours in contrast-enhanced T1 MRI 🩺

<br/>

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8n-00b894?style=for-the-badge)
![YOLOv7](https://img.shields.io/badge/YOLOv7--tiny-Detector-5f27cd?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-f9ab00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-f37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

*An educational research prototype that fine-tunes and compares two one-stage object detectors on a fixed patient-level split, then evaluates them on a held-out test set with common metric code and visual error analysis.*

</div>

---

## 🔬 What Does This Project Do?

This project answers one research question with a controlled experiment:

> **Under the same patient-level split and training budget, does YOLOv8n improve localization and classification over YOLOv7-tiny?**

Both detectors are fine-tuned from pretrained COCO weights on **2D contrast-enhanced T1 MRI slices** and asked to draw a bounding box, a confidence score, and a tumour class for each slice.

🧩 **Three tumour classes:** `glioma` · `meningioma` · `pituitary`

The pipeline keeps everything that *can* be shared identical between the two models — image size, batch size, epoch budget, optimizer, learning rate, augmentation policy, random seed, and GPU — and documents the framework differences that remain.

---

## 🏥 Dataset

| Property | Value |
|---|---|
| 📚 Source | Jun Cheng brain-tumour dataset |
| 🖼️ Slices | **3,064** T1 contrast-enhanced MRI images |
| 🧑‍🤝‍🧑 Patients | **233** |
| 🏷️ Classes | glioma `0`, meningioma `1`, pituitary `2` |
| 📦 Archive | ~350 MB ZIP (images + YOLO labels + CSV manifest) |
| ✂️ Split | Patient-level **70 / 15 / 15**, fixed with seed **41** |

Because several slices can come from the **same patient**, the split is done **per patient** — no patient ever appears in more than one subset. This prevents the model from "recognizing the patient" and inflating its test score. A **SHA-256 fingerprint** of the patient assignment makes any later change detectable.

```
162 patients → train   ·   34 → validation   ·   37 → test
2,027 train slices   ·   496 val slices   ·   541 test slices
✅ zero patient leakage
```

> ⚠️ The dataset contains **no healthy / no-tumour controls** — every image has a tumour. As a result, specificity and true-negative rate **cannot** be estimated.

---

## 🛠️ How It Works — Pipeline Phases

| Phase | 🔎 What happens |
|:---:|---|
| **1️⃣ Scope** | Fix the research task, inputs, classes, metrics, and clinical exclusion in a machine-readable spec |
| **2️⃣ Audit data** | Load the manifest, inspect class balance, assert expected patient/slice counts |
| **3️⃣ Annotations** | Convert binary tumour masks → tight YOLO boxes (`class x_center y_center w h`), then audit all 3,064 pairs numerically **and** visually |
| **4️⃣ Split** | Freeze and fingerprint the patient-level train/val/test partition |
| **5️⃣ Train** | Fine-tune YOLOv7-tiny and YOLOv8n under one shared training budget (50 epochs, SGD, T4 GPU) |
| **6️⃣ Evaluate** | Score both models on the untouched test set with the same metric code |
| **7️⃣ Report** | Results, conclusions, limitations, and future work |

---

## 📊 Results — Independent Test Set (541 slices, 37 patients)

<div align="center">

| Metric | YOLOv7-tiny | 🏆 YOLOv8n | Δ Difference |
|---|:---:|:---:|:---:|
| 🎯 Precision | 0.699 | **0.759** | 🟢 +0.060 |
| 🔁 Recall | 0.649 | **0.821** | 🟢 +0.172 |
| ⚖️ F1-score | 0.673 | **0.789** | 🟢 +0.116 |
| 📈 mAP@0.5 | 0.708 | **0.842** | 🟢 +0.135 |
| 📉 mAP@0.5:0.95 | 0.416 | **0.507** | 🟢 +0.090 |
| 🔬 Small-tumour recall | 0.706 | **0.860** | 🟢 +0.154 |
| 🧑‍⚕️ Patient-macro recall | 0.663 | **0.809** | 🟢 +0.146 |
| ⚡ Runtime / image | 36.02 ms | **21.79 ms** | 🟢 −14.23 ms |
| 🚀 Throughput | 27.76 img/s | **45.89 img/s** | 🟢 +18.13 img/s |

</div>

✨ **YOLOv8n won on every metric and in every tumour class.** It cut end-to-end latency by ~39.5%, processed ~65.3% more images per second, and reduced missed slices from **107 → 21** and class errors from **43 → 17**. The one trade-off: slightly more localization errors (60 vs 47) — higher recall did not remove every failure mode.

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics%208.4.115-00B894?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![seaborn](https://img.shields.io/badge/seaborn-4c72b0?style=flat-square)
![Pillow](https://img.shields.io/badge/Pillow-PIL-yellow?style=flat-square)

</div>

- 🧠 **Detectors:** YOLOv7-tiny (pinned commit) · YOLOv8n (`ultralytics==8.4.115`)
- 🔥 **Framework:** PyTorch on a Google Colab **Tesla T4** GPU
- 📐 **Data & viz:** NumPy, pandas, Matplotlib, seaborn, Pillow
- 💾 **Storage:** Google Drive (dataset archive, checkpoints, evaluation artefacts)

---

## ▶️ How to Run

1. 🗂️ Open `brain_tumor_detection_final_project.ipynb` in **Google Colab** with a **T4 GPU** runtime.
2. 🔌 Run the first cells to mount Google Drive and install the pinned dependencies.
3. ⬆️ Upload the prepared dataset archive (~350 MB ZIP) when prompted.
4. ▶️ Run the notebook **top to bottom** — package versions, the YOLOv7 commit, the random seed, patient assignments, checkpoints, and epoch completion are all validated in code.
5. 📁 Checkpoints, figures, and CSV metrics are written to Google Drive.

> 💡 The test set is never used for training, hyperparameter selection, or checkpoint selection — it stays untouched until Phase 6.

---

## ⚠️ Limitations

- 🚫 **No healthy controls** — specificity and normal-MRI performance can't be measured.
- 🧲 **Single modality** — contrast-enhanced T1 only (no T2 / FLAIR).
- 🩻 **2D only** — each slice is processed independently, without 3D tumour context.
- 🏛️ **Single source dataset** — results may depend on its scanners, protocols, and population.
- 👥 **Limited metadata** — age, sex, and ethnicity were unavailable, so no fairness analysis.
- 📦 **Boxes from masks** — rectangles simplify irregular tumour boundaries.
- ⚖️ **Class imbalance** — fewer meningioma slices means wider uncertainty for that class.
- 🎲 **Single seed (41)** — multiple runs are needed to estimate training variability.

---

## 🧭 Final Conclusion

Within the limits of this dataset and design, **YOLOv8n was the stronger detector** — better aggregate accuracy, better per-class performance, higher small-tumour and patient-level recall, fewer missed cases, and faster inference than YOLOv7-tiny. These results support choosing **YOLOv8n for the project demonstration**, but they do **not** establish clinical effectiveness or generalization beyond this public dataset.

---

<div align="center">

### 🧠🩺 Built as an educational deep-learning research project ⚕️🔬

<sub>Not for clinical use · Research prototype only</sub>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2ec6d6,50:1466a8,100:0b1e3d&height=120&section=footer" width="100%"/>

</div>
