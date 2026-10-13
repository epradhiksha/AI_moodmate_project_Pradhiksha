# 🎵 MoodMate — Emotion-Based Music Recommendation

MoodMate detects how you feel — from your **face** (photo or live webcam) or from a **sentence you type** — and recommends songs that match your mood, with direct Spotify links.

---

## ✨ Features

- **Image upload** – upload a photo and the app detects the emotion on the face.
- **Live webcam** – a camera window opens, draws a box around your face and shows the predicted emotion in real time; the most frequent emotion across the captured frames is used.
- **Text input** – type a sentence and a BERT model detects its emotion.
- **Song recommendations** – choose how many songs (3, 5, 10 or 20) and get a list with clickable Spotify links.

---

## 🧠 How It Works

```
 Photo / Webcam ──► Haar cascade (face) ──► VGG16 + Transformer ──┐
                                                                  ├──► Emotion ──► Song filter ──► Spotify links
 Text ───────────► BERT emotion classifier ───────────────────────┘
```

### Facial emotion detection
- Faces are located with OpenCV's **Haar cascade** (`haarcascade_frontalface_default.xml`).
- Each face is resized to **128×128**, converted to RGB and passed through VGG16 preprocessing.
- A **VGG16 + Transformer** model classifies it into 5 emotions:
  `angry`, `happy`, `sad`, `surprise`, `neutral`.
- The model was trained on a combined **FER-2013 + RAF-DB** dataset (28,361 training / 7,088 validation / 8,877 test images) and reaches **80% test accuracy**.

### Text emotion detection
- Uses the pre-trained Hugging Face model [`bhadresh-savani/bert-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion).

### Music recommendation
Songs from `Music Info.csv` are filtered by their **valence** (how positive a song sounds) and **energy**:

| Emotion  | Valence      | Energy       |
|----------|--------------|--------------|
| happy    | > 0.6        | > 0.5        |
| sad      | < 0.4        | < 0.5        |
| angry    | < 0.4        | > 0.7        |
| surprise | 0.4 – 0.7    | > 0.6        |
| neutral  | 0.4 – 0.6    | 0.4 – 0.6    |

A random sample of matching songs is returned so you get different suggestions each time.

---

## 🔬 Models Explored

Several facial emotion models were trained and compared (see the [`facial emotion detection/`](facial%20emotion%20detection/) notebooks). The MobileNetV2 + Conv head and VGG16 + Transformer models were trained on **Kaggle Notebooks** using Kaggle's free GPU (the final model on an NVIDIA Tesla T4).

| Model | Notebook | Input | Classes | Dataset |
|-------|----------|-------|---------|---------|
| CNN (from scratch) | `cnn.ipynb` | 48×48 grayscale | 7 | FER-2013 |
| MobileNetV2 (transfer learning) | `Mobilenet.ipynb` | 224×224 RGB | 7 | FER-2013 |
| MobileNetV2 + Conv head | `mobilenet-5.ipynb` | 128×128 RGB | 5 | FER-2013 |
| **VGG16 + Transformer** (final) | `vgg16+transformer.ipynb` | 128×128 RGB | 5 | FER-2013 + RAF-DB |

### Why VGG16 + Transformer?

<!-- Fill in the paper details:
The idea of combining a CNN with a Transformer comes from
[*Paper Title*](link) by Author et al. (Year).
-->
VGG16 (pre-trained on ImageNet) extracts local facial features such as the eyes, mouth and eyebrows. Transformer encoder layers (multi-head self-attention) on top learn how these regions relate to each other, which helps with subtle expressions.

---

## 📂 Project Structure

```
├── app.py                     # FastAPI app and routes
├── detectface.py              # Face detection + facial emotion prediction (image & webcam)
├── TEXT.py                    # Text emotion detection (BERT)
├── recommendation.py          # Emotion → song recommendations
├── Music Info.csv             # Music dataset (audio features + Spotify IDs)
├── facial emotion detection/  # Training notebooks and saved .keras models
├── templates/                 # HTML pages (home, emotion, sentiment)
├── static/                    # CSS and uploaded images
└── requirements.txt
```

---

## ⚙️ Setup

**Requirements:** Python 3.9.13 (Anaconda recommended), a webcam for live detection.

```bash
git clone https://github.com/epradhiksha/AI_moodmate_project_Pradhiksha.git
cd AI_moodmate_project_Pradhiksha

conda create -n moodmate python=3.9.13 -y
conda activate moodmate
pip install -r requirements.txt
```

## ▶️ Run

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

- **Image / Webcam** – upload a photo, or click *Use Webcam*. A camera window opens; press **q** to stop early.
- **Text Input** – type how you feel and submit.

> The first request is slower while the models load.

---

## 🛠️ Tech Stack

- **FastAPI**, **Jinja2**, **Uvicorn** – web app
- **TensorFlow / Keras** – facial emotion model
- **OpenCV** – face detection and webcam
- **Hugging Face Transformers**, **PyTorch** – text emotion model
- **Pandas**, **NumPy** – music data processing

---

## 🚧 Limitations & Future Work

- The face model recognises 5 emotions; fear and disgust are not included.
- The webcam window opens on the machine running the server, so it works locally only. A future version could capture video in the browser instead.
- Recommendations use simple valence/energy rules; content-based similarity (e.g. on genre and tags) could make them more personal.
- Possible additions: in-app Spotify playback, mood history, and combining face + text for a single prediction.

---

## 🙏 Acknowledgements


- **[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)** – facial expression dataset
- **RAF-DB** – Real-world Affective Faces Database (S. Li, W. Deng, J. Du, CVPR 2017)
- **Combined FER + RAF-DB dataset** on Kaggle 
- **Music dataset** – `Music Info.csv` (song metadata, Spotify audio features and IDs) 
- **[bhadresh-savani/bert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion)** – pre-trained text emotion model
- **[Kaggle](https://www.kaggle.com/)** – free GPU used to train the facial emotion models
- **ImageNet pre-trained weights** for VGG16 and MobileNetV2 via Keras Applications

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
