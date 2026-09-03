# 🖼️ Image Captioning using BLIP

A deep learning project that generates natural language descriptions for images using **BLIP (Bootstrapping Language-Image Pre-training)**. The project fine-tunes a pre-trained vision-language model on a subset of the **Flickr8k dataset** and evaluates generated captions using multi-reference BLEU scores.

## 📌 Project Overview

Image captioning is a multimodal task that combines **computer vision** and **natural language processing** to automatically generate descriptive captions for images.

In this project, a pre-trained **BLIP image captioning model** is fine-tuned on **3,000 samples from the Flickr8k dataset**. The training pipeline is optimized by freezing the vision encoder and using mixed-precision training for efficient GPU utilization.

The project includes:

* Image and caption preprocessing
* Fine-tuning a pre-trained BLIP model
* Freezing the vision encoder
* Mixed-precision training
* Beam search-based caption generation
* Multi-reference BLEU evaluation

---

## 🤖 Model

The project uses:

**`Salesforce/blip-image-captioning-base`**

BLIP is a vision-language model capable of understanding image content and generating natural language descriptions.

The implementation uses:

* `BlipProcessor`
* `BlipForConditionalGeneration`

from the Hugging Face Transformers library.

---

## 📊 Dataset

The project uses the **Flickr8k dataset**, which contains images paired with multiple human-written captions.

For efficient experimentation and fine-tuning, a subset of:

* **3,000 samples**

was randomly selected using a fixed random seed for reproducibility.

Multiple ground-truth captions are retained for evaluation.

---

## ⚙️ Training Optimization

### Freezing the Vision Encoder

The BLIP vision encoder was frozen during fine-tuning:

```python
for param in model.vision_model.parameters():
    param.requires_grad = False
```

This reduces the number of trainable parameters and focuses training on adapting the remaining model components to the captioning task.

### Mixed-Precision Training

The training pipeline uses PyTorch automatic mixed precision and gradient scaling to improve GPU efficiency.

Key components include:

* `torch.amp.autocast`
* `torch.amp.GradScaler`

---

## 🔄 Training Process

The model was fine-tuned for:

* **3 epochs**

During training, the model learns to generate captions based on image inputs and corresponding textual descriptions.

---

## 📝 Caption Generation

Captions are generated using **beam search decoding** with:

```python
num_beams=5
```

Beam search explores multiple possible caption sequences during generation and selects a high-probability output.

The maximum generated caption length is set to:

```text
40 tokens
```

---

## 📈 Evaluation

Generated captions are evaluated against the ground-truth descriptions using **multi-reference BLEU evaluation**.

The implementation uses NLTK's `sentence_bleu` function with multiple reference captions for each image.

### Final Result

**Final Multi-Reference BLEU Score:**

```text
0.22395
```

Rounded to:

> **BLEU Score: 0.22**

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **BLIP**
* **NLTK**
* **Pandas**
* **Pillow**
* **Jupyter Notebook**



## 📊 Key Results

| Metric                     |      Result |
| -------------------------- | ----------: |
| Training Samples           |       3,000 |
| Training Epochs            |           3 |
| Beam Width                 |           5 |
| Final Multi-Reference BLEU | **0.22395** |

