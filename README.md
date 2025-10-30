# Comment Toxicity Detection Model

This repository contains the implementation of a deep learning model designed to detect toxic comments on online platforms. The project was developed as part of the **CO542 – Neural Networks and Fuzzy Systems** module at the **Department of Computer Engineering, University of Peradeniya**.

The model classifies user comments as *toxic* or *non-toxic*, with subcategories such as *hate speech*, *threats*, and *offensive language*. The solution is based on **Transformer-based architectures** such as **BERT** and **RoBERTa**, fine-tuned for this specific NLP task.

---

## Project Overview

Online communication platforms often struggle with the spread of harmful content and hate speech. Traditional moderation systems, which rely on keyword filtering or rule-based methods, fail to detect context-dependent or implicit toxicity.

This project explores how **Transformer-based neural networks** can effectively identify and classify toxic content. By leveraging contextual embeddings and transfer learning, the model aims to support real-time content moderation for online communities.

---

## Model Design

### Architecture
- **Input Layer:** Tokenized text using the WordPiece tokenizer.  
- **Transformer Encoder:** Pre-trained BERT/RoBERTa model for contextual feature extraction.  
- **Dense Layers:** Fully connected layers for classification.  
- **Output Layer:** Sigmoid/Softmax activation to generate toxicity probabilities.

### Data Preprocessing
- Removal of links, emojis, and special symbols.  
- Tokenization and padding for uniform input length.  
- Use of pre-trained embeddings for better generalization.

### Training and Evaluation
- **Dataset:** Jigsaw Toxic Comment Classification dataset.  
- **Loss Function:** Binary Cross-Entropy (multi-label setup).  
- **Metrics:** Accuracy, Precision, Recall, F1-score.  
- **Regularization:** Dropout layers to prevent overfitting.

---

## Implementation Details

| Component | Tool / Framework |
|------------|------------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| NLP Models | BERT / RoBERTa |
| Dataset | Jigsaw Toxic Comment Dataset |
| MLOps Tools | MLflow (for experiment tracking and versioning) |

The project integrates **MLflow** to track experiments, manage model versions, and support reproducible research. The implementation emphasizes clarity, modular design, and adaptability for future enhancements.

---

## Repository Structure

```
comment-toxicity-detection/
│
├── docs/                         # Documentation and reports
│   ├── Comment_Toxicity_Model.ipynb
│   └── NueralProject.ipynb
│
├── chatbot.py                    # Script for chatbot integration (optional interface)
├── server.py                     # Backend server for deployment
├── toxicity.h5                   # Trained model file
├── toxicity_model.h5             # Alternate model checkpoint
├── vectorizer.pkl                # Saved TF-IDF/Count vectorizer
├── vectorizer_vocab.pkl          # Vectorizer vocabulary file
├── nueral_report (3).pdf         # Project proposal/report
└── README.md                     # Project documentation
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/tharakadilshandmt/comment-toxicity-detection.git
   cd comment-toxicity-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Jigsaw Toxic Comment Dataset: [Kaggle Link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

4. **Run the model**
   ```bash
   python server.py
   ```

5. **(Optional)** Integrate chatbot interface
   ```bash
   python chatbot.py
   ```

---

## Team Members and Contributions

This project was carried out as a **group research project** by undergraduate students of the **Department of Computer Engineering, University of Peradeniya**, under the supervision of the CO542 course instructors.

| Member | Student ID | Contribution |
|---------|-------------|--------------|
| **D.M.T. Dilshan** | E/20/069 | Model architecture, MLOps integration, documentation |
| **R.V.C. Rathnaweera** | E/20/328 | Data preprocessing, model fine-tuning |
| **K.N.P. Karunarathne** | E/20/189 | Literature review, model evaluation |
| **W.M.N. Dilshan** | E/20/455 | Dataset preparation, report writing |

---

## Credibility

This repository represents an official academic project submitted for assessment under the **CO542: Neural Networks and Fuzzy Systems** module. It reflects independent implementation, experimentation, and evaluation of deep learning concepts in natural language processing.

---

## Future Work
- Integrate explainable AI (XAI) for model transparency.  
- Deploy as a REST API for real-time comment moderation.  
- Extend to multilingual toxicity detection.  
- Explore lightweight Transformer variants for faster inference.

---

## Acknowledgement

We sincerely thank the **Department of Computer Engineering**, **Faculty of Engineering**, University of Peradeniya, for the guidance and facilities provided to complete this project successfully.

---

## Contact
For further inquiries or collaboration:

**Dilshan D.M.T.**  
Email: dmt.dilshan@example.com  
GitHub: [github.com/tharakadilshandmt](https://github.com/tharakadilshandmt)

**Karunarathne K.N.P.**  
Email: e20189@eng.pdn.ac.lk
GitHub: [github.com/KasunikaKarunarathne](https://github.com/KasunikaKarunarathne)
