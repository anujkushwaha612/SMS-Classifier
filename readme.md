# SMS Spam Classifier

This project implements an **SMS spam detection system entirely from scratch**, without relying on high-level machine learning libraries such as Scikit-learn.  
It demonstrates a **ground-up approach** to building machine learning systems, covering **text preprocessing, vectorization, and model design**, while achieving **98% classification accuracy**.

The project highlights strong fundamentals in **Bayesian probability, numerical stability, and natural language processing (NLP)**.

---

##  Project Highlights
- **No high-level ML libraries**: Implemented the entire pipeline manually in Python.  
- **Custom Text Vectorizer**: Designed and implemented a Bag-of-Words (BoW) model for feature extraction.  
- **Spam Classifier from Scratch**: Developed a Multinomial Naive Bayes classifier, incorporating:  
  - Bayesian probability for text classification  
  - Laplace smoothing to handle unseen words  
  - Log-probabilities to prevent numerical underflow  
- **High Accuracy**: Achieved **98% accuracy** on SMS spam detection.  

---

## Dataset
- Dataset: **[SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**  
- Contains **5000+ SMS messages**, each labeled as either **ham (legitimate)** or **spam**.  
- Preprocessed to remove punctuation, normalize text, and tokenize into words before training.  

---

## Architecture & Workflow

1. **Text Preprocessing**
   - Lowercasing
   - Removal of punctuation and stopwords
   - Tokenization into words

2. **Vectorization (Bag-of-Words Model)**
   - Built a custom vectorizer to map SMS messages into word-count feature vectors
   - Constructed a vocabulary dictionary from the training set

3. **Classifier (Multinomial Naive Bayes)**
   - Implemented Bayesian probability to calculate likelihood of classes
   - Applied Laplace smoothing to handle zero-frequency words
   - Used log-probabilities to improve numerical stability and prevent underflow

4. **Evaluation**
   - Train-test split of dataset
   - Accuracy metric computed for performance validation
   - Achieved **~98% accuracy**

---

## Project Structure

```
sms-spam-classifier/
│── data/ # Kaggle SMS Spam Collection dataset
│── src/
│ ├── preprocess.py # Text preprocessing functions
│ ├── vectorizer.py # Bag-of-Words vectorizer implementation
│ ├── naive_bayes.py # Multinomial Naive Bayes classifier
│ └── train.py # Training and evaluation pipeline
│── results/
│ └── metrics.txt # Accuracy and evaluation metrics
│── README.md # Project documentation
```

---

## Results
- **Accuracy**: 98%  
- **Model Strengths**:
  - Lightweight, interpretable classifier
  - Strong performance on small to medium-sized text datasets
- **Limitations**:
  - Bag-of-Words ignores word order and context
---

## Tech Stack
- **Language**: Python (No external ML libraries used for modeling)  
- **Libraries Used**:  
  - `numpy` (matrix operations)  
  - `pandas` (dataset handling)  
  - `re` (regex for text preprocessing)  
  - `nltk` (for stopwords corpus)  

---

## Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/anujkushwaha612/SMS-Classifier.git
   ```

2. **Download Dataset from Kaggle**


3. **Install the required libraries**


4. **Run the complete notebook cells**

``` bash
# Expected output

Accuracy on test set: 98.0%
```
