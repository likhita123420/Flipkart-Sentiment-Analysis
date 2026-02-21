# Flipkart Sentiment Analysis

## 📌 Project Overview
This project performs **sentiment analysis** on customer reviews collected from **Flipkart**, one of India’s largest e-commerce platforms.  
The goal is to automatically predict whether a given review is **positive**, **negative**, or **neutral** based on the text content.

Sentiment analysis is a common **Natural Language Processing (NLP)** technique used in customer feedback analysis, social media monitoring, and product reputation management.

---

## 🛠️ Technologies & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK / spaCy
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Dataset
- The dataset contains product reviews scraped from Flipkart.
- Each review is labeled with a sentiment category:
  - **Positive** → Satisfied / happy customer
  - **Negative** → Dissatisfied / unhappy customer
  - **Neutral** → Neither strongly positive nor negative
- Preprocessing includes:
  - Text cleaning
  - Tokenization
  - Stopword removal
  - Vectorization (e.g., TF-IDF)

---

## ⚙️ Project Workflow
1. **Data Collection**
   - Scraping reviews from Flipkart web pages
2. **Data Cleaning & Preprocessing**
   - Case normalization
   - Removing HTML tags, punctuation, special characters
   - Tokenization & stopword removal
3. **Feature Extraction**
   - Bag of Words / TF-IDF Vectorization
4. **Model Building**
   - Train a classification model (e.g., Logistic Regression / Naive Bayes / SVM)
5. **Model Evaluation**
   - Accuracy
   - Confusion Matrix
   - Precision, Recall & F1-score
6. **Visualization**
   - Plotting sentiment distribution and important features

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics provide a complete evaluation of classification performance — especially for imbalanced sentiment data.

---

## 📈 Results & Insights
- Visualized the overall distribution of sentiments
- Best performing model identified and evaluated
- Shows how customer satisfaction varies across products/categories

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/likhita123420/Flipkart-Sentiment-Analysis.git

Open the notebook in Jupyter:

jupyter notebook
Run all cells sequentially

🧪 Sample Outputs
Input: "Excellent product! Worth every penny."
Output: Positive

Input: "Very disappointed, product broke in 2 days."
Output: Negative

# Streamlit app
https://flipkart-sentiment-analysis-akk8dcqzcyqzzeuwyhf6ly.streamlit.app/

# 📌 Conclusion

This project applies machine learning and NLP techniques to analyze customer reviews and understand sentiment trends.
It is well suited for data science portfolios and demonstrates proficiency in:
Text preprocessing
Feature extraction
Model building & evaluation
