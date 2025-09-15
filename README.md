# 💌 通用垃圾訊息分類器 | Universal Spam Classifier (SMS & Email)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvchung/Universal_Spam_Classifier/blob/main/Universal_Spam_Classifier.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

受夠了每天收不完的垃圾簡訊和郵件嗎？😫 
這個專案將帶您打造一個強大的**通用垃圾訊息分類器**，它不僅能處理簡訊 📱，也能處理電子郵件 📧！

這個 Colab 筆記本展示了從資料探索、文字雲視覺化，到模型訓練與建立一個 rеаl-time 預測系統的完整流程。 ✨

---

## ✨ 專案亮點 (Features)

*   **📱/📧 通用設計**：一個模型就能同時應對簡訊和郵件的分類挑戰。
*   **📊 深入的資料探索 (EDA)**：透過漂亮的圓餅圖和文字雲 ☁️，深入了解「垃圾訊息」和「正常訊息」的用詞差異。
*   **📝 清晰的步驟**：從資料清理、特徵提取到模型訓練，每一步都有詳細的說明和程式碼。
*   **🚀 實用的預測系統**：專案的最後建立了一個簡單的函式，讓你可以輸入任何訊息，立即得到分類結果！

---

## 🚀 專案流程 (Workflow)

這個筆記本主要分為三個部分，帶你一步步成為垃圾訊息分類大師：

### **Part 1: 簡訊垃圾訊息分類 (SMS Spam Classification) 💬**
1.  **載入與清理資料**：使用 Kaggle 公開的 `spam.csv` 資料集。
2.  **探索性資料分析 (EDA)**：分析垃圾與非垃圾訊息的比例，並用超酷的**文字雲**視覺化最常出現的詞彙。
3.  **模型建立**：使用 `CountVectorizer` 和 `Multinomial Naive Bayes` 演算法建立一個高準確率的分類器。

### **Part 2: 郵件垃圾訊息分類 (Email Spam Classification) 📂**
1.  **處理不同資料格式**：示範如何讀取由大量獨立文字檔組成的 Ling-Spam 資料集。
2.  **模型訓練**：使用 `TfidfVectorizer` 處理郵件內容，並訓練一個 Naive Bayes 分類器。

### **Part 3: 終極預測系統 (Unified Predictive System) 🎯**
1.  **整合模型**：使用在 Part 1 訓練好的模型和向量化工具。
2.  **建立預測函式**：打造一個 `predict_message()` 函式，無論你輸入的是簡訊還是郵件，它都能馬上告訴你是不是垃圾訊息！

---

## 🎨 視覺化成果 (Visualizations)

一個好的分析絕對少不了視覺化！

### 文字雲 (Word Clouds) ☁️

看看垃圾訊息和正常訊息最常用的詞彙長什麼樣子！

**垃圾訊息文字雲 (Spam Word Cloud):**
![Spam Word Cloud](https://github.com/vvchung/Universal_Spam_Classifier/blob/main/spam_wordcloud.png)

**正常訊息文字雲 (Ham Word Cloud):**
![Ham Word Cloud](https://github.com/vvchung/Universal_Spam_Classifier/blob/main/ham_wordcloud.png)

### 混淆矩陣 (Confusion Matrix) 📊

我們的模型表現如何？混淆矩陣一目了然！
![Confusion Matrix](https://github.com/vvchung/Universal_Spam_Classifier/blob/main/confusion_matrix.png)

---

## 🛠️ 使用技術 (Tech Stack)

*   **資料處理**: Pandas, NumPy
*   **機器學習**: Scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`)
*   **自然語言處理**: NLTK
*   **資料視覺化**: Matplotlib, Seaborn, WordCloud

---

## 🏃‍♀️ 如何開始 (Getting Started)

想親手試試看嗎？超級簡單！

1.  **點擊上方 "Open in Colab" 徽章** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvchung/Universal_Spam_Classifier/blob/main/Universal_Spam_Classifier.ipynb)
2.  在 Google Colab 中，點擊**「執行階段」(Runtime)** > **「全部執行」(Run all)**。
3.  坐下來，好好享受這場機器學習的魔法秀吧！🍿

---

## ✨ 最終預測系統實戰 (Predictive System in Action)

專案的最後，你可以像這樣使用我們的預測函式：

```python
# 範例 1: 垃圾簡訊
sms_spam = 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)'
prediction = predict_message(sms_spam)
print(f"Prediction: {prediction}")
# 預期輸出: Spam

# 範例 2: 正常郵件
ham_message = 'Hi Sarah, just wanted to confirm our meeting for tomorrow at 10 AM. Let me know if that still works for you.'
prediction = predict_message(ham_message)
print(f"Prediction: {prediction}")
# 預期輸出: Not Spam (Ham)
```

---

## 📄 授權 (License)

本專案採用 [MIT License](https://opensource.org/licenses/MIT) 授權。
