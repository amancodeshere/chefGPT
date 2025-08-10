# Recipe Recommender System - ChefGPT

Over the past few weeks, I designed and implemented a **Recipe Recommender System** that integrates multiple public datasets to create a personalised cooking recommendation engine. This project combined **data engineering, natural language processing, and machine learning** to deliver relevant recipe suggestions based on user preferences and historical data.

### **Objective**

The aim was to build a system capable of:

* Analysing ingredients, recipe metadata, and user reviews.
* Providing both **content-based** and **collaborative filtering** recommendations.
* Handling large, noisy datasets from different sources through a unified pipeline.

---

### **Data Sources**

I merged and processed **three major datasets**:

1. **Recipe Ingredients and Reviews (Kaggle)** – structured recipe data with ingredients and detailed reviews.
2. **Food.com Recipes and Reviews (Kaggle)** – over 200,000 recipes with user ratings and comments.
3. **Recipe NLG Dataset (HuggingFace)** – rich text-based recipe data designed for NLP tasks.

---

### **Data Engineering & Preprocessing**

* **Cleaning & standardisation**: Removed nulls, reformatted ingredient lists, normalised units of measurement, and standardised naming conventions across datasets.
* **Text parsing**: Converted string representations of lists (ingredients, categories) into Python lists for structured processing.
* **Feature creation**: Generated keyword-based features from categories and ingredients for content-based matching.
* **Dataset merging**: Unified multiple datasets while preserving source-specific features.

---

### **Modelling Approach**

1. **Content-Based Filtering**

   * Used TF-IDF vectorisation on ingredients and category keywords.
   * Measured cosine similarity between recipes to recommend similar dishes.

2. **Collaborative Filtering**

   * Built a **User-Item matrix** mapping ratings from users to recipes.
   * Implemented nearest-neighbour search for similar users and items.

3. **Hybrid Recommendation**

   * Combined both approaches, using collaborative filtering for user-personalised suggestions and content-based filtering for cold-start cases.

---

### **Technical Stack**

* **Python** (Pandas, NumPy, Scikit-learn)
* **NLP**: TF-IDF vectorisation, cosine similarity
* **Data storage**: Google Drive (Colab integration)
* **Evaluation**: Precision\@K, Recall\@K, qualitative inspection of sample recommendations

---

### **Key Results**

* Successfully processed and merged datasets containing **millions of ratings and reviews**.
* Generated meaningful recommendations for both returning users and new users with no prior ratings.
* Implemented an **efficient similarity search** that returned results in milliseconds for medium-scale data.
* Established a **scalable pipeline** that can integrate additional recipe datasets with minimal modification.

---

### **Challenges Solved**

* **Data inconsistency**: Unified formats from disparate sources while preserving important details.
* **Cold start problem**: Integrated hybrid logic to recommend recipes even without prior user interaction.
* **Scalability**: Optimised similarity calculations to handle large datasets efficiently in memory.

---

### **Impact**

This project demonstrates how diverse datasets can be combined into a single machine learning pipeline to deliver practical, user-facing recommendations. The methodology could be adapted for other domains such as **movie recommendations, product suggestions, or e-learning content delivery**.

---
