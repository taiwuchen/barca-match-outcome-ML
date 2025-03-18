# ENSF 444 Project

## Predicting FC Barcelona Match Outcomes using Machine Learning

### Introduction

FC Barcelona is a club that heavily invests in data analytics to gain a competitive edge. The **Barça Innovation Hub**, for example, leverages sports data to enhance team performance and strategy. In this context, a viable machine learning problem is to analyze and predict the outcome of FC Barcelona's matches (**win, draw, or loss**) based on pre-match or in-match performance metrics. 

This is a **classification problem** (predicting categorical outcomes) that falls under **football performance analysis**. By using historical data and **key performance indicators (KPIs)** from past games, the club can identify patterns that lead to wins or losses, aiding coaching decisions and tactical planning. 

Such analysis aligns with Barcelona’s needs, as maintaining high performance and winning matches is a core challenge in both **domestic and international** competitions.

---

## Problem Description: Match Outcome Classification

### Problem Statement
Predict **FC Barcelona’s match result** (win, draw, loss) from performance metrics. Using match-specific features (**e.g., possession percentage, shots on goal, pass accuracy, etc.**), we can train a **classifier** to predict the outcome of a game. 

For example, before or during a match, the model could estimate the likelihood of a **win** based on these indicators. 

This classification task directly supports performance analysis:
- It quantifies how much various factors contribute to winning or not.
- Barcelona often dominates possession but doesn’t always win, showing the need to consider **multiple KPIs** simultaneously.
- **Machine learning** can capture complex patterns that human intuition might miss.

In sum, this problem addresses a key sporting challenge: **understanding and improving the drivers of match outcomes**.

---

## Dataset Selection

### Dataset: FC Barcelona Match Performance Dataset (Kaggle)
To tackle this problem, we will use the **FC Barcelona Match Performance Dataset** from Kaggle, which contains:
- **200 match observations** of FC Barcelona.
- **Key performance indicators (KPIs)** such as possession, shots, passes, etc.
- Each match’s **result** (win, draw, loss) with **detailed statistics** for Barcelona and their opponents.

### Why this Dataset?
- It is **tailor-made** for our problem.
- Saves time on **data gathering and cleaning**.
- **Recent, curated, and well-documented**, ensuring **data quality**.
- Aligns perfectly with **Barcelona’s style of play** and competition history.
- Ensures **relevance** without needing to filter out other teams.

Link to dataset: [https://www.kaggle.com/datasets/adnanshaikh10/fc-barcelona-statistics](https://www.kaggle.com/datasets/adnanshaikh10/fc-barcelona-statistics)

---

## Machine Learning Models for Comparison

In a **structured scikit-learn workflow**, we will experiment with **three different models** to compare their performance. These models include **both linear and non-linear algorithms** to capture complex relationships.

### Proposed Models:

1. **Logistic Regression** (Baseline Model)
   - A **linear classification model** predicting probabilities of outcomes.
   - **Fast & interpretable** with clear feature impact analysis.
   - Limitation: May not capture **interactions** between features.

2. **Random Forest Classifier** (Non-linear Model)
   - **Ensemble model** using multiple decision trees.
   - Handles **feature interactions automatically**.
   - Provides **feature importance scores** to highlight key KPIs.

3. **Gradient Boosting Classifier** (Non-linear Model)
   - Finds an **ensemble learning technique** that builds multiple decision trees sequentially.
   - Improves upon weak models iteratively to enhance predictive performance.
   - Well-suited for structured data and **highly competitive in predictive performance**

### Model Evaluation:
- **Data Preprocessing** (handling missing values, normalizing features).
- **Splitting** into **training & test sets**.
- **Metrics for evaluation**:
  - **Accuracy**
  - **F1-score**
  - **Confusion Matrix** (for imbalanced class distributions).
- **Cross-validation** for robust model tuning.

This structured approach ensures a **fair comparison** under the same conditions.

---

## Justification and Alignment with Club Needs

### Why this problem?
- **Predicting match outcomes** supports **FC Barcelona’s competitive goals**.
- Understanding **data behind wins & losses** allows **tactical improvements**.
- Can help coaches **focus on critical KPIs**:
  - If the model finds that **shots on target** strongly predict wins, training can emphasize **shot quality**.
  - If too many **turnovers in midfield** precede losses, strategies can be **adjusted**.

### Why this dataset?
- **Aligned with Barcelona’s needs** (contains club-specific performance metrics).
- **Granular insights** into Barcelona’s playing style.
- **Public & curated** dataset allows quick prototyping.

### Why compare multiple models?
- Football outcomes involve **complex interdependencies**.
- Linear models may **oversimplify** (e.g., “more possession is always better”).
- **Non-linear models** can uncover interactions (e.g., **high possession + high shot conversion** leads to wins).

By comparing models, we **find the best approach** for capturing **football performance complexities**.

---

## Summary

This project proposes a **machine learning solution** for **predicting FC Barcelona’s match outcomes** using **performance indicators**. By leveraging **historical match data**, we can build **classification models** that support **coaching decisions & tactical planning**.

### Key Takeaways:
- **Football performance analysis** is data-driven.
- **Machine learning** helps quantify **win/loss factors**.
- **Multiple models** ensure **robust predictions**.
- **Barcelona-specific data** increases **relevance**.

With a **structured scikit-learn implementation**, FC Barcelona can turn raw match data into **actionable insights**, helping maintain **peak performance** and **winning consistency**.

---

## Sources:
- **FC Barcelona Innovation Hub** – emphasis on sports analytics for performance.
- **Kaggle** – FC Barcelona Match Performance Dataset.
- **Soccermatics (David Sumpter)** – high possession & wins observation.
- **American Soccer Analysis** – Barcelona’s possession vs. results.

