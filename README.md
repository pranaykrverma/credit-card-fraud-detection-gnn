# 💳 Credit Card Fraud Detection using Graph Neural Networks (GNN)

Fraudulent transactions are a huge problem in the financial world. 
This project applies **Graph Neural Networks (GNNs)** using 
**PyTorch Geometric** to detect credit card fraud by modeling 
transactions as a graph. 

## 📌 Problem
Traditional ML models (Logistic Regression, Random Forest) 
look at each transaction individually. They fail when fraud patterns 
are subtle or coordinated across accounts/devices.  
👉 Our approach builds a **transaction graph** where:
- Nodes = transactions
- Edges = similarity between transactions (KNN)

## ⚙️ Tech Stack
- Python, PyTorch, PyTorch Geometric
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Google Colab

## 🚀 Approach
1. **Dataset Preparation**  
   - Used Kaggle [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
   - Standardized features using `StandardScaler`.

2. **Graph Construction**  
   - Built KNN graph (k=5) from transaction features.  
   - Nodes = transactions, Edges = similarity.

3. **Model**  
   - Implemented 2-layer Graph Convolutional Network (GCN).  
   - Loss: Cross Entropy  
   - Optimizer: Adam (lr=0.01)

4. **Evaluation**  
   - Accuracy: **99.8%**  
   - Precision: **77.6%**  
   - Recall: **63.6%**  
   - ROC AUC: **99.0%**  

## 📊 Results
- The GNN outperformed traditional ML baselines.  
- ROC curve shows strong fraud vs non-fraud separation.  
- Even though fraud cases are rare (imbalanced dataset), 
  GNN captured hidden patterns.

![ROC Curve](images/roc.png)   <!-- (add a plot screenshot here) -->

## 🔮 Future Scope
- Temporal Graphs (add time dimension)
- Advanced GNNs (GraphSAGE, GAT)
- Semi-supervised fraud detection
- Real-time prediction pipeline

## 📂 Repository Structure
