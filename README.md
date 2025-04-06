# Concrete Strength Predictor using Artificial Neural Networks (ANN)

This project presents a neural network-based approach to predict the 28-day compressive strength of concrete using mix proportion parameters. The model replaces traditional, destructive lab testing with an efficient, non-destructive, and reusable prediction tool.

---

## 📌 Overview

Concrete compressive strength is typically determined using cube specimens and Universal Testing Machines (UTMs). This method, although reliable, is time-consuming, costly, and environmentally wasteful.

This project introduces an Artificial Neural Network (ANN) trained to accurately predict compressive strength based on 7 key ingredients used in concrete mix design. The model is trained and validated using actual lab data, offering a practical machine learning alternative for engineers and researchers.

---

## 🎯 Objectives

- Develop a neural network for predicting compressive strength of concrete.
- Train the ANN using real-world concrete mix datasets.
- Evaluate model performance using RMSE and Coefficient of Efficiency (C.E.).
- Create a reusable and adaptable Python-based prediction program.
- Visualize training accuracy and model performance using graphs.

---

## 🧪 Input and Output

**Input Parameters (7 features):**
1. Water  
2. Cement  
3. Fine Aggregate (Sand)  
4. Coarse Aggregate  
5. Blast Furnace Slag  
6. Fly Ash  
7. Superplasticizer  

**Output:**
- Predicted 28-day compressive strength of concrete (in MPa)

---

## 🧠 ANN Architecture and Concept

| Component         | Details                          |
|------------------|----------------------------------|
| Input Layer       | 7 nodes (one for each feature)   |
| Hidden Layer      | 70 nodes (customizable)          |
| Output Layer      | 1 node (compressive strength)    |
| Activation        | Sigmoid Function                 |
| Training Method   | Backpropagation                  |
| Optimization      | Gradient Descent                 |
| Weight Init.      | Xavier Initialization            |
| Epochs            | 1000                             |
| Learning Rate     | 0.01                             |

- **Error Metric**: Root Mean Square Error (RMSE)
- **Model Efficiency**: Coefficient of Efficiency (C.E.)

---

## 📂 File Structure

```
.
├── Documentantation.pdf                     # Detailed technical report
├── Figure_1 Learning Process Graph.png      # RMSE vs Epochs
├── Figure_2 Equity Line Graph.png           # Predicted vs Actual (scatter plot)
├── Figure_3 Prediction and Actual Values... # Prediction trend (line plot)
├── LICENSE                                  # MIT License
├── project_code.py                          # Python source code
├── README.md                                # Full project documentation
├── training_data.csv                        # Training dataset (85%)
└── testing_Data.csv                         # Testing dataset (15%)
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Libraries:**
  - `numpy` – matrix operations
  - `scipy` – sigmoid activation
  - `matplotlib` – plotting
  - `math` – math functions

---

## 🚀 How to Run the Model

1. **Install dependencies**
   ```bash
   pip install numpy scipy matplotlib
   ```

2. **Run the script**
   ```bash
   python project_code.py
   ```

3. **Outputs Generated**
   - RMS loss curve saved as `Figure_1 Learning Process Graph.png`
   - Predicted vs actual (scatter) saved as `Figure_2 Equity Line Graph.png`
   - Line plot of actual vs predicted saved as `Figure_3 Prediction and Actual Values.png`
   - Model prints performance metrics in the terminal

---

## 📈 Model Performance

### ✅ Training Analysis
- RMS error decreases steadily across epochs.
- Indicates convergence without overfitting.

### ✅ Testing Analysis
- **Relative RMSE** is low, showing good predictive accuracy.
- **C.E. (Coefficient of Efficiency):** ~0.95 (95% accuracy)
- Graphs confirm high agreement between predicted and actual values.

### 📊 Visualizations


### 📉 Training Loss Curve
![Training Loss Curve](Figure_1%20Learning%20Process%20Graph.png)

### 📊 Predicted vs Actual (Scatter Plot)
![Predicted vs Actual](Figure_2%20Equity%20Line%20Graph.png)

### 📈 Prediction Trend
![Prediction and Actual Values](Figure_3%20Prediction%20and%20Actual%20Values.png)

---

## 📖 References

1. Lin, C.-J., & Wu, N.-J. (2021). *An ANN Model for Predicting the Compressive Strength of Concrete*. Applied Sciences, 11(9), 3798. https://doi.org/10.3390/app11093798  
2. Rashid, T. (2016). *Make Your Own Neural Network: A Gentle Journey Through the Mathematics of Neural Networks*.  
3. Hao, C.-Y., Shen, C.-H., Jan, J.-C., & Hung, S.-K. (2018). *A Computer-Aided Approach to Pozzolanic Concrete Mix Design*. Advances in Civil Engineering, 2018. https://doi.org/10.1155/2018/4398017  
4. Silva, V.P., Carvalho, R.d.A., Rêgo, J.H.d.S., & Evangelista, F. Jr. (2023). *Machine Learning-Based Prediction of the Compressive Strength of Brazilian Concretes: A Dual-Dataset Study*. Materials, 16(14), 4977. https://doi.org/10.3390/ma16144977  
5. Medium Blog: [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

---

## 📬 Contact

For collaborations, academic inquiries, or feedback, feel free to reach out via GitHub or email.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
