# Supervised Modeling for Tuition Payment Prediction

This project aims to build a predictive system capable of **estimating the probability of tuition payment** after a specific collection action is applied.  
The dataset contains historical tuition records and collection actions performed by a company responsible for managing payments for private higher education institutions.

Beyond pure prediction, the project evolves into an **AI-assisted decision-support system**, recommending the most effective and cost-efficient collection actions.

---

## 1. Project Objective

The main goal is to support operational and strategic decision-making by enabling the company to:

- Estimate the probability of payment after a collection action;
- Identify students with higher risk of non-payment;
- Prioritize cases requiring stronger intervention;
- Avoid unnecessary collection actions for students likely to pay;
- Optimize collection resources under cost constraints.

---

## 2. Linking Collection Actions to Tuition Installments

Since the collection dataset did not contain a direct identifier linking actions to tuition installments, a **temporal matching rule** was developed.

For each collection action, the tuition installment belonging to the same student with the closest due date was selected, within a **±10-day window**.  
When multiple candidates existed, the installment with the smallest absolute time difference was chosen.

This process produced a new dataset where **each row represents a collection action applied to a specific tuition installment**, including:

- `acao_cobranca` — type of collection action;
- `dias_dif` — difference (in days) between the action date and the installment due date;
- `foi_pago` — target variable indicating whether the installment was paid (1 = paid, 0 = not paid).

---

## 3. Modeling Results

Several classification models were evaluated, including:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- Gradient Boosting  
- LightGBM  
- Naive Bayes  
- KNN  

Models were compared using **AUC**, **recall**, **F1-score**, and **balanced accuracy**, depending on different business objectives.

| Objective                               | Model                        | Highlights                               |
|----------------------------------------|------------------------------|------------------------------------------|
| Predict payment (`foi_pago = 1`)        | Logistic Regression / Naive Bayes | Recall ≈ 0.83, F1-score ≈ 0.70 |
| Probability estimation                 | XGBoost / LightGBM           | AUC ≈ 0.65, good overall balance          |
| Identify non-payment (`foi_pago = 0`)  | Random Forest                | Recall ≈ 0.68, F1-score ≈ 0.60             |

---

## 4. AI-Assisted Decision Strategy

Instead of stopping at prediction, the project was extended into a **decision-support framework**, including:

### 🔹 Next Best Action (NBA)
For a given scenario (e.g., number of days past due), the system estimates:

> **P(payment | collection action)**

All available actions are ranked according to their predicted effectiveness.

---

### 🔹 Cost-Aware Expected Return
Collection actions have different operational costs.  
The system optionally computes **expected return**:

$\text{Expected Return} = P(\text{payment}) * \text{installment value} − {action cost}$


Monetary values are treated as **business parameters**, not as dataset features, reflecting real-world operational settings.

---

### 🔹 Policy Simulation Over Time
For different levels of delinquency (`dias_dif`), the system simulates an **optimal collection policy**, selecting the action that maximizes expected return at each stage.

This allows decision-makers to define **data-driven collection strategies**, rather than isolated, ad-hoc actions.

---

### 🔹 Model Explainability
To ensure transparency and trust, SHAP-based explainability is used to show:

- Which factors increase or decrease the probability of payment;
- Why a specific action is recommended in a given scenario.

---

## 5. Interactive Dashboard (Streamlit App)

An interactive **Streamlit dashboard** was developed to demonstrate the system as a real decision-support tool.

The app allows users to:
- Select the number of days past due;
- View the ranking of collection actions;
- Compare actions by probability and expected return;
- Visualize the optimal policy over time;
- Inspect model explanations.

This transforms the project from a notebook-based analysis into a **product-oriented prototype**.

---

## 6. Modeling Workflow

- Exploratory data analysis;
- Temporal matching between actions and installments;
- Feature engineering based on action type and timing;
- Training and comparison of multiple classifiers;
- Probability calibration for decision-making;
- Integration of AI-assisted ranking and policy simulation.

---

## 7. Tools and Technologies

- Python 3.10  
- Jupyter Notebook (VS Code)  
- Streamlit (interactive dashboard)  
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `lightgbm`
  - `shap`

---

## 8. Possible Improvements

- Cross-validation for increased robustness;
- Advanced imbalance handling (SMOTE, ADASYN);
- Decision threshold optimization;
- Uplift modeling for action-effect estimation;
- Contextual bandit approaches for sequential decision-making;
- Integration with real-time systems.

---

## Author

**Ricardo Luís Bertolucci Filho**

- [LinkedIn](https://www.linkedin.com/in/ricardo-lu%C3%ADs-bertolucci-filho/)
- [GitHub](https://github.com/ric-rky)
- Email: bertolucci.rl@gmail.com

---

## Main Notebook

- [`predicao_mensalidades.ipynb`](https://github.com/ric-rky/Predicao-de-pagamento-de-mensalidades/blob/main/predicao_mensalidades.ipynb)
