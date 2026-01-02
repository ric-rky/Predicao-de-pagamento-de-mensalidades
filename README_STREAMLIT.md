# Streamlit App — Next Best Action (Collections)

This Streamlit app is a decision-support prototype that recommends the **best collection action** based on the predicted probability of payment, with **ROI simulation** and optional **SHAP explainability**.

## How to run

From the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## What you can demo in a video (suggested flow)

1. Adjust **Days past due** and watch the **action ranking** update.
2. Compare **Best by probability** vs **Best by ROI** (cost-aware decision).
3. Open **Policy over time** to see the best action for `days_past_due = 0..10`.
4. If you want a more “senior” touch, enable **Explainability (SHAP)** and show why the model prefers an action.

## Notes

- Costs are editable and used only for the ROI simulation.
- If SHAP is slow or unavailable in your environment, keep it turned off for the demo.
