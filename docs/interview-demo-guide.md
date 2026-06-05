# Interview Demo Guide — readmission-risk-scorer

5-scenario walkthrough. Total runtime: ~12 minutes.
Start with the API running: `MOCK_MODEL=true uvicorn src.api.main:app --port 8000 --reload`

---

## Scenario 3 (3 min): Model Quality — "How do you know the model works?"

**Setup:** Open the eval dashboard or run `curl http://localhost:8000/eval-report`

**Walk through:**
1. AUROC (target 0.78-0.82): "This measures rank-ordering — does the model score high-risk patients
   higher than low-risk ones? AUROC of 0.80 means it does this correctly 80% of the time."

2. Calibration plot: "Accuracy alone doesn't tell me whether 0.75 means 75% likely.
   The calibration plot checks whether the model's probabilities are trustworthy, not just
   well-ranked. A perfectly calibrated model sits on the diagonal. This matters for a
   discharge nurse who needs to know whether 'HIGH risk' means 65% or 90% likely."

3. SHAP global summary: "This is the sanity check. The model learned to rely on prior
   admissions (strongest predictor), comorbidity index, and SNF discharge disposition.
   These match clinical literature on readmission risk. If the SHAP summary showed
   'weekend_discharge' as the top feature, something would be wrong with the training data
   and this plot would catch it before deployment."

**Key point:** "Accuracy alone isn't enough. Calibration says whether probabilities
are trustworthy. SHAP global summary says whether the model learned the right patterns."

---

## Scenario 1 + 2 (4 min): Prediction Contrast — "Show me the model reasoning"

**Scenario 1 — High-Risk Patient:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "primary_dx_group": "HF",
    "comorbidity_index": 5,
    "length_of_stay_days": 7,
    "age_group": "75-84",
    "prior_admissions_12m": 3,
    "procedure_count": 2,
    "discharge_disposition": "SNF",
    "icu_flag": true,
    "emergency_admit_flag": true,
    "insurance_type": "MEDICARE",
    "specialist_consult_ct": 1,
    "incomplete_dc_flag": true,
    "weekend_discharge": false
  }'
```

Expected: `risk_score: 0.81, risk_tier: "HIGH", requires_review: true`
Top SHAP factors: prior_admissions_12m +0.34, discharge_disposition +0.21, comorbidity_index +0.18

**Walk through SHAP factors:**
"Prior admissions +0.34 is the single strongest driver. This patient has been admitted
3 times in the past year. The +0.34 SHAP value means this feature alone pushed the
readmission probability substantially upward from the base rate. SNF discharge adds another
+0.21 — care transitions to skilled nursing facilities are high-risk periods. The discharge
planner sees exactly why this patient scored HIGH, not just that they did."

**Scenario 2 — Low-Risk Patient:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "primary_dx_group": "KNEE_HIP",
    "comorbidity_index": 1,
    "length_of_stay_days": 3,
    "age_group": "45-64",
    "prior_admissions_12m": 0,
    "procedure_count": 2,
    "discharge_disposition": "HOME",
    "icu_flag": false,
    "emergency_admit_flag": false,
    "insurance_type": "COMMERCIAL",
    "specialist_consult_ct": 1,
    "incomplete_dc_flag": false,
    "weekend_discharge": false
  }'
```

Expected: `risk_score: 0.12, risk_tier: "LOW", requires_review: false`
Top SHAP factors: prior_admissions_12m -0.28, discharge_disposition -0.19, comorbidity_index -0.11

**Key point:** "Same model, completely different reasoning. The SHAP waterfall for each patient
is specific and auditable. A discharge nurse can read these factors and say 'yes, that's why
I'm worried about this patient.' That's the difference between a model a clinician will use
and one they won't."

---

## Scenario 4 (2 min): Graceful Degradation — "What if data is missing?"

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "primary_dx_group": "HF",
    "comorbidity_index": 5,
    "length_of_stay_days": 7,
    "age_group": "75-84",
    "prior_admissions_12m": 3,
    "procedure_count": 2,
    "discharge_disposition": "SNF",
    "icu_flag": true,
    "emergency_admit_flag": true,
    "insurance_type": "MEDICARE",
    "incomplete_dc_flag": true,
    "weekend_discharge": false
  }'
```
Note: `specialist_consult_ct` is omitted.

Expected: `imputed_features: ["specialist_consult_ct"]` in response.

**Walk through:**
"Real EHR data has missing fields. The model doesn't fail — it applies median imputation
and tells the caller exactly which features were imputed via the `imputed_features` field.
No silent failure. The response is still valid and the caller can decide whether to trust
a prediction where specialist consultations were imputed. Transparency over silence."

---

## Design Decisions (3 min): "Why not an LLM for this?"

**Three decisions to hit:**

1. **XGBoost not LLM:**
   "This is a structured tabular classification problem with 13 features. XGBoost consistently
   beats neural networks on tabular data at this sample size. It infers in under 5ms. A language
   model would be slower, more expensive, harder to explain, and no more accurate. The skill is
   knowing when not to use an LLM."

2. **SHAP not just feature importance:**
   "Feature importance tells you which features matter globally across all predictions.
   SHAP gives you the per-patient explanation: this specific prediction, this specific score,
   here's why. A discharge planner can't act on 'prior admissions are generally important.' They
   can act on 'this patient's 3 prior admissions pushed their risk from 0.17 to 0.51.'"

3. **Three tiers not a continuous score:**
   "A probability from 0-1 is not actionable for a discharge nurse. Three tiers map directly to
   protocols: HIGH triggers a social work consult, MODERATE triggers telephonic follow-up, LOW
   triggers standard discharge. The eval report documents the precision/recall tradeoff at each
   threshold so the CMO can justify the cutoff to the care coordination team."

---

## Production Path (2 min): "What would it take to go live?"

**Four changes:**

1. **Real EHR pipeline:** Connect to the hospital's ADT feed or discharge worklist API.
   The DischargeRecord schema is the contract; the data pipeline feeds it.

2. **Model retraining cadence:** Quarterly retraining on new discharge data.
   Monitor SHAP global summary between runs — if the top features shift, investigate
   why before retraining.

3. **HIPAA controls:** The feature hash in each prediction traces the input without
   storing PHI. In production: encrypt the feature hash mapping, restrict review queue
   access to authorized discharge planning staff, audit log all API calls.

4. **Threshold calibration:** Run the threshold sensitivity analysis from the eval report
   on the hospital's own patient population. CMS target conditions (HF, COPD) may need
   different thresholds than the synthetic training data. The thresholds are env vars —
   no code changes required.
