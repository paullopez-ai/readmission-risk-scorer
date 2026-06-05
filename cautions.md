# Build Cautions — readmission-risk-scorer

Dependency and environment issues encountered during the Phase 1-3 build.
Reference these when writing the README setup instructions.

---

## 1. uv defaults to the system Python (3.14) — breaks numba/llvmlite

**Symptom:** `uv run python ...` or `uv sync` fails with:
```
RuntimeError: Cannot install on Python version 3.14.2;
only versions >=3.6,<3.10 are supported.
hint: llvmlite was included because shap depends on numba which depends on llvmlite
```

**Cause:** uv picks the newest Python on the machine (3.14). `numba` and `llvmlite`
(transitive deps of `shap`) do not support Python 3.14 yet.

**Fix:** Pin the project to Python 3.11 in two places:
- `.python-version` file containing `3.11`
- `pyproject.toml`: `requires-python = ">=3.11,<3.14"`

Then recreate the venv: `uv venv --python 3.11`

---

## 2. llvmlite has no source-buildable wheel without LLVM installed

**Symptom:** `uv sync` or `pip install shap` fails during llvmlite build:
```
subprocess.CalledProcessError: Command [.../ffi/build.py] returned non-zero exit status 1
hint: Build failures usually indicate a problem with the package or the build environment
```

**Cause:** `uv` attempts to build `llvmlite` from source. This requires LLVM to be
installed (`brew install llvm`), which it isn't by default.
A pre-built binary wheel for `llvmlite-0.45.1` (cp311, macosx_10_15_x86_64) does exist
but uv resolves to a newer version (0.47.0) that has no wheel for this platform.

**Fix (two parts):**
1. Install dependencies via `pip` with binary preference:
   ```bash
   pip install --prefer-binary -e ".[dev]"
   ```
2. Prevent uv from trying to compile llvmlite in future `uv sync` runs by adding to `uv.toml`:
   ```toml
   [install]
   no-build-package = ["llvmlite"]
   ```

---

## 3. XGBoost requires OpenMP (libomp) on macOS

**Symptom:** `import xgboost` or `python scripts/train.py` fails with:
```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Library not loaded: @rpath/libomp.dylib
Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

**Cause:** XGBoost's macOS wheel links against `libomp.dylib` which is not
shipped with macOS and must be installed separately.

**Fix:**
```bash
brew install libomp
```
No venv restart needed; retry the command immediately after.

---

## 4. Stale venv activation after uv recreates the venv

**Symptom:** `python scripts/generate-dataset.py` partially runs (numpy/pandas work)
then fails with `ModuleNotFoundError: No module named 'sklearn'` despite `(.venv)`
showing in the prompt.

**Cause:** uv deleted and recreated `.venv` during a failed sync attempt while the
old venv was already activated in the shell. The shell's `PATH` still points to
`.venv/bin` but some module state is stale.

**Fix:** Deactivate and reactivate the venv:
```bash
deactivate
source .venv/bin/activate
```

**Prevention:** Always run `deactivate && source .venv/bin/activate` after any
`uv venv` or `rm -rf .venv` operation.

---

## 5. SHAP 0.49 incompatible with XGBoost 3.x (base_score format)

**Symptom:** `shap.TreeExplainer(model)` raises:
```
ValueError: could not convert string to float: '[1.8375E-1]'
```
in `shap/explainers/_tree.py` at `_set_xgboost_model_attributes`.

**Cause:** XGBoost 3.x changed the internal model config format — `base_score`
is now stored as `[value]` (bracketed) instead of a plain float string.
SHAP 0.49 cannot parse this format.

**Fix:** Bypass `shap.TreeExplainer` entirely and use XGBoost's built-in native
SHAP computation, which produces identical exact Shapley values:

```python
import xgboost as xgb

booster = model.get_booster()
dm = xgb.DMatrix(X, feature_names=feature_names)
# Last column is the baseline term — drop it
shap_values = booster.predict(dm, pred_contribs=True)[:, :-1]
```

`shap.summary_plot(shap_values, X, feature_names=feature_names)` still works
since it only needs a numpy array, not a TreeExplainer object.

Applied in: `scripts/evaluate.py` (`plot_shap_summary`, `run_latency_benchmark`)
and `src/utils/shap_explainer.py`.
