# Code Audit Report: DataEngineering_MLPipeline

## Executive Summary
This audit reviews the `DataEngineering_MLPipeline` repository to assess its readiness as a professional AI/ML portfolio piece. While the repository successfully demonstrates an end-to-end ETL and ML pipeline with a Flask web application, there are significant architecture and code quality issues. The pipeline relies on interactive inputs instead of command-line arguments, executes package installations directly from executing scripts, and the Flask app runs in debug mode. Resolving these issues, adding dependency management, and refactoring scripts to follow best practices will align the repository with production-grade engineering standards.

## 1. Critical Issues (Blockers)

- **Unsafe Package Management in Scripts (`train_classifier.py`: L5-29, `run.py`: L4-38)**
  - **Issue**: The scripts use `os.system` and `subprocess.check_call` to automatically install packages via pip during script execution. This is highly discouraged as it circumvents virtual environments, ignores pinned dependencies, and causes unexpected mutations to the host environment.
  - **Fix**: Remove all in-code package installation routines. Create a `requirements.txt` or `Pipfile` holding the explicit dependencies for the project. Provide installation instructions via a `README.md`.

- **Interactive Prompts Break ML Pipeline Automation (`train_classifier.py`: L138, `run.py`: L79)**
  - **Issue**: `train_classifier.py` and `run.py` rely on the built-in `input()` function to ask for file paths interactively. This completely breaks CI/CD and automated retraining workflows.
  - **Fix**: Use the standard library `argparse` module to accept `database_filepath` and `model_filepath` as command-line arguments, similar to the approach currently used in `process_data.py`.

- **Flask App Security Risk (`run.py`: L176)**
  - **Issue**: The Flask app starts with `debug=True`. Exposing the Werkzeug debugger in a generic or production-like environment is a massive security vulnerability allowing arbitrary code execution.
  - **Fix**: Set `debug=False` for production deployments or use environment variables (e.g., `FLASK_ENV=development`) to conditionally enable it.

## 2. High Issues

- **Missing Dependency Manifest**
  - **Issue**: There is no `requirements.txt`, `Pipfile`, or `environment.yml`. An end-user cloning this repository will not know the exact environment configuration. 
  - **Fix**: Generate and commit a `requirements.txt` file (e.g., locking versions for `flask`, `scikit-learn`, `pandas`, `nltk`, `sqlalchemy`, and `joblib`).

- **Bloated Repository Artifacts (`models/classfier.pkl`)**
  - **Issue**: The `models/classfier.pkl` file is over 500MB, consuming massive repository bandwidth and expanding clone times. 
  - **Fix**: Add `*.pkl` to `.gitignore`. Use tools such as Git LFS if persisting the model is strictly necessary, or provide instructions/scripts to download a pretrained model from object storage (like AWS S3) instead of tracking it in git.

## 3. Medium Issues

- **Missing Modularity and DRY Principles (`train_classifier.py`, `run.py`)**
  - **Issue**: `tokenize` function is duplicated across both `train_classifier.py` and `run.py`.
  - **Fix**: Move `tokenize` to a separate `utils.py` module and import it into both scripts to enforce DRY (Don't Repeat Yourself) principles.

- **Use of print() instead of Logging (`process_data.py`, `train_classifier.py`)**
  - **Issue**: Progress tracking relies entirely on print statements, which don't support verbosity levels and are difficult to pipe properly.
  - **Fix**: Integrate the standard `logging` library to track pipeline status, emitting INFO, WARN, and ERROR messages optimally.

## 4. Low Issues / Nice-to-haves

- **Hardcoded Flask Port (`run.py`: L176)**
  - **Issue**: The web app runs on port `3001` statically.
  - **Fix**: Allow the port to be configurable via standard environment variables (e.g., `PORT`), defaulting to `3001` or `5000`.

- **Pickle Security**
  - **Issue**: The model loading mechanism relies on `pickle.load` (via joblib), which is unsafe if loading from untrusted sources.
  - **Fix**: While mostly acceptable for portfolio proofs-of-concept, a note acknowledging model serialization security risks (or migrating to safer formats like ONNX) demonstrates mature engineering awareness.

## 5. Dependency Health
- **Missing Check**: Completely missing `requirements.txt`.
- **NLTK Download Hooks**: The `train_classifier.py` dynamically downloads `omw-1.4`, `punkt_tab`, and `wordnet` at runtime. These should ideally be documented in the README as a pre-flight execution step.

## 6. Documentation Completeness
- Current Readmes (`README_processData.md`, `README_run.md`, `README_trainClassifier.md`, `README.md`) exist but are fragmented. 
- **Recommendation**: Consolidate setup, testing, and pipeline execution into a single, cohesive `README.md`. It must list the ML classifier choice (RandomForest) and justify it.

## 7. Recommended Next Steps
1. Delete dynamic `pip install` lines from all `.py` files.
2. Create standard `requirements.txt` with locked versions.
3. Replace `input()` prompts in `train_classifier.py` and `run.py` with `argparse`.
4. Refactor `tokenize` into a shared module (`utils.py`).
5. Set `app.run(debug=False)` in `run.py`.
6. Consolidate fragmented README files into one robust portfolio README.
