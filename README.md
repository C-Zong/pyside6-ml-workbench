# PySide6 ML Workbench

A desktop data-preparation and machine-learning workbench built with PySide6, pandas, and scikit-learn.

The app is designed for spreadsheet-style workflows: load raw tables, clean columns and rows, aggregate data to a modeling grain, merge prepared datasets, train models, and generate prediction datasets.

## Features

- Load `.csv`, `.xlsx`, `.xls`, and `.xlsm` files
- Clean data with focused UI tools
  - delete columns
  - replace `null / na / empty` with `0`
  - delete specific rows
  - drop rows with high bad-value ratios
  - split and combine columns
  - create time-range datasets
- Aggregate one dataset into a new grouped dataset
  - `sum`, `count`, `mean`, `max`, `min`, `nunique`
  - optional binary conversion after aggregation
- Merge prepared datasets
  - `inner`, `left`, `right`, `outer`, `append`
  - optional right-side aggregation before merge
  - merge-risk preview
  - optional coalescing of merged columns
- Train classification or regression models
  - auto or manual task-type selection
  - configurable model parameters
  - prediction output as a new dataset
- Manage multiple trained or loaded models in the left panel
  - select active model
  - export model
  - delete model

## Supported Models

Classification:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression`
- `SVC`
- `SGDClassifier`

Regression:

- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `SVR`
- `SGDRegressor`

Incremental training support:

- `SGDClassifier`
- `SGDRegressor`

## Project Structure

```text
app/
  controller.py        # workflow/state coordination
  data_ops.py          # data loading, cleaning, merge, aggregate helpers
  ml_ops.py            # training and prediction logic
  models.py            # model registry
  plotting.py          # matplotlib figures
  state.py             # shared app state
  worker.py            # background workers
  views/               # PySide6 UI widgets
main.py                # app entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Workflow

1. Load one or more datasets from the left panel.
2. Use `Clean` to prepare columns and rows.
3. Use `Aggregate` if raw data must be grouped before modeling or merging.
4. Use `Merge` to combine prepared datasets.
5. Use `Train` to choose:
   - target
   - features
   - task type
   - model
6. Train a model, inspect metrics and feature importance, then predict on the current dataset if needed.

## Notes

- Dataset export defaults to `CSV` to avoid Excel cell-length limits on very long text.
- Loaded model files can be reused for prediction only when feature metadata is available.
- Feature importance for categorical variables is based on encoded features, so one raw column may appear as multiple expanded importance entries.
- Classification accuracy is shown when the active trained model is a classification model.

## Dependencies

- PySide6
- pandas
- numpy
- openpyxl
- scikit-learn
- matplotlib
- joblib

## Development Note

This project was iteratively developed and refined with AI-assisted code generation and editing. Final feature decisions, workflow design, debugging, and validation were directed by the author.
