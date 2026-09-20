# La Liga Player Valuation

![CI](https://github.com/lugavt/tooling_for_data_science/actions/workflows/ci.yml/badge.svg)

A Streamlit app that looks up a La Liga player, predicts his market value with
a pre-trained model, and explains the prediction feature by feature with
SHAP. Build on top of data and a model ported from an earlier
personal project ([Soccer_player_market_value](https://github.com/lugavt/Soccer_player_market_value)).

## What it does

Two tabs:

- **Player search** — pick a club, then a player, and see the model's
  valuation against the value listed on Transfermarkt (2025), with a visual range
  band and a verdict (undervalued / fairly valued / overvalued). Below that,
  the positive and negative attributes pushing the valuation up or down, with
  a glossary explaining the less obvious feature names.
- **Leaderboard** — the most undervalued players, filterable by position.

## Data and model provenance

The committed `data/sample_players.parquet` (499 players, 2025 season) and
`models/xgb_model.pkl` are ported from an earlier personal project, not
generated for this course. Only a sample of real data is committed — enough
to run and test the app — not the full original dataset.

## Running it locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv run streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Running it with Docker

**Pull the published image:**

```bash
docker pull lugavt/tooling-for-data-science:latest
docker run -p 8501:8501 lugavt/tooling-for-data-science:latest
```

**Or build it yourself from the repo:**

```bash
docker build -t tooling-for-data-science .
docker run -p 8501:8501 tooling-for-data-science
```

Either way, the app is then available at `http://localhost:8501`.

## Project structure

```
app.py                              # Streamlit UI -- presentation only
src/tooling_for_data_science/
    loading.py                      # data import + validation (tested)
    features.py                     # undervalue score
    leaderboard.py                  # filtering + ranking (tested)
    model.py                        # load model, predict, explain (tested)
data/sample_players.parquet         # committed data sample
models/xgb_model.pkl                # committed trained model bundle
assets/favicon.png                  # browser-tab icon
tests/                               # pytest suite
Dockerfile, .dockerignore
.github/workflows/ci.yml
```

## Tests

22 tests that cover data loading and validation, undervalue
score computation, leaderboard filtering/ranking, and the model layer.
```bash
uv run pytest --cov=src --cov-report=term
```
