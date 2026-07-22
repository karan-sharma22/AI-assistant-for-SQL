# GitHub Presentation Guide

Use this file as a quick checklist for polishing the repository on GitHub.

## Suggested Repository Description

AI-powered Streamlit analytics assistant that converts natural language into SQL, queries SQLite or uploaded CSV data, and generates tables, charts, and insights with Gemini and LangChain.

## Short Tagline

Talk to your data with Gemini, LangChain, and SQL.

## Suggested GitHub Topics

- ai
- analytics
- streamlit
- langchain
- gemini
- google-gemini
- text-to-sql
- sql-agent
- sqlite
- data-visualization
- plotly
- pandas
- llm
- machine-learning
- python

## Suggested Website Field

If deployed, use the Streamlit Community Cloud, Hugging Face Spaces, Render, Railway, or another public demo URL.

Until deployed, leave the website field empty rather than linking to a non-working demo.

## Suggested Pinned Repository Description

An AI Data Analytics Assistant that lets users ask questions in natural language, generates SQL with Gemini and LangChain, executes queries on SQLite/CSV data, and renders interactive results in Streamlit.

## Screenshot Checklist

Capture these images and save them under `docs/images/`:

- `dashboard.png`
- `csv-upload.png`
- `sql-generation.png`
- `results-table.png`
- `automatic-chart.png`
- `model-selection.png`

After screenshots are added, the README screenshot section can be updated with image embeds.

## Code Quality Notes

These are presentation/code-quality observations only. They are not required for the documentation upgrade.

- Some UI labels and icon strings appear mojibake-encoded in the current file output. A future pass can normalize the source file encoding to UTF-8.
- Runtime-generated files such as logs, uploaded CSVs, virtual environments, and local SQLite temp artifacts should remain ignored by Git.
- The Streamlit app currently contains both UI rendering and model-list helper logic in one file. This is acceptable for a compact app, but a future refactor could move model discovery helpers into a dedicated frontend utility module.
- A formal `LICENSE` file should be added if the project is intended to be MIT licensed.
- Automated tests could be added for CSV validation, model filtering, answer formatting, and chart selection.
