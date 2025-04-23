# bydoux

This repository contains tools and notebooks for analyzing and comparing financial tickers, with a focus on Canadian and US ETFs (FNBs). The main analysis notebook is `side_by_side.ipynb`, which allows you to visually and statistically compare the performance of two tickers over their overlapping history.

## Features

- Download and process historical quotes for selected tickers using `bydoux_tools`.
- Visualize price evolution, annual returns, and compare trends between two assets.
- Compute and plot annualized returns, differences, and statistical summaries.
- Reference data for SociéTerre 100% actions included for benchmarking.

## Ticker Reference

The list of FNBs (ETFs) and their tickers used in `bydoux_tools` is maintained in a shared Google Sheet:

👉 [Google Sheet: FNB & Ticker Reference](https://docs.google.com/spreadsheets/d/1bx3oBEFAmksB6no7_DV7AP_qM9zQ5iHUepc9wcPjXO8/edit?gid=0#gid=0)

Please refer to this sheet for the most up-to-date list of supported tickers and their details.

## Usage

1. Clone this repository.
2. Install the required Python dependencies (see your environment or requirements).
3. Open `side_by_side.ipynb` in Jupyter or VS Code.
4. Select the tickers you want to compare and run the notebook cells.

