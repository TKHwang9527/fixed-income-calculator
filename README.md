# Fixed Income Calculator

A Python-based fixed income analytics tool built for CFA Level 2 exam preparation and portfolio demonstration. Implements core bond mathematics from scratch using NumPy and SciPy.

## Features

- **Bond Calculator** — Price, Macaulay Duration, Modified Duration, DV01, Convexity
- **Yield Curve Bootstrapper** — Bootstrap spot rates and forward rates from par rates
- **Binomial Interest Rate Tree** — Calibrated BDT model with backward induction; prices straight, callable, and putable bonds
- **OAS Calculator** — Z-spread and Option-Adjusted Spread for callable bonds

## Tech Stack

Python · Streamlit · NumPy · SciPy · Pytest

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure
