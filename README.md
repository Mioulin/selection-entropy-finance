# Selection Entropy for Financial Markets
*A novel entropy-based framework for regime detection and market microstructure analysis*

---

## Overview
This repository demonstrates the application of **Selection Entropy (SE)** — a new information-theoretic metric originally developed in neuroscience — to **financial time series and order book data**.

Unlike KL-divergence or Shannon entropy, **Selection Entropy** is:
- **More sensitive** to subtle distributional changes in noisy, non-stationary systems.  
- **Robust** to small-sample fluctuations.  
- **Interpretable** as the *minimal information required to predict future states*.  

SE was first introduced in the context of brain dynamics (see citation below), but its principles transfer naturally to **financial markets**, where early detection of structural changes is critical.

---

## Why Selection Entropy for Finance?
- **Regime Detection**  
  Identify early-warning signals of volatility clustering, liquidity droughts, or factor rotations.
- **Market Microstructure**  
  Quantify entropy of order flow (buy/sell/cancel events) to capture hidden information in the limit order book.
- **Cross-Asset Information Flow**  
  Detect subtle contagion and lead–lag effects across correlated markets.

---

## Repository Structure
```
selection-entropy-finance/
│── README.md
│── requirements.txt
│── src/
│    ├── selection_entropy.py        # SE & KL implementations for time series
│    ├── se_orderbook.py             # SE for order-flow events
│    ├── se_regime_detection.py      # placeholder for additional demos
│    └── utils.py
│── data/
│    ├── toy_SP500.csv               # synthetic S&P500 returns with regime shifts
│    └── sample_orderbook.csv        # synthetic order book events with regime change
│── notebooks/
│    ├── demo_SP500_entropy.ipynb    # SE vs KL on S&P500-like returns
│    └── demo_orderbook_entropy.ipynb# SE applied to synthetic order book
```

---

## Quick Start
Clone the repo and install dependencies:
```bash
git clone git@github.com:mioulin/selection-entropy-finance.git
cd selection-entropy-finance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the demos:
```bash
jupyter notebook notebooks/demo_SP500_entropy.ipynb
jupyter notebook notebooks/demo_orderbook_entropy.ipynb
```

---

## Results
### 1. Regime Detection (S&P500-like Returns)
- **Selection Entropy** detects synthetic regime shifts earlier and with clearer signal than KL divergence.  
- Demonstrates robustness to noise and small-sample fluctuations.  

### 2. Order Book Microstructure
- SE applied to synthetic LOB events highlights structural change in event flow (shift toward cancels and market sells).  
- Captures information not visible in simple volume or imbalance statistics.  



---

## Roadmap
- [ ] Extend to FX tick-level data.  
- [ ] Benchmark against LSTMs/Transformers for forecasting.  
- [ ] Explore FPGA-friendly implementations for low-latency deployment.  

---

## Citation
If using this work, please cite:

> Erik D. Fagerholm, Zalina Dezhina, Rosalyn J. Moran, Karl J. Friston, Federico Turkheimer and Robert Leech *Selection Entropy: The Information Hidden Within Neuronal Patterns*, Physical Review Research, 2023.

---

## License
MIT License © 2025 Zalina Dezhina
