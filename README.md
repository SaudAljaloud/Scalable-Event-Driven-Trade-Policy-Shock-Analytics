# TPNS-Transmission

## Scalable Event-Driven Trade Policy Shock Analytics

### A Big Data Architecture for Cross-Asset Volatility Intelligence

------------------------------------------------------------------------

## Overview

TPNS-Transmission implements a scalable, event-driven big data
architecture for analyzing trade policy shocks and their cross-asset
volatility transmission effects.

This repository accompanies the manuscript:

**Scalable Event-Driven Trade Policy Shock Analytics: A Big Data
Architecture for Cross-Asset Volatility Intelligence**\
Submitted to *Big Data and Cognitive Computing (BDCC)*.

The framework enables structured ingestion of policy events, systematic
volatility estimation, and scalable cross-asset transmission analysis
suitable for macro-financial research and systemic risk monitoring.

------------------------------------------------------------------------

## Research Contributions

-   Event-driven macro-financial shock analytics architecture\
-   Modular trade policy shock processing pipeline\
-   Cross-asset volatility transmission measurement framework\
-   Scalable big data design for time-series computation\
-   Reproducible research implementation

------------------------------------------------------------------------

## Repository Structure

    tpns-transmission/
    │
    ├── src/                 # Core source code
    │   ├── ingestion/       # Data ingestion modules
    │   ├── processing/      # Shock and volatility processing
    │   ├── models/          # Analytical models
    │   ├── utils/           # Utility functions
    │   └── main.py          # Execution entrypoint
    │
    ├── data/                # Data directory (not included)
    ├── notebooks/           # Reproducibility notebooks
    ├── docs/                # Documentation and architecture diagrams
    ├── tests/               # Unit tests
    │
    ├── requirements.txt     # Python dependencies
    ├── LICENSE              # License file
    └── README.md            # Project documentation

------------------------------------------------------------------------

## Core Features

-   Event-driven processing design\
-   Time-indexed volatility estimation\
-   Cross-asset transmission analytics\
-   Configurable event window construction\
-   Parallel-ready computation\
-   Reproducible execution environment

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/YOUR_USERNAME/tpns-transmission.git
cd tpns-transmission
```

Create virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Framework

Basic execution:

``` bash
python src/main.py
```

With configuration file:

``` bash
python src/main.py --config config.yaml
```

------------------------------------------------------------------------

## Methodological Pipeline

1.  Trade policy event ingestion\
2.  Signal normalization\
3.  Event window construction\
4.  Volatility computation\
5.  Cross-asset transmission estimation\
6.  Aggregation and reporting

------------------------------------------------------------------------

## Data Requirements

The framework expects:

-   Time-indexed asset return data (CSV format)\
-   Structured event timestamps\
-   Asset metadata configuration

Due to licensing constraints, datasets are not distributed in this
repository.

------------------------------------------------------------------------

## Reproducibility

To replicate results:

1.  Place required datasets in `/data`\
2.  Configure parameters via configuration file\
3.  Execute main pipeline\
4.  Run notebooks in `/notebooks` for analysis

Recommended environment: Python 3.10+

------------------------------------------------------------------------

## Testing

``` bash
pytest tests/
```

------------------------------------------------------------------------

## Citation

If you use this framework in academic work, please cite:

Author(s). (2026).\
*Scalable Event-Driven Trade Policy Shock Analytics:\
A Big Data Architecture for Cross-Asset Volatility Intelligence.*\
Big Data and Cognitive Computing.

(Citation details will be updated upon publication.)

------------------------------------------------------------------------

## License

This project is released under the MIT License.

------------------------------------------------------------------------

## Contact

For academic inquiries, reproducibility questions, or collaboration
requests, please contact:

Saud Aljaloud
s.aljaloud@uoh.edu.sa