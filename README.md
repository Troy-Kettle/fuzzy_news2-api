# Fuzzy NEWS-2

A fuzzy logic implementation of the National Early Warning Score 2 (NEWS-2) system for early detection of clinical deterioration.

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Fuzzy NEWS-2 enhances the traditional NEWS-2 clinical assessment tool by applying fuzzy logic to create smoother transitions between risk categories. Instead of relying on sharp thresholds that can lead to abrupt changes in risk assessment, this implementation allows for more nuanced evaluation of patient status.

## About NEWS-2

The National Early Warning Score 2 (NEWS-2) is a standardized system developed by the Royal College of Physicians for assessing acute illness severity. It evaluates six physiological parameters:

| Parameter | Measurement |
|-----------|-------------|
| Respiratory rate | breaths per minute |
| Oxygen saturation | % |
| Systolic blood pressure | mmHg |
| Pulse rate | beats per minute |
| Level of consciousness | Alert/Voice/Pain/Unresponsive |
| Temperature | °C |

Each parameter is assigned a score based on its deviation from normal values. The total NEWS-2 score determines the risk category and corresponding clinical response.

## Advantages of Fuzzy Logic Approach

Our implementation enhances the traditional NEWS-2 system with fuzzy logic, providing several benefits:

- **Gradient transitions**: Smoother transitions between risk categories rather than sudden jumps
- **Improved handling of borderline cases**: More accurate risk assessment for patients with values near thresholds
- **Uncertainty management**: Better handling of measurement uncertainties
- **Personalization potential**: Framework for more personalized risk assessment

## Features

- Complete NEWS-2 implementation based on official guidelines
- Fuzzy logic enhancement for more nuanced scoring
- Command-line interface for quick calculations
- REST API for integration with other systems
- Comprehensive Python package for easy incorporation into healthcare applications

## Installation

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-username/fuzzy-news2.git
cd fuzzy-news2

# Install dependencies with Poetry
poetry install
```

## Usage

### Command Line Interface

Calculate NEWS-2 score for a patient:

```bash
# Activate the Poetry environment
poetry shell

# Calculate a NEWS-2 score
python -m fuzzy_news2 calculate \
  --respiratory-rate 22 \
  --oxygen-saturation 94 \
  --systolic-bp 110 \
  --pulse 105 \
  --consciousness A \
  --temperature 38.5 \
  --supplemental-oxygen
```

### Python API

```python
from fuzzy_news2 import FuzzyNEWS2

# Create a FuzzyNEWS2 instance
fuzzy_news = FuzzyNEWS2()

# Calculate NEWS-2 score with fuzzy logic
result = fuzzy_news.calculate(
    respiratory_rate=22,
    oxygen_saturation=94,
    systolic_bp=110,
    pulse=105,
    consciousness="A",  # A for Alert, V for Voice, P for Pain, U for Unresponsive
    temperature=38.5,
    supplemental_oxygen=False
)

print(f"Crisp NEWS-2 Score: {result.crisp_score}")
print(f"Fuzzy NEWS-2 Score: {result.fuzzy_score}")
print(f"Risk Category: {result.risk_category}")
print(f"Recommended Response: {result.recommended_response}")
```

### REST API

Start the API server:

```bash
poetry run python -m fuzzy_news2 api
```

This starts a FastAPI server on http://localhost:8000 with the following endpoints:

- `POST /api/calculate`: Calculate NEWS-2 score
- `GET /api/history/{patient_id}`: Get patient history
- `GET /api/statistics/{patient_id}`: Get patient statistics
- `GET /api/health`: Health check endpoint

Example API request:

```bash
curl -X POST http://localhost:8000/api/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST-001",
    "respiratory_rate": 22,
    "oxygen_saturation": 94,
    "systolic_bp": 110,
    "pulse": 105,
    "consciousness": "A",
    "temperature": 38.5,
    "supplemental_oxygen": false
  }'
```

## API Documentation

When the API server is running, full API documentation is available at:
- OpenAPI UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technical Details

### Custom Fuzzy Logic Implementation

This project implements a custom fuzzy logic system compatible with Python 3.12+, as scikit-fuzzy is not currently compatible with newer Python versions. The implementation includes:

- Triangle, trapezoidal, and Gaussian membership functions
- Mamdani inference method
- Centroid defuzzification

### Project Structure

```
fuzzy-news2/
├── fuzzy_news2/           # Main package
│   ├── __init__.py        # Package exports
│   ├── api.py             # REST API implementation
│   ├── custom_fuzzy.py    # Custom fuzzy logic implementation
│   ├── fuzzy_logic.py     # Fuzzy logic wrapper
│   ├── news2.py           # NEWS-2 implementation
│   └── utils.py           # Helper utilities
├── tests/                 # Test suite
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Development

### Running Tests

```bash
# Run the full test suite
poetry run pytest

# Run specific tests
poetry run pytest tests/test_news2.py

# Run with verbose output
poetry run pytest -v
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Plans

- Graphical user interface for easier use in clinical settings
- Integration with Electronic Health Record (EHR) systems
- Machine learning extensions to personalize risk assessments
- Electron-based desktop application
- Mobile application for bedside assessments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Royal College of Physicians](https://www.rcplondon.ac.uk/) for developing the NEWS-2 system
- The fuzzy logic and healthcare informatics communities

