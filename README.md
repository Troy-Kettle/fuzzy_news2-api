# Fuzzy NEWS-2

A fuzzy logic implementation of the National Early Warning Score 2 (NEWS-2) system for early detection of clinical deterioration.

## About NEWS-2

The National Early Warning Score 2 (NEWS-2) is a standardized system developed by the Royal College of Physicians for assessing and responding to acute illness. It uses six physiological parameters:

1. Respiratory rate
2. Oxygen saturation
3. Systolic blood pressure
4. Pulse rate
5. Level of consciousness or new confusion
6. Temperature

## Fuzzy Logic Approach

This implementation uses fuzzy logic to enhance the traditional NEWS-2 system. Instead of discrete thresholds, fuzzy logic allows for a gradual transition between categories, potentially providing more nuanced clinical decision support.

Benefits of the fuzzy approach include:
- Handling uncertainty in measurements
- Smoother transitions between risk categories
- Potential for more personalized risk assessment

## Installation

This project uses Poetry for dependency management.

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Usage

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

## API Server

To start the API server for integration with an Electron front-end:

```bash
poetry run start-api
```

This will start a FastAPI server on http://localhost:8000

## Electron Integration

This project is designed to be easily integrated with an Electron front-end. The API provides all necessary endpoints for:

- Calculating NEWS-2 scores
- Retrieving history of assessments
- Visualizing trends over time

See the `/docs` endpoint when running the API server for complete API documentation.

## License

[MIT](LICENSE)

