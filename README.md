# Movie Review Sentinal

This is the repository containing Python scripts for downloading, training and testing on the data.

Requires Python 3.6 or higher

# Download dataset

Run `dataset.sh`

# Prerequisites

    pip install -r requirements.txt
    python3 -m spacy download en_core_web_sm

# Train Model

    python3 main.py train [--iterations ITERATIONS] [--name NAME] [--data DATA]

# Test Model

    python3 main.py test [--model MODEL] [--limit LIMIT] [--data DATA]
