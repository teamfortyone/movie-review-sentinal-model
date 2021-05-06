# Movie Review Sentinal

This is the repository containing Python scripts for downloading, training and testing on the data.

Requires Python 3.6 or higher

The model is built with spaCy, which is a natural language processing library in Python. spaCy written in Cython which makes it extremely performant.

We use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

# Download dataset

Run `dataset.sh`

# Prerequisites

    pip3 install -r requirements.txt
    python3 -m spacy download en_core_web_sm

# Train Model

    python3 main.py train [--iterations ITERATIONS] [--name NAME] [--data DATA]

# Test Model

    python3 main.py test [--model MODEL] [--limit LIMIT] [--data DATA]

# References

1. [spaCy: Industrial-Strength Natural Language Processing](https://spacy.io/)
2. [Use Sentiment Analysis With Python to Classify Movie Reviews – Real Python](https://realpython.com/sentiment-analysis-python/)

# Citation

Large Movie Review Dataset

    @InProceedings{maas-EtAl:2011:ACL-HLT2011,
    author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
    title     = {Learning Word Vectors for Sentiment Analysis},
    booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
    month     = {June},
    year      = {2011},
    address   = {Portland, Oregon, USA},
    publisher = {Association for Computational Linguistics},
    pages     = {142--150},
    url       = {http://www.aclweb.org/anthology/P11-1015}
    }
