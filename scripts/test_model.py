import sys
import random

import spacy
from tqdm import tqdm


def test_model(
    testing_data: list,
    path_to_model: str,
    limit: int = 0
) -> (int, int):
    print('Loading model...')
    try:
        loaded_model = spacy.load(path_to_model)
    except Exception as e:
        print(f"Could not load model:\n {e}")
        sys.exit(1)

    if limit:
        random.shuffle(testing_data)
        testing_data = testing_data[:limit]

    num_correct_preds = 0
    num_reviews = len(testing_data)

    print(f"Testing on {num_reviews} reviews...")
    for review, sentiment in tqdm(testing_data):
        ground_sentiment = "pos" if sentiment['cats']['pos'] else "neg"
        parsed_text = loaded_model(review)
        pred = "pos" if parsed_text.cats["pos"] > parsed_text.cats["neg"] else "neg"
        if pred == ground_sentiment:
            num_correct_preds += 1

    accuracy = (num_correct_preds/len(testing_data))*100
    return num_correct_preds, accuracy
