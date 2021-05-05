import pickle
import random

import spacy
from spacy.util import minibatch, compounding
from tqdm import tqdm


def train_model(
    training_data: list,
    iterations: int = 20,
    model_name: str = "model_artifacts"
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        print(f"Training for {iterations} iterations...")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in tqdm(range(iterations)):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2,
                           sgd=optimizer, losses=loss)

    # Save model to disk
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("trained_models/{model_name}")
