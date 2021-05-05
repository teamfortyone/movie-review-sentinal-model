import os
import pickle


def load_data(
    data_directory: str = "aclImdb/train",
    limit: int = 0
) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding='utf-8') as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))

    if limit:
        reviews = reviews[:limit]
    return reviews


def save_data_to_pickle(
    reviews: tuple,
    filename: str
):
    with open(f"data/{filename}", "wb") as f:
        pickle.dump(reviews, f)


def get_data_from_pickle(
    path_to_pickle: str
) -> list:
    with open(path_to_pickle, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    try:
        print("Loading training dataset...")
        training_reviews = load_data()
        save_data_to_pickle(training_reviews, 'train.pkl')
        print("Saved training dataset")

        print("Loading testing dataset...")
        training_reviews = load_data()
        save_data_to_pickle(training_reviews, 'test.pkl')
        print("Saved testing dataset")
    except Exception as e:
        print(f"Something went wrong while loading datasets: {e}")
