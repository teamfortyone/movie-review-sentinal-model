import argparse
import sys

from utils import dataset
from scripts.train_model import train_model
from scripts.test_model import test_model


def handle_args(args: dict):
    if args.get('train'):
        training_data = dataset.get_data_from_pickle(args.get('data'))
        iterations = args.get('iterations')
        model_name = args.get('model_name')
        train_model(training_data, iterations, model_name)

    if args.get('test'):
        testing_data = dataset.get_data_from_pickle(args.get('data'))
        path_to_model = args.get('name')
        limit = args.get('limit')
        correct_preds, accuracy = test_model(
            testing_data, path_to_model, limit)
        print(
            f"Correct Predictions: {correct_preds}/{limit if limit else 25000}")
        print(f"Accuracy of the model: {accuracy}")


def main():
    parser = argparse.ArgumentParser(
        description='Movie Review Sentiment Analysis')
    sub_parser = parser.add_subparsers(title='Commands')

    # Commands
    train = sub_parser.add_parser('train')
    test = sub_parser.add_parser('test')

    # Training
    train.add_argument(
        '--iterations', type=int, help='Number of iterations to make for training')
    train.add_argument('--name', default='models/model_artifacts',
                       help='Name to save model as')
    train.add_argument('--data', default='data/train.pkl',
                       help='Data to train model on')
    train.set_defaults(train=True)

    # Testing
    test.add_argument(
        '--model', default='models/model_artifacts', help='Relative path to model')
    test.add_argument(
        '--limit', type=int, help='Number of reviews to test on (should be less than 25000)')
    test.add_argument('--data', default='data/test.pkl',
                      help='Data to test model on')
    test.set_defaults(test=True)

    args = vars(parser.parse_args())
    handle_args(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, Quitting...")
        sys.exit()
    # except Exception as e:
    #     print(e)
    #     sys.exit(1)
