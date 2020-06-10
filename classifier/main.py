import click
from utils import *

@click.group()
def execute():
    pass


@execute.command()
@click.option("--filepath", "-f", default="./data/reviews_undersampled.csv", help="Filepath to file used for evaluation.")
@click.option("--review_field", "-r", default="review", help="Field containing the review text.")
@click.option("--categories", "-c", help="Fields containing the categories.", multiple=True)
@click.option("--cached", "-ch", is_flag=True, help="Whether to use cached results or recalculate them")
@click.option("--cross_validation", "-k", default=5, help="The k value, for k-fold crossvalidation")
def evaluate(filepath, review_field, categories, cached, cross_validation):
    load_or_evaluate_classification(filepath, review_field, categories, cached, cross_validation)


@execute.command()
@click.option("--filepath", "-f", default="./data/reviews_undersampled.csv", help="Filepath to file used for training.")
@click.option("--review_field", "-r", default="review", help="Field containing the review text.")
@click.option("--categories", "-c", help="Fields containing the categories.", multiple=True)
def train(filepath, review_field, categories):
    train_and_save_model(filepath, review_field, categories)


@execute.command()
@click.option("--filepath", "-f", default="./data/reviews_undersampled.csv", help="Filepath to file used for classification.")
@click.option("--categories", "-c", default="classification", help="Fields containing the categories.", multiple=True)
def classify(filepath, categories):
    classify_and_save_results(filepath, categories)


if __name__ == '__main__':
    execute()