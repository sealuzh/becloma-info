# Reviews Classification

## Installation

After cloning the repository, in the main directory type the following commands:

$ virtualenv venv

$ . venv/bin/activate

$ pip install --editable .

pip install pandas

pip install nltk

pip install scikit-learn

pip install scipy

## Usage

### Evaluation

To evaluate the precision, recall and f1-scores using k-fold cross-validation type the following command:

$ main evaluate -c "IS_FEATURE_BUG" -c "IS_CRASH_BUG"

This will run 5-fold crossvalidation on the ./data/reviews_undersampling.csv file using the "IS_FEATURE_BUG" and "IS_CRASH_BUG"
as categories for classification. At the end it will report the mean precision, recall and f1-scores.

The full command with all the possible options is:

$ main evaluate -f <filepath> -r <review_field> -c <category_1> -c <category_2> -k n

*filepath*: the filepath to the csv file containing the labeled reviews used for training.

*review_field*: the column name of the csv file containing the text of the reviews.

*category_1*: the name of the target category column, each time a review belongs to category_1, there should be
value 1 in the cell and value 0 otherwise.

*category_2...category_n*: there can be a variable number of target categories, in this case a new classifier will be
evaluated for each category.

*n*: the k value for the k-fold cross-validation.

Because the cross-validation is time consuming, the results of the last run are cached and can be accessed using the following
command: 

$ main evaluate --cached

### Training

To train a new classifier and store it, you need to run the following command:

$ main train -f <filepath> -r <review_field> -c <category_1> -c <category_2>

The command's parameters have the same meaning as for the previous command.

### Classification

After having trained one or more classifiers you can classify the reviews of a new datafile using the stored models with
the following command:

$ main classify -f <filepath> -c <category_1> -c <category_2> 

This will open <filepath>, classify each review (it saves the review field from the previous command, the names must match),
and save the classification in a new column PREDICTED_category_1. At the end the changes to the file are save to the same 
<filepath> path. For files that also have the category_1, category_2 columns at the end the resulting precision, recall and
f1-scores will also be printed.

### Example

First run 10-fold cross-validation to see how well the ML model can classify the given training set (stored in "./data/understampled_reviews.csv":

$ main evaluate -c "IS_FEATURE_BUG" -c "IS_CRASH_BUG" -k=10

After running the evaluation, first train the classifiers for the IS_FEATURE_BUG and IS_CRASH_BUG categories:
