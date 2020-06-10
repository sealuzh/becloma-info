import pandas as pd
import sys

def discern(package_name):
	frame = pd.read_csv("../crawler/reviews.csv")
	frame = frame[frame['package_name']==package_name]
	frame.to_csv('../crawler/reviews.csv', index = False)

def only_crash():
	crash = 'PREDICTED_IS_CRASH_BUG'
	feature = 'PREDICTED_IS_FEATURE_BUG'
	frame = pd.read_csv("../crawler/reviews_1.csv")
	frame = frame[frame[crash]==1]
	frame.to_csv('../crawler/reviews_1.csv', index = False)


if __name__ == '__main__':
	opt = sys.argv[1]
	package_name = sys.argv[2]
	if opt == 'dis':
		discern(package_name)
	else:
		only_crash()