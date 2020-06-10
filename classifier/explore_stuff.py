from utils import *


data = pd.read_csv("./data/filtered_reviews.csv")
print(data.columns)
crash_bugs = data[data["PREDICTED_IS_CRASH_BUG"] == 1]
crash_bugs.to_csv("./data/crash_bugs.csv")

feature_bugs = data[data["PREDICTED_IS_FEATURE_BUG"] == 1]
feature_bugs.to_csv("./data/feature_bugs.csv")
print(len(feature_bugs))

app_crash_bugs = {}

for index, row in crash_bugs.iterrows():
    app = row["package_name"]
    if not app in app_crash_bugs:
        app_crash_bugs[app] = []
    app_crash_bugs[app].append(row["review"])

app_crash_bugs_counts = [(app, len(crash_bugs)) for app, crash_bugs in app_crash_bugs.iteritems()]
app_crash_bugs_counts = sorted(app_crash_bugs_counts, key=lambda x: x[1], reverse=True)
for app, count in app_crash_bugs_counts:
    print("%s: %d" % (app, count))