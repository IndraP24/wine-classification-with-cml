import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

# set random seed
seed = 90

################################
########## DATA PREP ###########
################################

# Load the data
data = pd.read_csv("wine_quality.csv")

# Feature Engineering
#1 - Bad
#2 - Good
#This will be split in the following way. 
#1,2,3,4 --> Bad
#5,6,7,8,9,10 --> Good
#Create an empty list called Reviews
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 4:
        reviews.append(0)
    elif i >= 4 and i <= 10:
        reviews.append(1)
data['Reviews'] = reviews


# Scaling and Split into train and test sets
X = data.drop(["quality", "Reviews"], axis=1)

sc = StandardScaler()
X = sc.fit_transform(X)
y = data["Reviews"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit model on train dataset
rf = RandomForestClassifier(n_estimators=200, max_depth=20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Report training set scores
train_score = rf.score(X_train, y_train) * 100
# Report testing set scores
test_score = rf.score(X_test, y_test) * 100

# write scores to a file
with open("metrics.txt", "w") as f:
    f.write("Training accuracy score: %2.2f%%\n" % train_score)
    f.write("Test accuracy score: %2.2f%%\n" % test_score)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = rf.feature_importances_
labels = data.columns
feature_data = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importance"])
feature_data = feature_data.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_data)
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)  # ylabel
ax.set_title('Random forest\nfeature importance', fontsize=title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()


##########################################
############ PLOT CONFUSION MATRIX  #############
##########################################

plot_confusion_matrix(rf, X_test, y_test)

plt.tight_layout()
plt.savefig("confmat.png", dpi=120)
