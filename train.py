import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# set random seed
seed = 90

################################
########## DATA PREP ###########
################################

# Load the data
data = pd.read_csv("wine_quality.csv")

# Split into train and test sets
X = data.drop(["quality"], axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit model on train dataset
rf = RandomForestRegressor(n_estimators=200, max_depth=7)
rf.fit(X_train, y_train)

# Report training set scores
train_score = rf.best_score_(X_train, y_train) * 100
# Report testing set scores
test_score = rf.score(X_test, y_test) * 100

# write scores to a file
with open("metrics.txt", "w") as f:
    f.write("Training variance explained: %2.1f%%\n" % train_score)
    f.write("Test variance explained: %2.1f%%\n" % test_score)

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
############ PLOT RESIDUALS  #############
##########################################

y_pred = rf.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
res_data = pd.DataFrame(list(zip(y_jitter, y_pred)), columns=["true", "pred"])

ax = sns.scatterplot(x="true", y="pred", data=res_data)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality', fontsize=axis_fs)
ax.set_ylabel('Predicted wine quality', fontsize=axis_fs)  # ylabel
ax.set_title('Residuals', fontsize=title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5, 8.5))
plt.xlim((2.5, 8.5))

plt.tight_layout()
plt.savefig("residuals.png", dpi=120)
