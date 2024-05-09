
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score, f1_score ,PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
#-------------


data = pd.read_csv('train.csv')
#drop id column as its useless in predict later
test = pd.read_csv('test.csv', usecols=lambda x: x != 'id')


# check data info to check data  types and null values
data.info()
test.info()
pd.set_option('display.max_columns', None)
print(data.describe())
"""
from decripe we find that there are problems in smallest
height and width in Cm and pixels which actual minimum value should be
(240*320)pxiel (6.35cm width , 8.46 height )cm
the samllest px_width =500 so its fine we just change px_height=320, 
sc_h=6.35 , sc_w=8.46
"""
data.loc[data["px_height"] < 320, "px_height"] = 320
data.loc[data["sc_h"] < 8.46 ,"sc_h"]= 8.46
data.loc[data["sc_w"] < 6.35, "m_dep"] = 6.35
# and smallest mobile depth should be .5 cm
data.loc[data["m_dep"] < 0.5, "m_dep"] = 0.5
# Same for test data
test.loc[test["px_height"] < 320,"px_height"]=320
test.loc[test["sc_h"] < 8.46 ,"sc_h"]= 8.46
test.loc[test["sc_w"] < 6.35,"sc_w"]= 6.35
test.loc[test["m_dep"]< 0.5,"m_dep"]= 0.5

cmap_dict = {0: '#FFFFFF', 1: '#ff2a00', 2: '#ff5500', 3: '#ff8000', 4: '#ffaa00', 5: '#ffd500'}
cmap = ListedColormap([cmap_dict[i] for i in range(6)])
# checking corr between elements
fig=plt.gcf()
fig.set_size_inches(18.5, 10.5)
ax = sns.heatmap(data.corr(), annot = True,cmap=cmap, linewidths=0.2)
plt.title('Correlation Between The Features')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
# we observe that there is strong corr between  ram and price_range .92


#building the model using Random Forest
datacopy=data
X = datacopy.drop('price_range', axis=1)
y = datacopy['price_range'].values.reshape(-1, 1)
print( X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 0)
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train.ravel())
y_pred = RF_model.predict(X_test)
#accracy
print(accuracy_score(y_test, y_pred))
# Score
print(RF_model.score(X_train, y_train))
print(RF_model.score(X_test, y_test))

#Confusion Matrix
ConfusionMatrixDisplay.from_estimator(RF_model, X_test, y_test, cmap=cmap)

plt.show()
#classification report
print(classification_report(y_test, y_pred))

# Predicting Test data

predicted_price = RF_model.predict(test)
Ptest=test
Ptest["predicted_price"]=predicted_price
print(Ptest[["predicted_price"]])

#creat price Categories
price_cat = {
    0: 'Low Price',
    1: 'Medium Price',
    2: 'High Price',
    3: 'Very High Price',
}

Ptest["predicted_price_cat"] = Ptest["predicted_price"].map(price_cat)

print(Ptest)

