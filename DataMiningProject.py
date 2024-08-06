import numpy as np
import pandas as pd
from scipy import stats 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnxruntime as rt
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import graphviz


data = pd.read_csv('D://DataScience/2ndSem/Data Mining/Project/data_public.csv')

np.sum(pd.isna(data))
data.isnull().sum()

data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
X = data.iloc[:,0:15]
y = data.iloc[:, 15]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 123)

fig = sns.pairplot(data.sample(1000), hue = 'Class')
fig.savefig('D://DataScience/2ndSem/Data Mining/Project/Pairplot.png', 
            format = 'png', 
            dpi = 600)

corrMatt = X_train.corr()
fig, ax = plt.subplots()
sns.heatmap(corrMatt, annot = True, annot_kws={"size": 5}, square = True)
fig.savefig('D://DataScience/2ndSem/Data Mining/Project/CorrMatt.png', 
            format = 'png', 
            dpi = 1200)

cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

plt.figure()
fig, ax = plt.subplots(3,5)
fig.subplots_adjust(hspace = 0.7, wspace = 0.7)
plt.suptitle('Feature Distribution')
for i in range(0,3):
    for j in range(0,5):
        if i == 0:
            sns.distplot(data[cols[i+j]], bins = 10, ax = ax[i][j])
        elif i == 1:
            sns.distplot(data[cols[i+j+4]], bins = 10, ax = ax[i][j])
        else:
            sns.distplot(data[cols[i+j+8]], bins = 10, ax = ax[i][j])
fig.savefig('D://DataScience/2ndSem/Data Mining/Project/FeatureDist.png', 
            format = 'png', 
            dpi = 1200)

summary_stats = X.describe()
summary_stats
stats.kurtosis(X, axis = 1)
stats.skew(X, axis = 1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


pca = PCA(random_state = 123)
PC = pca.fit_transform(X) 

plt.figure(figsize = (8,5))
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.suptitle('Scree Plot')
plt.subplot(1,2,1)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.subplot(1,2,2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('D://DataScience/2ndSem/Data Mining/Project/ScreePlot.png', 
            format = 'png', 
            dpi = 1200)

pca = PCA(n_components = 3)
pca.fit(X_train)
X_train = pd.DataFrame(pca.transform(X_train))
X_test = pd.DataFrame(pca.transform(X_test))  

Logit = LogisticRegression(solver = 'lbfgs')
Logit.fit(X_train, y_train)
y_pred_logit = Logit.predict(X_test)


print(classification_report(y_test, y_pred_logit)) 
confMat_logit = confusion_matrix(y_test, y_pred_logit) 
# Visualising the confusion matrix
plt.figure()
fig, ax = plt.subplots()
sns.heatmap(confMat_logit, annot = True, ax = ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix - Logistic Regression')
fig.savefig('D://DataScience/2ndSem/Data Mining/Project/ConfMat_Logit.png', 
            format = 'png', 
            dpi = 1200)

Tree = DecisionTreeClassifier(max_depth = 5) 
Tree.fit(X_train, y_train)
y_pred_tree = Tree.predict(X_test)
print(classification_report(y_test, y_pred_tree))

confMat_dt = confusion_matrix(y_test, y_pred_tree) 
# Visualising the confusion matrix
plt.figure()
fig, ax = plt.subplots()
sns.heatmap(confMat_dt, annot = True, ax = ax)
ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix - Decision Tree')
fig.savefig('D://DataScience/2ndSem/Data Mining/Project/ConfMat_DT.png', 
            format = 'png', 
            dpi = 1200)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 1000, 
                                                    random_state = 234)

pipeline = PMMLPipeline([
    ('mapper',
     DataFrameMapper([
         (X_train.columns.values,
          [ContinuousDomain(),
           SimpleImputer(strategy = 'median'),
           StandardScaler()])])),
    ('pca', PCA(n_components=3)),
    ('selector', SelectKBest(k=2)),
    ('classifier', DecisionTreeClassifier(random_state = 45))
])
    

pipeline.fit(X_train, y_train)
pipeline_pred = pipeline.predict(X_test)   
print(classification_report(y_test, pipeline_pred))
 
sklearn2pmml(pipeline,
             'D://DataScience/2ndSem/Data Mining/Project/Project_PMML_pipeline.pmml',
             with_repr = True)


column_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler()) 
])
preprocessor = ColumnTransformer(transformers=[
        ('feature', column_transformer, cols)
]) 

pipeline_onnx = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components = 3)),
        ('selector', SelectKBest(k = 2)),
        ('classifier', DecisionTreeClassifier())
])    

pipeline_onnx.fit(X_train, y_train)
pipeline_onnx.predict(X_test)   




inputs_onnx_train = dict([(x, FloatTensorType([None, 1])) for x in X_train.columns.values])

try:
    model_onnx = convert_sklearn(pipeline_onnx,
                                 'pipeline_project_onnx',
                                initial_types=list(inputs_onnx_train.items()))
except Exception as e:
    print(e)

with open("D://DataScience/2ndSem/Data Mining/Project/Project_ONNX_pipeline.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())


inputs_onnx_test = {k: np.array(v).astype(np.float32)[:, np.newaxis] for k, v in X_test.to_dict(orient='list').items()}
session_onnx = rt.InferenceSession("D://DataScience/2ndSem/Data Mining/Project/Project_ONNX_pipeline.onnx")
predict_onnx = session_onnx.run(None, inputs_onnx_test)   
print(classification_report(y_test, predict_onnx[0]))

pydot_graph = GetPydotGraph(model_onnx.graph,
                            name=model_onnx.graph.name,
                            rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring",
                                                            color="black",
                                                            fillcolor="yellow",
                                                            style="filled"))

gv = graphviz.Source(pydot_graph)
gv
gv.render('D://DataScience/2ndSem/Data Mining/Project/pydot_graph.dot')
graphviz.render('dot', 'png', 'D://DataScience/2ndSem/Data Mining/Project/pydot_graph.dot')

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
feature_importances = clf.feature_importances_
#std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(feature_importances)[::-1]
sorted_cols = []
for i in indices:
    sorted_cols.append(cols[i])

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), feature_importances[indices], color="blue", align="center")
plt.xticks(range(X.shape[1]), sorted_cols)
#plt.xlim([-1, X.shape[1]])
plt.show()








            
