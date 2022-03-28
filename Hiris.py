#%%
#THE SPARKS FOUNDATION #GRIPMARCH2022
#AUTHOR: OBERE JUSTIN
#TASK6: PREDICTION USING DECISION TREE ALGORITHM
#LEVEL: INTERMEDIATE #DATA SCIENCE AND BUSINESS ANALYTICS
#%%
#Import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
# %%
#Load Data into DataFrame
df = pd.read_csv('Iris.csv')
# %%
df.head()
# %%
df.shape
# %%
#Missing value check
df.isnull().sum()
# %%
df.nunique()
#%%
df.dtypes
# %%
df['Species'].unique()
# %%
df['Species'].value_counts()

#%%
#Dropping 'Id' column
df = df.drop(['Id'], axis = 1)
#%%
#Transforming Target variable 'Species' into ML mode
df['Species'] = df['Species'].astype('category')
#%%
df["Target"] = df['Species'].cat.codes
#%%
df.sample(10)
#%%
#Droping Species column
df = df.drop(['Species'], axis = 1)
#%%
#Splitting Features vs Target
y = df['Target']
x = df.drop('Target', axis = 1)
# %%
# Define and fit the decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x, y)
# %%
# Import necessary libraries for graph viz
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
# %%
# To visualize the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=x.columns,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# %%
#CONCLUSION: IF WE FEED ANY NEW DATA TO THIS CLASSIFIER
#IT WILL PREDICT THE RIGHT CLASS ACCORDINGLY