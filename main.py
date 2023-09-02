# import all required library and Algorithm/Model
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("""
Explore Different Datasets with Different Classifier
""")

# A side_bar in which we select the required Dataset
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine', 'Diabetes', 'Digits')
)

st.write(f"## {dataset_name} Dataset")

# A side bar in which we select the required Algorithm/Model
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)


def get_dataset(name):
    data = None
    # load Iris dataset on data
    if name == 'Iris':
        data = datasets.load_iris()
    # load Wine dataset on data
    elif name == 'Wine':
        data = datasets.load_wine()
    # Load the diabetes dataset on data
    elif name == 'Diabetes':
        data = datasets.load_diabetes()
    # Load the digits dataset on data
    elif name == 'Digits':
        data = datasets.load_digits()
    # Load the Breast Cancer dataset on data
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

x , y = get_dataset(dataset_name)

# to find the shape of data (dataset)
st.write('Shape of dataset:', x.shape)

# to find the number of class in  data (dataset)
st.write('number of classes:', len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        c = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
        params['C'] = c
        params['kernel'] = kernel
    elif clf_name == 'KNN':
        k = st.sidebar.slider('Number of Neighbors (K)', 1, 15)
        params['K'] = k
        weights = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
        params['weights'] = weights
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'], kernel=params['kernel'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])
    else:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            criterion=params['criterion'],
            random_state=1234
        )
    return clf


clf = get_classifier(classifier_name, params)
# CLASSIFICATION

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train Our Classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculating Accuracy (Actual and Predicted Label)
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

# PLOT DATASET
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(x)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
