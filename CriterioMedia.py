import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff


def criterioMedia():
    data,meta = arff.loadarff('./pima_diabetes.arff')

    attributes = meta.names()
    data_value = np.asarray(data)


    preg = np.asarray(data['preg']).reshape(-1,1)
    plas = np.asarray(data['plas']).reshape(-1,1)
    pres = np.asarray(data['pres']).reshape(-1,1)
    skin = np.asarray(data['skin']).reshape(-1,1)
    insu = np.asarray(data['insu']).reshape(-1,1)
    mass = np.asarray(data['mass']).reshape(-1,1)
    pedi = np.asarray(data['pedi']).reshape(-1,1)
    age = np.asarray(data['age']).reshape(-1,1)
    features = np.concatenate((preg , plas, pres, skin, insu, mass, pedi, age),axis=1)
    target = data['class']


    Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

    plt.figure(figsize=(100, 6.5))
    tree.plot_tree(Arvore,feature_names=['preg' , 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'],class_names=['tested_negative', 'tested_positive'],
                   filled=True, rounded=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 10))
    metrics.plot_confusion_matrix(Arvore,features,target,display_labels=['tested_negative', 'tested_positive'], values_format='d', ax=ax)
    plt.show()

criterioMedia()