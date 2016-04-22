import sys 
import os 

def main():
    import pandas as pd
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from IPython.display import Image
    
    from sklearn import tree
    from sklearn.cross_validation import train_test_split, cross_val_score
    from sklearn.externals.six import StringIO  
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

    pd.set_option('display.notebook_repr_html', False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.max_seq_items', None)
    
    import seaborn as sns
    sns.set_context('notebook')
    sns.set_style('white')
    
    df = pd.read_csv("/Users/ewenwang/Dropbox/Data Science/SLML/Statistical Learning and Data Mining/ISL/Hitters.csv").dropna()
    df.info()
    
    X = df[['Years', 'Hits']].as_matrix()
    y = np.log(df.Salary.as_matrix())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 4))
    ax1.hist(df.Salary.as_matrix())
    ax1.set_xlabel('Salary')
    ax2.hist(y)
    ax2.set_xlabel('Log(Salary)')
    plt.show()
    
    regr = DecisionTreeRegressor(max_leaf_nodes=3)
    
    df.plot('Years', 'Hits', kind = 'scatter', color = 'orange', figsize = (7, 6))
    plt.xlim(0, 25)
    plt.ylim(ymin = -5)
    plt.xticks([4.5])
    plt.yticks([177.5])
    plt.vlines(4.5, ymin=-5, ymax=250)
    plt.hlines(117.5, xmin=4.5, xmax=25)
    plt.annotate('R1', xy=(2,117.5), fontsize='xx-large')
    plt.annotate('R2', xy=(11,60), fontsize='xx-large')
    plt.annotate('R3', xy=(11,170), fontsize='xx-large')
    plt.show()
    
    
if __name__ == "__main__":
    main()