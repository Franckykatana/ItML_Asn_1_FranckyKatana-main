# Import necessary libraries
import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns

# Set default figure size for plots
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    This class provides functionalities for Exploratory Data Analysis (EDA).
    Made for better visualitation and basic statistics calculation to better underestand the data.
    
    """

    def giveTarget(self):
        """
        Returns the name of the target variable.
        """
        return self.target
    
    def __init__(self, data, target):
        """
        Initializes the edaDF class.

        Parameters:
        data (DataFrame): The dataset for EDA.
        target (str): The target variable name in the dataset.
        """
        self.data = data
        self.target = target
        self.cat = []  # List to store names of categorical columns
        self.num = []  # List to store names of numerical columns

    def info(self):
        """
        Prints info about the dataset.
        """
        return self.data.info()

    def setCat(self, catList):
        """
        Sets the list of categorical columns.

        Parameters:
        catList (list): A list of column names that are categorical.
        """
        self.cat = catList

    def setNum(self, numList):
        """
        Sets the list of numerical columns.

        Parameters:
        numList (list): A list of column names that are numerical.
        """
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        """
        Generates count plots for each categorical column.

        Parameters:
        splitTarg (bool): If True, splits the plots by the target variable.
        show (bool): If True, displays the plot immediately.
        """
        n = len(self.cat)
        cols = 2
        rows = math.ceil(n/cols)
        figure, ax = plt.subplots(rows, cols, figsize=(15, 5*rows))
        ax = ax.flatten()
        for i, col in enumerate(self.cat):
            if splitTarg:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[i])
            else:
                sns.countplot(data=self.data, x=col, ax=ax[i])
        plt.tight_layout()
        if show:
            plt.show()

    def histPlots(self, kde=True, splitTarg=False, show=True):
        """
        Generates histogram plots for each numerical column.

        Parameters:
        kde (bool): If True, adds a Kernel Density Estimate to the histogram.
        splitTarg (bool): If True, splits the plots by the target variable.
        show (bool): If True, displays the plot immediately.
        """
        n = len(self.num)
        cols = 2
        rows = math.ceil(n/cols)
        figure, ax = plt.subplots(rows, cols, figsize=(15, 5*rows))
        ax = ax.flatten()
        for i, col in enumerate(self.num):
            if splitTarg:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[i])
            else:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[i])
        plt.tight_layout()
        if show:
            plt.show()

    def fullEDA(self):
        """
        Displays the full EDA process in an interactive tab format.

        It includes data info, count plots for categorical data, and histogram plots for numerical data.
        """
        tab = widgets.Tab()
        out1, out2, out3 = widgets.Output(), widgets.Output(), widgets.Output()
        tab.children = [out1, out2, out3]
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')

        with out1:
            with widgets.Output():
                self.info()

        with out2:
            with widgets.Output():
                self.countPlots(splitTarg=True, show=False)
                plt.show()

        with out3:
            with widgets.Output():
                self.histPlots(kde=True, show=False)
                plt.show()

        display(tab)
    
