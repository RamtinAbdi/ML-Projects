
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

class Preprocessor: 
    def __init__(self, df):
        self.df = df.copy()
        
    def handle_missing_values(self):
        self.df.fillna(0, inplace=True)

    def divider(self):
       self.df['Age_group'] = pd.cut(self.df['Age'], bins=5, labels=[1, 2, 3, 4, 5])       
       self.df['Income_group'] = pd.cut(self.df['Annual Income (k$)'], bins=5, labels=[1, 2, 3, 4, 5])     
       self.df['Score_group'] = pd.cut(self.df['Spending Score (1-100)'], bins=5, labels=[1, 2, 3, 4, 5])     

    def drop_id(self):
        self.df.drop(columns=['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'], inplace=True)

    def label(self):
        mapping = {'Female': 0, 'Male': 1}
        self.df['Gender'] = self.df['Gender'].map(mapping)

    def scaler(self):
        std = StandardScaler()
        self.df[self.df.columns] = std.fit_transform(self.df[self.df.columns])

    def transform(self): 
        self.handle_missing_values()
        self.divider()
        self.label()
        self.drop_id()
        self.scaler()
        return self.df
