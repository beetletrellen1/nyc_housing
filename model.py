import numpy as np
import pandas as pd
from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from data_guide import GatherData

class BuildModel(object):

    def __init__(self, df):
        # preprocessed data
        self.df = df
        # Independent Variables
        self.ind = ['BOROUGH', 'NEIGHBORHOOD', 'ZIP CODE', 'LAND SQUARE FEET', 
                    'GROSS SQUARE FEET', 'YEAR BUILT', 'BUILDING CLASS', 'PROPERTY TYPE_Condo',
                    'PROPERTY TYPE_Coop', 'PROPERTY TYPE_Land', 'PROPERTY TYPE_MultiFamily',
                    'PROPERTY TYPE_Rental', 'PROPERTY TYPE_SingleFamily', 'NEW HOMES_Yes',
                    'LARGE HOMES_Yes']
        # Dependent Variable
        self.dep = ['SALE PRICE']
        self.write_path_pred = "./data/nyc_housing_PREDS.csv"
        self.write_path_inf = "./data/RF_Importance.csv"

    def create_X_y(self):
        model_df = self.df.copy()
        X = model_df[self.ind]
        y = model_df[self.dep]
        return X, y
    
    def create_additional_features(self, X):
        # Additional Feature Creation
        X = pd.get_dummies(X, columns=["BOROUGH", "NEIGHBORHOOD", "ZIP CODE", "BUILDING CLASS"], drop_first=True)
        # Age of building
        X['YEAR BUILT'] = X['YEAR BUILT'].astype(int).map(lambda row: 2022-row)
        return X

    def split_data(self, X, y):
        # Split Data
        test_size = 0.2
        print(f"Splitting Data with {test_size} test size")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_linear_regression(self, X, y):
        # Run Model
        print("Building Linear Regression Model")
        lr = LinearRegression()
        lr.fit(X, y)
        return lr
    
    def build_random_forest(self, X, y):
        print("Building Random Forest Model")
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y.values.ravel())
        return rf

    def build_feature_importance(self, rf_object, X):
        print("Inferring Feature Importance")
        feature_names = X.columns
        importances = rf_object.feature_importances_
        rf_importance = pd.Series(importances, index=feature_names)
        return rf_importance
    
    def predict_model(self, model_object, X):
        return model_object.predict(X)
    
    def score_model(self, model_object, X, y):
        print("Scoring Data")
        # Score using MAE
        X_pred = self.predict_model(model_object, X)
        y['preds'] = X_pred
        print(np.mean(np.abs(y['SALE PRICE'] - y['preds'])))
        print("Predictions Made")


    def save_output(self, df, lr_pred, rf_pred, inference):
        print("Writing Output")
        df['LR_Pred'] = lr_pred
        df['RF_Pred'] = rf_pred
        df.to_csv(self.write_path_pred)
        inference.to_csv(self.write_path_inf)

if __name__ == '__main__':
        
    # Load Data
    data = GatherData("./data/nyc_housing.csv")
    data.load_data()
    data.preprocess_data()
    df = data.remove_price_outliers(10)
    model = BuildModel(df)
    X, y = model.create_X_y()
    X = model.create_additional_features(X)
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    lr = model.build_linear_regression(X_train, y_train)
    rf = model.build_random_forest(X_train, y_train)
    infer = model.build_feature_importance(rf, X)
    lr_pred = model.predict_model(lr, X)
    rf_pred = model.predict_model(rf, X)
    model.score_model(rf, X, y)
    model.save_output(df, lr_pred, rf_pred, infer)

   
