import pandas as pd

class GatherData(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def _print_file_path(self):
        print(self.file_path)

    def data_shape(self):
        print(self.df.shape)

    def load_data(self):
        if self.file_path:
            self._print_file_path()
            self.df = pd.read_csv(self.file_path)
            print(self.df.shape)
        else:
            print("No File Loaded")
    
    def _remove_nulls(self):
        print("REMOVING NULLS")
        self.df.dropna(inplace=True)
    
    def _build_building_class(self):
        print("BUILDING THE BUILDING CLASS")
        df = self.df.copy()
        # Remove the blank space
        df.loc[:,'BUILDING CLASS'] = df.loc[:,'BUILDING CLASS CATEGORY'].map(lambda x: x.lstrip()[0:2])
        df = df[df['BUILDING CLASS'] != ""]
        df.loc[:,'BUILDING CLASS'] = df.loc[:,'BUILDING CLASS'].astype(int)
        # all Building Classes greater than 17 are non residential
        self.df = df[df.loc[:,'BUILDING CLASS'] <= 17]
    
    def remove_price_outliers(self, min_price=0):
        print("REMOVE OUTLIERS")
        self.df = self.df[self.df['SALE PRICE'] > min_price]
        return self.df

    def _rename_building_category(self):
        """
        Numerous classifications in BUILDING CLASS CATEGORY are poorly labeled, duplicates,
        or contain additional unneeded information.  Clean dictionary applied to replace improper
        classifications
        """
        print("RENAMING CLASS CATEGORY")
        class_cat_rename = {1:"ONE FAMILY DWELLINGS", 2:"TWO FAMILY DWELLINGS", 3:"THREE FAMILY DWELLINGS", 4:"TAX CLASS 1 CONDOS",
                    5:"TAX CLASS 1 VACANT LAND", 6:"TAX CLASS 1 - OTHER", 7:"RENTALS - WALKUP APARTMENTS",
                    8:"RENTALS - ELEVATOR APARTMENTS", 9:"COOPS - WALKUP APARTMENTS", 10:"COOPS - ELEVATOR APARTMENTS",
                    11:"CONDO-RENTALS", 12:"CONDOS - WALKUP APARTMENTS", 13:"CONDOS - ELEVATOR APARTMENTS",
                    14:"RENTALS - 4-10 UNIT", 15:"CONDOS - 2-10 UNIT RESIDENTIAL", 16:"CONDOS - 2-10 UNIT WITH COMMERCIAL UNIT",
                    17:"CONDO COOPS"}
        self.df.loc[:,'BUILDING CLASS CATEGORY'] = self.df.loc[:,'BUILDING CLASS'].replace(class_cat_rename)
    
    def _build_property_type(self):
        print("BUILDING PROPERTY TYPE")
        prop_type = {1:"SingleFamily", 2:"MultiFamily", 3:"MultiFamily", 4:"Condo", 5:"Land", 6:"Other", 7:"Rental",
             8:"Rental", 9:"Coop", 10:"Coop", 11:"Rental", 12:"Condo", 13:"Condo",
             14:"Rental", 15:"Condo", 16:"Condo", 17:"Condo"}
        self.df['PROPERTY TYPE'] = self.df['BUILDING CLASS'].replace(prop_type)
        self.df = pd.get_dummies(self.df, columns=['PROPERTY TYPE']).drop('PROPERTY TYPE_Other', axis=1)

    def _transform_zipcode(self):
        """Convert the zipcode from float to string and front fill with 0s if needed"""
        print("TRANSFORMING ZIPCODE")
        self.df['ZIP CODE'] = self.df['ZIP CODE'].astype(int).astype(str).map(lambda x: x.zfill(5))

    def _build_new_homes(self):
        """Create New feature of year newness for homes built after 2000"""
        print("BUILDING NEW HOMES")
        df = self.df
        df['YEAR BUILT'] = df['YEAR BUILT'].astype(int)
        df['NEW HOMES'] = ["Yes" if year > 2000 else "No" for year in df['YEAR BUILT']]
        self.df = pd.get_dummies(df, columns=["NEW HOMES"]).drop("NEW HOMES_No", axis=1)

    def _build_large_homes(self):
        print("BUILDING LARGE HOMES")
        df = self.df
        df['LARGE HOMES'] = ["Yes" if size > 4000 else "No" for size in df['GROSS SQUARE FEET']]
        self.df = pd.get_dummies(df, columns=['LARGE HOMES']).drop("LARGE HOMES_No", axis=1)

    def preprocess_data(self):
        # self._load_data()
        self._remove_nulls()
        self._build_building_class()
        self._rename_building_category()
        self._build_property_type()
        self._transform_zipcode()
        self._build_new_homes()
        self._build_large_homes()
        self.data_shape()
        return self.df