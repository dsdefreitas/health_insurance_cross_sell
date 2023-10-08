import pandas as pd
import pickle
import inflection

class HealthInsurance(): 
    #transformando nomes de coluna para snake case    
    def __init__(self): 
        self.annual_premium_scaler = pickle.load(open('src/features/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler = pickle.load(open('src/features/age_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open('src/features/vintage_scaler.pkl', 'rb'))
        self.target_encode_gender_scaler = pickle.load(open('src/features/target_encode_gender_scaler.pkl', 'rb'))
        self.target_encode_region_code = pickle.load(open('src/features/target_encode_region_code_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel = pickle.load(open('src/features/fe_policy_sales_channel_scaler.pkl', 'rb'))
        
    def data_cleaning(self, df1):  
        
        cols_old = df1.columns

        snakecase = lambda x: inflection.underscore(x)

        df1.columns = list(map(snakecase, cols_old))
        
        return df1 

    
    def feature_engineering(self, df2): 
        
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' 
                                                      else 'between_1_and_2_years' if x == '1-2 Year' 
                                                      else 'bellow_1_year')

        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' 
                                                      else 0 )

        return df2 

    def data_preparation(self, df4):  

        #annual_premium

        df4['annual_premium'] = self.annual_premium_scaler.fit_transform(df4[['annual_premium']].values)

        #não se aproximam de uma normal 
        
        #Age

        df4['age'] = self.age_scaler.fit_transform(df4[['age']].values) 

        # vintage

        df4['vintage'] = self.age_scaler.fit_transform(df4[['vintage']].values) 
        
        #gender - Target Enconding 
        df4.loc[:, 'gender'] = df4['gender'].map(self.target_encode_gender_scaler) #substitui male por 0.138411 e female por 0.103902

        #region_code - Target Encoding
        df4.loc[:, 'region_code'] = df4['region_code'].map(self.target_encode_region_code)

        #vehicle_age - One Hot Encoding 
        df4 = pd.get_dummies(df4, prefix='vehicle_age', columns=['vehicle_age'])

        #policy_sales_channel - Frequency Encoding 
        df4.loc[:, 'policy_sales_channel'] = df4['policy_sales_channel'].map(self.fe_policy_sales_channel)    
        
        # 6.0 Feature Selection 
        
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured', 'policy_sales_channel']
        
        return df4[cols_selected]
    
    def get_prediction(self, model, original_data, test_data): 
        #model prediction 
        pred = model.predict_proba(test_data) 

        #join prediciton into original data 
        original_data['prediction'] = pred[:, 1].tolist() 

        return original_data.to_json(orient = 'records', date_format = 'iso')