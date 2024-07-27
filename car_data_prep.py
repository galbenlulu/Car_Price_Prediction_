import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Load the trained imputers, scaler, and encoder
supply_imputer = pd.read_pickle('supply_imputer.pkl')
mean_imputer = pd.read_pickle('mean_imputer.pkl')
median_imputer = pd.read_pickle('median_imputer.pkl')
mode_imputer = pd.read_pickle('mode_imputer.pkl')
scaler = pd.read_pickle('scaler.pkl')
one = pd.read_pickle('one_hot_encoder.pkl')
avg_km_per_year=pd.read_pickle('avg_km_per_year.pkl')

def convert_excel_date(excel_date):
    if isinstance(excel_date, (int, float)):
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(excel_date, 'D')
    else:
        try:
            return pd.to_datetime(excel_date)
        except (ValueError, TypeError):
            return pd.NaT 


def prepare_data(df):
    #df['manufactor'] = df['manufactor'].astype(str)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['model'] = df['model'].astype(str)
    df['Hand'] = pd.to_numeric(df['Hand'], errors='coerce').astype('Int64')
    df['Gear'] = pd.Categorical(df['Gear'])
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce').astype('Int64')
    df['Engine_type'] = pd.Categorical(df['Engine_type'])
    df['Prev_ownership'] = pd.Categorical(df['Prev_ownership'])
    df['Curr_ownership'] = pd.Categorical(df['Curr_ownership'])
    df['Area'] = df['Area'].astype(str, errors='ignore')
    df['City'] = df['City'].astype(str)
    df['Pic_num'] = pd.to_numeric(df['Pic_num'], errors='coerce').fillna(0).astype(int)
    df['Description'] = df['Description'].astype(str)
    df['Color'] = df['Color'].astype(str)
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce').astype('Int64')
    df['Test'] = pd.to_numeric(df['Test'], errors='coerce').astype('Int64')

    #arrange the columns
    df['Cre_date'] = df['Cre_date'].apply(convert_excel_date)
    df['Repub_date'] = df['Repub_date'].apply(convert_excel_date)
    df['Engine_type'] = df['Engine_type'].replace('היבריד', 'היברידי')
    df['Engine_type'] = df['Engine_type'].replace('טורבו דיזל', 'דיזל')
    df['Engine_type'].replace(['דיזל','היברידי','חשמלי' ,'גז'], 'אחר', inplace=True)
    df['manufactor'] = df['manufactor'].replace('Lexsus',"לקסוס")
    df['Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    df['Gear'].replace(['טיפטרוניק'"לא מוגדר",'ידנית' ,'רובוטית',], 'אחר', inplace=True)
    
    # Calculate Car_Age and Ad_Age for the dataset
    current_year = datetime.now().year
    current_date = datetime.now()
    df['Car_Age'] = current_year - df['Year']
    df['Ad_Age'] = (current_date - df['Cre_date']).dt.days
    columns_to_drop = ["Year", "Cre_date"]
    df = df.drop(columns=columns_to_drop)
 
    # Imputation of missing values 
    # Km - Use average kilometers per year from train dataset and multiply by car's age for missing Km values
    missing_km = df['Km'].isnull()
    df.loc[missing_km, 'Km'] = df.loc[missing_km, 'Car_Age'] * avg_km_per_year


    # Supply Score - Fill missing values using KNNImputer trained on train dataset
    df["Supply_score"] = supply_imputer.transform(df[['Supply_score']])

    # Ad_Age - Impute missing values with mean
    df[['Ad_Age']] = mean_imputer.transform(df[['Ad_Age']])

    # Engine capacity - Impute missing values with median
    df['capacity_Engine'] = median_imputer.transform(df[['capacity_Engine']])

    # Gear, Engine_type, Prev_ownership, Curr_ownership - Impute missing values with most frequent
    df[['Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership']] = mode_imputer.transform(df[['Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership']])
    df['Prev_ownership'].replace(['מונית','ליסינג','ממשלתי' 'לא מוגדר','השכרה', 'חברה'], 'אחר', inplace=True)
    
    # List of all numerical columns
    numeric_cols = ['capacity_Engine', 'Km', 'Supply_score', 'Car_Age', "Hand", "Car_Age"]
    df['Supply_score'] = df['Supply_score'].astype(int)
    df['capacity_Engine'] = df['capacity_Engine'].astype(int)

    
    # Scaling the data
    numeric_cols = ['Supply_score', 'Ad_Age', "Km", "capacity_Engine", "Hand", "Car_Age"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df=df.drop(columns=[ "Hand", "Ad_Age"])
    
    columns_to_drop = ['model', 'Description', 'City', 'Area', 'Color', 'Repub_date', 'Pic_num', 'Test', "Curr_ownership", "Gear"]
    df = df.drop(columns=columns_to_drop)

    # One hot encoder for categorical variables in test set using the same encoder fitted on train set
    one_df = pd.DataFrame(one.transform(df[["manufactor", 'Prev_ownership', "Engine_type"]]), 
                          columns=one.get_feature_names_out())

    # Identify encoded columns for each variable in test set
    manufactor_columns = [col for col in one_df.columns if col.startswith('manufactor_')]
    prev_columns = [col for col in one_df.columns if col.startswith('Prev_ownership_')]
    engine_type_columns = [col for col in one_df.columns if col.startswith('Engine_type_')]

    # Remove the first column from each encoded group in test set
    columns_to_drop = [manufactor_columns[0], prev_columns[0], engine_type_columns[0]]
    one_reduced = one_df.drop(columns=columns_to_drop)

    # Combine the original test set (without the original categorical columns) with the reduced one hot encoded columns
    df = pd.concat([df.drop(columns=["manufactor", 'Prev_ownership', "Engine_type"]).reset_index(drop=True), one_reduced], axis=1)

    return df