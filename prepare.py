import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

###################### Prep Iris Data ######################

def prep_iris():
    df = acquire.get_iris_data()
    cols_to_drop = ['species_id']
    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df


# split iris data into train test and validate samples
def split_iris(df):
    iris_train, iris_test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species )
    iris_train, iris_validate = train_test_split(iris_train, test_size=.3, random_state=123, stratify=iris_train.species)
    return iris_train, iris_validate, iris_test 

# Validate my split

print(f'train  {iris_train.shape}')
print(f'validate  {iris_validate.shape}')
print(f'test  {iris_test.shape}')

###################### Prep Titanic Data ######################

def prep_titantic():
    df = acquire.get_titanic_data()
    cols_to_drop = ['deck', 'embarked', 'class']
    df = df.drop(columns=cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    df['age'] = df.age.fillna(value='29.70')
    df['age'] = df['age'].astype(float)
    df['fare'] = df.fare.round(2)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

# split iris data into train test and validate samples
def split_titantic(df):
    titantic_train, titantic_test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived )
    titantic_train, titantic_validate = train_test_split(titantic_train, test_size=.3, random_state=123, stratify=titantic_train.survived)
    return titantic_train, titantic_validate, titantic_test


# Validate my split

print(f'train  {titantic_train.shape}')
print(f'validate  {titantic_validate.shape}')
print(f'test  {titantic_test.shape}')

###################### Prep Telco Data ######################

def prep_telco():
    df = acquire.get_telco_data()
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype(float)
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)   
    df = pd.concat([df, dummy_df], axis=1)
    return df

# split telco data into train test and validate samples

def split_telco(df):
    telco_train, telco_test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn )
    telco_train, telco_validate = train_test_split(telco_train, test_size=.3, random_state=123, stratify=telco_train.churn)
    return telco_train, telco_validate, telco_test

# Validate my split

print(f'train  {telco_train.shape}')
print(f'validate  {telco_validate.shape}')
print(f'test  {telco_test.shape}')