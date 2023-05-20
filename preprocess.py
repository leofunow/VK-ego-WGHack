import pandas as pd
import numpy as np
import gc

from sklearn.metrics import mean_squared_error

def tmp(x, y):
    if x == -1 or y == -1:
        return -1
    return abs(x - y) 

def dis(x,y):
    return abs(x-y) if x > 0 and y > 0 else -1

def sex_cat(x, y):
    if  x == 1 and y == 1:
        return 1
    elif x == 2 and y == 2:
        return 2
    else:
        return 0
    
def is_the_same(x, y):
    if  x == y and x != -1:
        return 2
    elif x != -1 or y != -1:
        return 0
    else:
        return 1
    
def age_eps(x):
    if x <= 8:
        return 1
    else: 
        return 0



def preprocess(df_name):
    train = pd.read_csv(df_name)
    attr = pd.read_csv("attr.csv")

    train_2 = pd.merge(train, attr, on=['ego_id', 'u']).merge(attr, right_on=['ego_id', 'u'], left_on=['ego_id', 'v'])
    del attr 
    gc.collect()
    train_2 = train_2.drop_duplicates()
    train_2["city_dist"] = list(map(tmp, train_2["city_id_x"], train_2["city_id_y"]))
    train_2["university_dist"] = list(map(tmp, train_2["university_x"], train_2["university_y"]))
    train_2["sex"] = (train_2["sex_x"] + train_2["sex_y"]) % 2
    train_2["school_dist"] = list(map(tmp, train_2["school_y"], train_2["school_x"]))
    train_2["age_dist"] = abs(train_2["age_x"] - train_2["age_y"])
    train_2["age_mean"] = (train_2["age_x"] + train_2["age_y"]) / 2
    train_2['sex2'] = list(map(sex_cat, train_2["sex_x"], train_2["sex_y"]))
    train_2["city"] = list(map(is_the_same, train_2["city_id_x"], train_2["city_id_y"]))
    train_2["university"] = list(map(is_the_same, train_2["university_x"], train_2["university_y"]))
    train_2["school"] = list(map(is_the_same, train_2["school_y"], train_2["school_x"]))
    train_2["gen"] = list(map(age_eps, train_2["age_dist"]))
    train_3 = train_2.drop(["city_id_x", "city_id_y", "university_x", "university_y", "sex_x", "sex_y", "school_x", "school_y", "age_x", "age_y", "u_y"], axis=1)
    del train_2
    gc.collect()
    train_3.rename(columns= {"u_x": "u"} , inplace=True)
    grouped_data1 = train_3.groupby(['ego_id', 'u']).agg({'v':'count'}).reset_index().rename(columns = {'v':'v_count'})
    grouped_data = train_3.groupby(['ego_id', 'v']).agg({'u':'count'}).reset_index().rename(columns = {'u':'u_count'})
    grouped_data = pd.merge(grouped_data1, grouped_data, left_on = ['ego_id', 'u'], right_on = ['ego_id', 'v'])
    grouped_data['freinds'] = grouped_data['v_count']+grouped_data['u_count']
    df = pd.merge(train_3, grouped_data[['ego_id', 'u', 'freinds']], on = ['ego_id', 'u'])
    df = pd.merge(df, grouped_data[['ego_id', 'v', 'freinds']], on = ['ego_id', 'v'])
    del train_3, grouped_data, grouped_data1
    gc.collect()
    df = df.rename(columns = {'freinds_x':'freinds_u', 'freinds_y':'freinds_v'})
    df['u_v'] = (1 + df['x2'])*df['freinds_u'] + (1 + df['x3'])*df['freinds_v']
    df = df.drop(['freinds_u', 'freinds_v'], axis=1)

    df = pd.get_dummies(df, columns=['gen', 'sex2', 'city', 'university'])
    colnames = df.columns[5:]
    return df, colnames


