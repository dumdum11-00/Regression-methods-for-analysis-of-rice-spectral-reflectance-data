import pandas as pd
import numpy as np
import os
import re
def get_data_from_file(file):
    with open(file) as f:
        raw_data = f.read().splitlines()
        file_name = os.path.basename(f.name)
    
    # lat_index = [i for i, j in enumerate(raw_data) if 'Latitude' in j][0]
    # long_index = [i for i, j in enumerate(raw_data) if 'Longitude' in j][0]
    reflect_index = [i for i, j in enumerate(raw_data) if 'Data' in j][0]+2
    

    replicate = file_name.split("_")[2].rstrip(".sed").split(".") # Get Replicate data from file name

    if not replicate:
        print("incorrect file: " + file_name)
        return
    elif len(replicate) == 1: # handle abnormally name ex: BC3
        replicate = re.split('(\d+)', replicate[0])
        replicate= [replicate[0] + replicate[1], "1", replicate[2]]
    else:
        replicate[1] = re.split('(\d+)', replicate[1])[1:]
        replicate = np.hstack(replicate)
    
    reflect_data = []
    for i in raw_data[reflect_index:len(raw_data)]:  # Get reflect data from file
        var = i.split("\t")
        reflect_data.append(float(var[3]))

    if replicate[2] == "":
        reflect_data = reflect_data[1:]
        null_percentage = sum(1 for item in reflect_data if item == 0.0)/ len(reflect_data) * 100
        proccessed_data =  np.hstack([replicate[0], float(replicate[1]) , replicate[2], null_percentage, reflect_data])
        print(null_percentage)
        return proccessed_data
    else: 
        return

def megre_data(reflect_df):
    df = pd.read_csv("./DATA_Mua1_2022.csv")
    df = df[df["Sub-Replicate"].notnull()] #Create new dataframe based on sub-replicate
    df.loc[:, ['Replicate']] = df.loc[:,['Replicate']].ffill() #Input the misssing replicate in data

    base_data = df[['Replicate','Sub-Replicate', 'P conc. (mg/kg)', 'K conc. (mg/kg)', 'Chlorophyll-a']]

    base_data['Sub-Replicate'] = base_data['Sub-Replicate'].astype(str)
    merged = pd.merge(base_data, reflect_df, how='left', left_on=['Replicate', 'Sub-Replicate'], right_on=['0', '1'])
    # merged = merged.drop(columns=['Replicate', 'Sub-Replicate', '0', '1', '2', '3']).dropna()
    return merged

reflect_data = []

directory = './Spectral reflectance measurement'
for filename in os.listdir(directory):
    file_dir = os.path.join(directory, filename)
    a = get_data_from_file(file_dir)
    if a is not None:
        reflect_data.append(a)


col = pd.Series(reflect_data)
arr = np.array(col.values.tolist())
reflect_df = pd.DataFrame(columns=[str(i) for i in range(reflect_data[0].size)])
reflect_df[reflect_df.columns] = arr

processed_data= (megre_data(reflect_df)) 
print(processed_data)

processed_data.to_csv('processed_data.csv', index=False)