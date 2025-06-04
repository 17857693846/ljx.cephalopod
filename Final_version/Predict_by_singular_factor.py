import pandas as pd
import numpy as np

keys = ["Lat", "Lon", "SST", "Chla", "SSH", "SOI", "SSS", "DO", "BT"]
dic_mid = {
    "Lat": [29, 29.25],
    "Lon": [122.5, 122.5],
    "SST": [15.415, 20.738],
    "Chla": [2.98, 2.5],
    "SSH": [-0.0606, 0.1209],
    "SOI": [0.2, 0.6],
    "SSS": [30.103, 31.3825],
    "DO": [9.197, 7.197],
    "BT": [16.277, 21.156]
    }

dic = {
    "Lat": [],
    "Lon": [],
    "SST": [],
    "Chla": [],
    "SSH": [],
    "SOI": [],
    "SSS": [],
    "DO": [],
    "BT": []
    }

scope = {
    "Spring_SST": [12.94967, 21.91134],
    "Spring_Chla": [0.524639, 23.28921],
    "Autumn_SST": [17.0193, 22.94265],
    "Autumn_Chla": [0.25594, 27.14986],
}

#factor is from keys and type(str), min and max are related to csv, step are set by users
def generate_test_dataset(factor, season=4, step=100):
    if season == 4:
        idx = 0
        min_of_fa = scope[f"Spring_{factor}"][0]
        max_of_fa = scope[f"Spring_{factor}"][1]
    elif season == 11:
        idx = 1
        min_of_fa = scope[f"Autumn_{factor}"][0]
        max_of_fa = scope[f"Autumn_{factor}"][1]
    #dots = np.arange(min_of_fa, max_of_fa, (max_of_fa, min_of_fa)/step)
    for i in range(step):
        for key in keys:
            dic[key].append(dic_mid[key][idx])
        dic[factor][i] = min_of_fa + (max_of_fa-min_of_fa)/step*i
    df = pd.DataFrame(dic)
    #print(df)
    return df

if __name__=="__main__":
    df1 = generate_test_dataset("SST")
    df2 = generate_test_dataset("SST")

#嵌入RF获得值，分季节



