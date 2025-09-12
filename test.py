import pandas as pd

import numpy as np 


df= pd.read_json("test.json")
print(df.head())

arr=[i for i in df['messages']]

question = []
answer =[] 

for t in arr:
    for i in range(len(t)//2):
        question.append(t[i]['content'])
        answer.append(t[i+1]['content'])


df= pd.DataFrame({"questions":question,"answer":answer})
df.to_excel(r"test_data.xlsx",index=False,engine='openpyxl')