import os
import numpy as np
import scipy
import pickle
import json

directory = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133107142023/reward_seeking/days_with_miniscope_recording/"
day = 'day22'
fileName = day + '_experiment'

dict_variables = {'experiment_type' : 'poke_lick_sham',
        'analyzed' : 1}


with open(directory + os.sep + day + os.sep + fileName + '.pickle','rb') as f:
    aaa = pickle.load(f)

with open(directory + os.sep + day + os.sep + fileName + '.txt','w') as f:
    json.dump(aaa,f)
print(fileName,'done')