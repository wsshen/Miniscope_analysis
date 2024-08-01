from hmmlearn import hmm
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fileName='/media/watson/UbuntuHDD/feng_Xin/Xin/BLA_TS_lever_press_optogenetic_inhibition/63494.txt'

with open(fileName) as f:
    data = pd.read_csv(f,header=None)

def risingEdgeDetection(trigBit):
    trig_ind = []
    for i in range(1, len(trigBit), 1):
        if trigBit[i - 1] == 0 and trigBit[i] == 1:
            trig_ind.append(int(i))
    return trig_ind

def findNextPoke(press,poke):
    pos = []
    for i in range(1,len(press),1):
        tDiff = [press[i] -j for j in poke if j-press[i]>=0]
        pos.append(poke[np.argmin(tDiff)])
    return pos
def findLastPress(press,sound):
    pos = []
    for i in range(1,len(sound),1):
        tDiff = [sound[i]-j for j in press if sound[i]-j>=0]
        pos.append(press[np.argmin(tDiff)])
    return pos

laser_bit = 7
poke_bit = 0
lever_bit= 2
lick_bit = 6
pump_bit = 4
sound_bit = 1

sound_timeStamps = risingEdgeDetection(data.iloc[:,sound_bit].to_numpy())
poke_timeStamps = risingEdgeDetection(data.iloc[:,poke_bit].to_numpy())
press_timeStamps = risingEdgeDetection(data.iloc[:,lever_bit].to_numpy())

lastPress_pos = findLastPress(press_timeStamps,sound_timeStamps)
nextPoke_pos = findNextPoke(lastPress_pos,poke_timeStamps)

laser_data = data.iloc[:,laser_bit]
off = laser_data.index[laser_data==0]
on = laser_data.index[laser_data==1]

lastPress = np.zeros(laser_data.shape)
nextPoke = np.zeros(laser_data.shape)
lastPress[lastPress_pos] = 1
nextPoke[nextPoke_pos] = 1

data_on_np = data.iloc[on,[poke_bit,lever_bit]].to_numpy()
data_off_np = data.iloc[off,[poke_bit,lever_bit]].to_numpy()
lastPress_on = lastPress[on]
lastPress_off = lastPress[off]
nextPoke_on = nextPoke[on]
nextPoke_off = nextPoke[off]

data_on = pd.concat([pd.DataFrame(lastPress_on.astype(int)),pd.DataFrame(nextPoke_on.astype(int))],axis=1)
data_off = pd.concat([pd.DataFrame(lastPress_off.astype(int)),pd.DataFrame(nextPoke_off.astype(int))],axis=1)
data_all = pd.concat([pd.DataFrame(lastPress.astype(int)),pd.DataFrame(nextPoke.astype(int))],axis=1)

model_on = hmm.MultinomialHMM(n_components = 2)
model_on.fit(data_all)
Z_all = model_on.predict(data_all)
states_all = pd.unique(Z_all)

model_off = hmm.MultinomialHMM(n_components = 2)
model_off.fit(data_off)
Z_off = model_off.predict(data_off)
states_off =  pd.unique(Z_off)

tVector = range(len(Z_all))

plt.figure()
for i in range(2):
    plt.subplot(3,1,i+1)
    plt.plot(tVector,data_on.iloc[:,i])
plt.subplot(3,1,3)
plt.plot(tVector,Z_on)
plt.savefig('test.eps', format='eps')
