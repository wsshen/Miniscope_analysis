import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mchmm as mc
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
fileName = '/media/watson/UbuntuHDD/feng_Xin/Xin/BLA_TS_lever_press_optogenetic_inhibition/63493.txt'

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
    for i in range(0,len(press),1):
        tDiff = [[j-press[i], j] for j in poke if j-press[i]>=0]
        tDiff_np = np.array(tDiff)
        pos_temp = np.argmin(tDiff_np[:, 0])
        pos.append(tDiff_np[pos_temp,1])
    return pos

def findLastPress(press,sound):
    pos = []
    for i in range(1,len(sound),1):
        tDiff = [[sound[i]-j, j] for j in press if sound[i]-j>=0]
        tDiff_np = np.array(tDiff)
        pos_temp = np.argmin(tDiff_np[:,0])
        pos.append(tDiff_np[pos_temp,1])
    return pos

def downsamplingStates(n,data,threshold):
    data_size = data.shape
    new_data = []
    for i in range(0, data_size[0], n):
        if len(data_size) >1:
            data_block = data[i:i + n, :]
            data_sum = np.sum(np.reshape(data_block, [1, data_block.shape[0] * data_block.shape[1]]))
        else:
            data_block = data[i:i+n]
            data_sum = np.sum(data_block)
        if data_sum < threshold:
            new_data.append(int(0))
        else:
            new_data.append(int(1))
    return np.array(new_data)
def randomSampleChunk_noreplace(array, len_chunk):
    pos = np.random.randint(0,high=len(array)-len_chunk)
    sample_Chunk = array[pos:pos+len_chunk]
    array_rest = np.concatenate((array[0:pos],array[pos+len_chunk:-1]),axis=0)
    return array_rest,sample_Chunk
sampling_rate = 5000

laser_bit = 7
poke_bit = 0
lever_bit= 2
lick_bit = 6
pump_bit = 4
sound_bit = 1

data_np = data.iloc[:,[poke_bit,lever_bit,lick_bit]].to_numpy()
laser_np = data.iloc[:,laser_bit].to_numpy()
data_size = data_np.shape
data_states = []
laser_states = []


data_np = downsamplingStates(sampling_rate,data_np,200)
laser_np = downsamplingStates(sampling_rate,laser_np,200)
data_np = downsamplingStates(4,data_np,2)
laser_np = downsamplingStates(4,laser_np,2)
# for i in range(0,data_size[0],sampling_rate):
#     data_block = data_np[i:i+sampling_rate,:]
#
#     data_sum = np.sum(np.reshape(data_block,[1,data_block.shape[0]*data_block.shape[1]]))
#     if data_sum == 0:
#         data_states.append(int(0))
#     else:
#         data_states.append(int(1))
#     laser_block = laser_np[i:i+sampling_rate]
#     laser_sum = np.sum(laser_block)
#     if laser_sum == 0:
#         laser_states.append(int(0))
#     else:
#         laser_states.append(int(1))

data_states_pd = pd.Series(data_np)
laser_states_pd = pd.Series(laser_np)
on = laser_states_pd.index[laser_states_pd==1]
off = laser_states_pd.index[laser_states_pd==0]
array_rest = off
off_resampled = []
for i in range(len(on)-1):
    if i ==0:
        beginning = i
    if on[i+1]-on[i]>1:
        lenChunk = i - beginning
        beginning = i+1

        array_rest, sampleChunk = randomSampleChunk_noreplace(array_rest,lenChunk)
        print(len(sampleChunk))
        off_resampled = np.concatenate((off_resampled,sampleChunk),axis=0)
off_resampled_pd = pd.Series(off_resampled)
laser_on = laser_states_pd[on]
laser_off = laser_states_pd[off]
data_on = pd.DataFrame(data_states_pd[on])
data_off = pd.DataFrame(data_states_pd[off_resampled_pd])

np.random.seed(42)
model_on = hmm.CategoricalHMM(n_components = 2,params='st',init_params='st')
model_on.emissionprob_ = np.array([[0.8,0.2],
                                  [0.2,0.8]])
model_on.fit(data_on.values)
Z_on = model_on.predict(data_on.values)
states_on = pd.unique(Z_on)

model_off = hmm.CategoricalHMM(n_components = 2,params='st',init_params='st')
model_off.emissionprob_ = np.array([[0.8,0.2],
                                  [0.2,0.8]])
model_off.fit(data_off.values)
Z_off = model_off.predict(data_off.values)
states_off = pd.unique(Z_off)

print("Model_on transition matrix is:", model_on.transmat_)
print("Model_on emission matrix is:",model_on.emissionprob_)
print("Model_off transition matrix is:", model_off.transmat_)
print("Model_off emission matrix is:",model_off.emissionprob_)
input()
# sound_timeStamps = risingEdgeDetection(data.iloc[:,sound_bit].to_numpy())
# poke_timeStamps = risingEdgeDetection(data.iloc[:,poke_bit].to_numpy())
# press_timeStamps = risingEdgeDetection(data.iloc[:,lever_bit].to_numpy())
#
# sound_np = np.array(sound_timeStamps)
# poke_np = np.array(poke_timeStamps)
# press_np = np.array(press_timeStamps)
#
# lastPress_pos = findLastPress(press_np,sound_np)
# nextPoke_pos = findNextPoke(lastPress_pos,poke_np)
#
# laser_data = data.iloc[:,laser_bit]
# poke_data = data.iloc[:,poke_bit]
# lever_data = data.iloc[:,lever_bit]
#
# off = laser_data.index[laser_data==0]
# on = laser_data.index[laser_data==1]
#
# lastPress = np.zeros(laser_data.shape)
# nextPoke = np.zeros(laser_data.shape)
# lastPress[lastPress_pos] = 1
# nextPoke[nextPoke_pos] = 1



# data_on_np = data.iloc[on,[poke_bit,lever_bit]].to_numpy()
# data_off_np = data.iloc[off,[poke_bit,lever_bit]].to_numpy()
# lastPress_on = lastPress[on]
# lastPress_off = lastPress[off]
# nextPoke_on = nextPoke[on]
# nextPoke_off = nextPoke[off]

# data_on = pd.concat([pd.DataFrame(lastPress_on.astype(int)),pd.DataFrame(nextPoke_on.astype(int))],axis=1)
# data_off = pd.concat([pd.DataFrame(lastPress_off.astype(int)),pd.DataFrame(nextPoke_off.astype(int))],axis=1)
# data_all = pd.concat([pd.DataFrame(lastPress.astype(int)),pd.DataFrame(nextPoke.astype(int))],axis=1)

# a = mc.MarkovChain().from_data(nextPoke_on)

# model_on = hmm.MultinomialHMM(n_components = 2)
# model_on.fit(data_on)
# Z_on = model_on.predict(data_on)
# states_on = pd.unique(Z_on)
#
# model_off = hmm.MultinomialHMM(n_components = 2)
# model_off.fit(data_off)
# Z_off = model_off.predict(data_off)
# states_off = pd.unique(Z_off)

plt.figure
plt.subplot(2,1,1)
plt.plot(data_on.values)
plt.subplot(2,1,2)
plt.plot(data_off.values)
plt.show(block=True)