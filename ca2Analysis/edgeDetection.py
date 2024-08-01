def edgeDetection(trigBit, stimTime):
    trig_time = []
    for i in range(1, len(trigBit), 1):
        if trigBit[i - 1] != trigBit[i]:
            trig_time.append(float(stimTime[i]))
        else:
            #print(stimTime[i])
            continue
    return trig_time

def fallingEdgeDetection(trigBit, stimTime):
    trig_time = []
    for i in range(1, len(trigBit), 1):
        if trigBit[i - 1] == '1' and trigBit[i] == '0':
            trig_time.append(float(stimTime[i]))
    return trig_time

def risingEdgeDetection(trigBit, stimTime):
    trig_time = []
    for i in range(1, len(trigBit), 1):
        if trigBit[i - 1] == '0' and trigBit[i] == '1':
            trig_time.append(float(stimTime[i]))
    return trig_time

def boutEndDetection(trigTime,bout_interval):
    trig_time = []
    for i in range(1, len(trigTime), 1):
        if trigTime[i] - trigTime[i-1] > bout_interval:
            trig_time.append(float(trigTime[i-1]))
    return trig_time