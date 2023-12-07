def fallingEdgeDetection(trigBit, stimTime):
    trig_time = []
    for i in range(1, len(trigBit), 1):
        if trigBit[i - 1] == '1' and trigBit[i] == '0':
            trig_time.append(float(stimTime[i]))
    return trig_time
