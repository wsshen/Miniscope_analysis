import numpy as np
def calTempResolution(timePoints):

    timeDiff = np.subtract(timePoints[1:-1], timePoints[0:-2])
    frameDur = float(format(np.mean(timeDiff),'.3f'))
    return frameDur