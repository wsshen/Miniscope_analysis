import csv

def readTabTxtFile(fileName,str):
    with open(fileName, "r", newline="") as readFile:
        file_reader = csv.reader(readFile, delimiter='\t')
        for entry in file_reader:
            stimData = entry
    try:
        stimData
    except:
        print(fileName+" is empty")
        return ""
    if str == 'string':
        cleanStimData = [i for i in stimData if i]
        return cleanStimData
    if str == 'float':
        cleanStimData = [float(i) for i in stimData if i]
        return cleanStimData
    if str == 'int':
        cleanStimData = [int(i) for i in stimData if i]
        return cleanStimData
    if str == 'int_float':
        cleanStimData = [int(float(i)) for i in stimData if i]
        return cleanStimData
