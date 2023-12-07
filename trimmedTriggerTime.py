def trimmedTriggerTime(trigTime,caTime,start,stop):
    # trimming trigger time based on ca2+ time. start and stop indicate the period of ca2+ data
    trimmed_trigTime = [i for i in trigTime if i >= caTime[start] and i<= caTime[stop]]
    return trimmed_trigTime