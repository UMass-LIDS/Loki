class Query:
    def __init__(self, requestID, queryID, userID, applicationID, taskID, data,
                 startTimestamp, queuedTimeStamp, latencyBudget,
                 sequenceNum):
        self.requestID = requestID
        self.queryID = queryID
        self.userID = userID
        self.applicationID = applicationID
        self.taskID = taskID
        self.data = data
        self.startTimestamp = startTimestamp
        self.queuedTimestamp = queuedTimeStamp
        self.latencyBudget = latencyBudget
        self.sequenceNum = sequenceNum
        return
