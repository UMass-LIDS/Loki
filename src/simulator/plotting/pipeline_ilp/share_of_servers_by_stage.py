import os
import pandas as pd
import matplotlib.pyplot as plt


logPath = 'logs/pipeline_ilp'
figuresPath = 'figures/pipeline_ilp/share_of_servers_by_stage/'

serversOptions = range(10, 151, 10)

for servers in serversOptions:
    plt.figure()

    logFilename = f'experiment_servers_{servers}.csv'
    df = pd.read_csv(os.path.join(logPath, logFilename))

    demand = df['demand'].values
    xVariableValues = df['x_variable'].values

    sharesOfStage1 = []
    sharesOfStage2 = []
    for xVariable in xVariableValues:
        entries = xVariable.split(';')
        
        instancesOfStage1 = 0
        instancesOfStage2 = 0

        for entry in entries:
            variant, instances = entry.split(':')

            if 'yolo' in variant:
                instancesOfStage1 += float(instances)
            elif 'efficientnet' in variant:
                instancesOfStage2 += float(instances)
            else:
                raise Exception(f'Unhandled variant: {variant}')

        shareOfStage1 = instancesOfStage1 / (instancesOfStage1 + instancesOfStage2)
        shareOfStage2 = instancesOfStage2 / (instancesOfStage1 + instancesOfStage2)

        sharesOfStage1.append(shareOfStage1)
        sharesOfStage2.append(shareOfStage2)

    plt.plot(demand, sharesOfStage1, label='stage1', color='green')
    plt.plot(demand, sharesOfStage2, label='stage2', color='blue')

    plt.title(f'Servers: {servers}')
    plt.xlabel('Demand (QPS)')
    plt.ylabel('Share of Cluster')
    plt.legend()

    figureFilename = f'servers_{servers}.png'
    figureFilepath = os.path.join(figuresPath, figureFilename)
    plt.savefig(figureFilepath)

    print(f'Saved plot at: {figureFilepath}')
