import os
import pandas as pd
import matplotlib.pyplot as plt


logPath = 'logs/pipeline_ilp'
figuresPath = 'figures/pipeline_ilp/accuracy_vs_demand/'

serversOptions = range(10, 151, 10)

for servers in serversOptions:
    plt.figure()

    logFilename = f'experiment_servers_{servers}.csv'
    df = pd.read_csv(os.path.join(logPath, logFilename))

    demand = df['demand'].values
    accuracy = df['accuracy'].values

    plt.plot(demand, accuracy)

    plt.title(f'Servers: {servers}')
    plt.xlabel('Demand (QPS)')
    plt.ylabel('Accuracy (%)')

    figureFilename = f'servers_{servers}.png'
    figureFilepath = os.path.join(figuresPath, figureFilename)
    plt.savefig(figureFilepath)

    print(f'Saved plot at: {figureFilepath}')
