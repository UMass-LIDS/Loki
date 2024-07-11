import os
import pandas as pd
import matplotlib.pyplot as plt


logPath = 'logs/pipeline_ilp'
figuresPath = 'figures/pipeline_ilp/share_of_paths/'

serversOptions = range(10, 151, 10)

for servers in serversOptions:
    logFilename = f'experiment_servers_{servers}.csv'
    df = pd.read_csv(os.path.join(logPath, logFilename))

    demand = df['demand'].values
    cVariableValues = df['c_variable'].values

    sharesOfPaths = {
                     ('yolov5n', 'efficientnet-b0'): [],
                    #  ('yolov5s', 'efficientnet-b0'): [],
                     ('yolov5m', 'efficientnet-b0'): [],
                     ('yolov5l', 'efficientnet-b0'): [],
                     ('yolov5x', 'efficientnet-b0'): [],
                     ('yolov5n', 'efficientnet-b1'): [],
                    #  ('yolov5s', 'efficientnet-b1'): [],
                     ('yolov5m', 'efficientnet-b1'): [],
                     ('yolov5l', 'efficientnet-b1'): [],
                     ('yolov5x', 'efficientnet-b1'): [],
                     ('yolov5n', 'efficientnet-b2'): [],
                    #  ('yolov5s', 'efficientnet-b2'): [],
                     ('yolov5m', 'efficientnet-b2'): [],
                     ('yolov5l', 'efficientnet-b2'): [],
                     ('yolov5x', 'efficientnet-b2'): [],
                     ('yolov5n', 'efficientnet-b3'): [],
                    #  ('yolov5s', 'efficientnet-b3'): [],
                     ('yolov5m', 'efficientnet-b3'): [],
                     ('yolov5l', 'efficientnet-b3'): [],
                     ('yolov5x', 'efficientnet-b3'): [],
                     ('yolov5n', 'efficientnet-b4'): [],
                    #  ('yolov5s', 'efficientnet-b4'): [],
                     ('yolov5m', 'efficientnet-b4'): [],
                     ('yolov5l', 'efficientnet-b4'): [],
                     ('yolov5x', 'efficientnet-b4'): [],
                     ('yolov5n', 'efficientnet-b5'): [],
                    #  ('yolov5s', 'efficientnet-b5'): [],
                     ('yolov5m', 'efficientnet-b5'): [],
                     ('yolov5l', 'efficientnet-b5'): [],
                     ('yolov5x', 'efficientnet-b5'): [],
                     ('yolov5n', 'efficientnet-b6'): [],
                    #  ('yolov5s', 'efficientnet-b6'): [],
                     ('yolov5m', 'efficientnet-b6'): [],
                     ('yolov5l', 'efficientnet-b6'): [],
                     ('yolov5x', 'efficientnet-b6'): [],
                     }
    
    for cVariable in cVariableValues:
        entries = cVariable.split(';')

        instancesOfPaths = {
                            ('yolov5n', 'efficientnet-b0'): 0,
                            # ('yolov5s', 'efficientnet-b0'): 0,
                            ('yolov5m', 'efficientnet-b0'): 0,
                            ('yolov5l', 'efficientnet-b0'): 0,
                            ('yolov5x', 'efficientnet-b0'): 0,
                            ('yolov5n', 'efficientnet-b1'): 0,
                            # ('yolov5s', 'efficientnet-b1'): 0,
                            ('yolov5m', 'efficientnet-b1'): 0,
                            ('yolov5l', 'efficientnet-b1'): 0,
                            ('yolov5x', 'efficientnet-b1'): 0,
                            ('yolov5n', 'efficientnet-b2'): 0,
                            # ('yolov5s', 'efficientnet-b2'): 0,
                            ('yolov5m', 'efficientnet-b2'): 0,
                            ('yolov5l', 'efficientnet-b2'): 0,
                            ('yolov5x', 'efficientnet-b2'): 0,
                            ('yolov5n', 'efficientnet-b3'): 0,
                            # ('yolov5s', 'efficientnet-b3'): 0,
                            ('yolov5m', 'efficientnet-b3'): 0,
                            ('yolov5l', 'efficientnet-b3'): 0,
                            ('yolov5x', 'efficientnet-b3'): 0,
                            ('yolov5n', 'efficientnet-b4'): 0,
                            # ('yolov5s', 'efficientnet-b4'): 0,
                            ('yolov5m', 'efficientnet-b4'): 0,
                            ('yolov5l', 'efficientnet-b4'): 0,
                            ('yolov5x', 'efficientnet-b4'): 0,
                            ('yolov5n', 'efficientnet-b5'): 0,
                            # ('yolov5s', 'efficientnet-b5'): 0,
                            ('yolov5m', 'efficientnet-b5'): 0,
                            ('yolov5l', 'efficientnet-b5'): 0,
                            ('yolov5x', 'efficientnet-b5'): 0,
                            ('yolov5n', 'efficientnet-b6'): 0,
                            # ('yolov5s', 'efficientnet-b6'): 0,
                            ('yolov5m', 'efficientnet-b6'): 0,
                            ('yolov5l', 'efficientnet-b6'): 0,
                            ('yolov5x', 'efficientnet-b6'): 0
                            }

        for entry in entries:
            path = ()

            pathVariants, instances = entry.split(':')
            
            splitVariants = pathVariants.split('|')
            for variant in splitVariants:
                path = path + (variant,)

            # print(f'path: {path}')

            instancesOfPaths[path] += float(instances)
        
        for path in instancesOfPaths:
            sharesOfPaths[path].append(instancesOfPaths[path])


    plt.figure(figsize=(15, 7))
    for path in sharesOfPaths:
        if 'efficientnet-b6' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='x')
        elif 'efficientnet-b5' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='o')
        elif 'efficientnet-b4' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='.')
        elif 'efficientnet-b3' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='^')
        elif 'efficientnet-b2' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='+')
        elif 'efficientnet-b1' in path:
            plt.plot(demand, sharesOfPaths[path], label=path, marker='|')
        else:
            plt.plot(demand, sharesOfPaths[path], label=path)

    # Set a colormap for the lines
    plt.gca().set_prop_cycle(None)

    plt.title(f'Servers: {servers}')
    plt.xlabel('Demand (QPS)')
    plt.ylabel('Shares of Path')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.tight_layout()
    figureFilename = f'servers_{servers}.png'
    figureFilepath = os.path.join(figuresPath, figureFilename)
    plt.savefig(figureFilepath)
    print(f'Saved plot at: {figureFilepath}')

    # print(f'sharesOfPath: {sharesOfPaths}')

