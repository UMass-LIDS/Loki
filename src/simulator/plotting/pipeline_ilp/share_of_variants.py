import os
import pandas as pd
import matplotlib.pyplot as plt


logPath = 'logs/pipeline_ilp'
figuresPath = 'figures/pipeline_ilp/share_of_variants/'

serversOptions = range(10, 151, 10)

for servers in serversOptions:
    logFilename = f'experiment_servers_{servers}.csv'
    df = pd.read_csv(os.path.join(logPath, logFilename))

    demand = df['demand'].values
    xVariableValues = df['x_variable'].values

    instancesOfVariants1 = {'yolov5n': [], 'yolov5m': [],
                            'yolov5l': [], 'yolov5x': []
                            }
    instancesOfVariants2 = {'efficientnet-b0': [], 'efficientnet-b1': [],
                            'efficientnet-b2': [], 'efficientnet-b3': [],
                            'efficientnet-b4': [], 'efficientnet-b5': [],
                            'efficientnet-b6': []
                            }
    for xVariable in xVariableValues:
        entries = xVariable.split(';')
        
        variants1 = {'yolov5n': 0, 'yolov5m': 0, 'yolov5l': 0,
                     'yolov5x': 0}
        variants2 = {'efficientnet-b0': 0, 'efficientnet-b1': 0,
                     'efficientnet-b2': 0, 'efficientnet-b3': 0,
                     'efficientnet-b4': 0, 'efficientnet-b5': 0,
                     'efficientnet-b6': 0}

        for entry in entries:
            variant, instances = entry.split(':')

            if 'yolo' in variant:
                variants1[variant] += float(instances)
            elif 'efficientnet' in variant:
                variants2[variant] += float(instances)
            else:
                raise Exception(f'Unhandled variant: {variant}')

        for variant in variants1:
            instancesOfVariants1[variant].append(variants1[variant])
        for variant in variants2:
            instancesOfVariants2[variant].append(variants2[variant])

    # YOLOv5
    plt.figure()
    for variant in instancesOfVariants1:
        plt.plot(demand, instancesOfVariants1[variant], label=variant)
    plt.title(f'Servers: {servers}')
    plt.xlabel('Demand (QPS)')
    plt.ylabel('Instances of Variant')
    plt.ylim([0, servers])
    plt.legend()
    figureFilename = f'yolov5_servers_{servers}.png'
    figureFilepath = os.path.join(figuresPath, figureFilename)
    plt.savefig(figureFilepath)
    print(f'Saved plot at: {figureFilepath}')

    # EfficientNet
    plt.figure()
    for variant in instancesOfVariants2:
        plt.plot(demand, instancesOfVariants2[variant], label=variant)
    plt.title(f'Servers: {servers}')
    plt.xlabel('Demand (QPS)')
    plt.ylabel('Instances of Variant')
    plt.ylim([0, servers])
    plt.legend()
    figureFilename = f'efficientnet_servers_{servers}.png'
    figureFilepath = os.path.join(figuresPath, figureFilename)
    plt.savefig(figureFilepath)
    print(f'Saved plot at: {figureFilepath}')

