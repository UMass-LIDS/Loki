import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


approaches = ['proteus', 'inferline', 'loki']
approachToLabel = {'proteus': 'Proteus', 'inferline': 'InferLine', 'loki': 'Loki'}

preprocessedDir = 'logs/preprocessed/social_media'
figuresDir = 'figures/social_media'
# Create a figure and three subplots in a vertical arrangement
# fig, axs = plt.subplots(nrows=4, figsize=(8, 12))
fig, axs = plt.subplots(nrows=4, figsize=(6,6))

demand_df = pd.read_csv(os.path.join(preprocessedDir,
                                        'requestsingested_loki.csv'))
time = demand_df['time']
demand = demand_df['requests_ingested']
axs[0].plot(time, demand, color='gray')
axs[0].set_xticklabels([])
axs[0].set_yticks(np.arange(0, 1251, 250))
axs[0].set_ylabel('Demand (QPS)')

for approach in approaches:

    accuracy_df = pd.read_csv(os.path.join(preprocessedDir,
                                           f'accuracy_{approach}.csv'))
    time = accuracy_df['time']
    accuracy = accuracy_df['accuracy']
    axs[1].plot(time, accuracy, label=approachToLabel[approach])
    axs[1].set_xticklabels([])
    axs[1].set_yticks(np.arange(60, 101, 10))
    axs[1].set_ylabel('System\nAccuracy (%)')

    cluster_df = pd.read_csv(os.path.join(preprocessedDir,
                                          f'clusterusage_{approach}.csv'))
    time = cluster_df['time']
    cluster_usage = cluster_df['cluster_usage']
    axs[2].plot(time, cluster_usage, label=approachToLabel[approach])
    axs[2].set_xticklabels([])
    axs[2].set_yticks(np.arange(0, 101, 25))
    axs[2].set_ylabel('Cluster\nUtilization (%)')

    slo_violations_df = pd.read_csv(os.path.join(preprocessedDir,
                                                 f'droppedsubrequestsratio_{approach}.csv'))
    time = slo_violations_df['time']
    slo_violations_usage = slo_violations_df['dropped_subrequests_ratio']
    axs[3].plot(time, slo_violations_usage, label=approachToLabel[approach])
    axs[3].set_yticks(np.arange(0, 0.51, 0.1))
    axs[3].set_ylabel('SLO Violation\nRatio')

    axs[3].set_xlabel('Time (seconds)')

# Customize layout and add legends if needed
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 5.0))

# axs[0].legend()
axs[0].grid()
# axs[1].legend()
axs[1].grid()
# axs[2].legend()
axs[2].grid()
axs[3].grid()
plt.tight_layout()

# # Show the plot
# plt.show()

# Save the plot
plt.savefig(os.path.join(figuresDir, 'end_to_end.pdf'))
