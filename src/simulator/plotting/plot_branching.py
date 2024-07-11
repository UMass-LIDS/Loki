import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


END_TIME = 108000

# path = ('../../datasets/infaas/asplos/zipf_exponential/trace_files/')
# path = ('../traces/twitter/asplos/zipf_flat_bursty/300/')
# path = ('../traces/twitter/asplos/low_load/300/')
# path = ('../traces/twitter/asplos/zipf_exponential_bursty/300/')
# path = ('../traces/twitter/asplos/medium-normal_load/300/')
# path = 'traces/twitter/traffic_analysis_test/'
# path = 'traces/maf/shape/trace/'
path = 'traces/maf/traffic_analysis/branching/'
# path = 'traces/maf/traffic_analysis/branching_smoothed/'


logfile_list = glob.glob(os.path.join(path, '*.csv'))

# # We want the latest log file
# logfile = logfile_list[-1]
print(logfile_list)

markers = ['o', 'v', '^', '*', 's']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']

total_requests = np.zeros(END_TIME)

for logfile in logfile_list:
    df = pd.read_csv(logfile)
    
    # print(f'df: {df}')
    # aggregated = df.groupby(df.frame // 1000).count()
    # print(f'aggregated: {aggregated}')
    # df = aggregated

    window_size = 5000

    # smoothed_car_series = df['car_classification'].rolling(window=window_size).mean()
    # smoothed_car_array = smoothed_car_series.dropna().values
    # smoothed_car_array = np.ceil(smoothed_car_array)
    plt.plot(df['frame'].values, df['car_classification'].values,
             label='car_classification')
    # plt.plot(df['frame'].values[:len(smoothed_car_array)], smoothed_car_array,
    #          label='car_classification')
    # print(smoothed_car_array)
    
    # smoothed_face_series = df['facial_recognition'].rolling(window=window_size).mean()
    # smoothed_face_array = smoothed_face_series.dropna().values
    # smoothed_face_array = np.ceil(smoothed_face_array)
    plt.plot(df['frame'].values, df['facial_recognition'].values,
             label='facial_recognition')
    # plt.plot(df['frame'].values[:len(smoothed_face_array)], smoothed_face_array,
    #          label='facial_recognition')
    # print(smoothed_face_array)
    
    figName = f'{logfile.split("/")[-1].split(".")[0]}.png'
    outName = os.path.join(path, figName)
    plt.legend()
    # max_y = max(max(smoothed_face_array), max(smoothed_car_array))
    # plt.ylim([0, max_y+2])
    plt.savefig(outName)
    plt.close()
    print(f'Figure saved at {outName}')

# for idx in range(len(logfile_list)):
#     logfile = logfile_list[idx]
    
#     model_family = logfile.split('/')[-1].split('.')[0]

#     df = pd.read_csv(logfile, names=['time', 'accuracy', 'deadline'])

#     print(f'df: {df}')
#     aggregated = df.groupby(df.time // 1000).count()
#     print(f'aggregated: {aggregated}')
#     df = aggregated
    
#     requests = df.time.to_list()
#     if len(requests) < END_TIME:
#         for i in range(END_TIME-len(requests)):
#             requests.append(0)
#     total_requests += requests
#     # print(f'df.time: {len(df.time.to_list())}')

#     start_cutoff = 0

#     time = df['time'].values[start_cutoff:]    

#     # time = time
#     # time = [x - time[0] for x in time]
#     # print(time[-1])
#     # time = [x / time[-1] * 24 for x in time]
#     # print(time[0])
#     # print(time[-1])

#     plt.plot(time, label=model_family)
#     # plt.plot(time, demand, label='Demand')
#     # plt.plot(time, throughput, label=algorithm, marker=markers[idx])
#     # plt.plot(time, throughput, label=algorithm)
#     # plt.plot(time, throughput, color=colors[idx])
# # plt.plot(time, capacity, label='capacity')
# plt.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

# y_cutoff = max(demand) + 50
# # y_cutoff = 200

# print(f'total requests: {total_requests}')

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=10)
# plt.xlabel('Time (min)', fontsize=25)
# plt.ylabel('Requests per second', fontsize=25)
# # plt.xticks(np.arange(0, 25, 4), fontsize=15)
# # plt.yticks(np.arange(0, y_cutoff, 200), fontsize=15)
# plt.savefig(os.path.join(path, 'traces_by_model.pdf'), dpi=500, bbox_inches='tight')

# plt.close()
# plt.plot(total_requests)
# plt.savefig(os.path.join(path, 'traces_total.pdf'), dpi=500, bbox_inches='tight')
