import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_cdf(data, xlabel="Data Points", ylabel="Cumulative Probability",
             title="CDF Plot", filename='plot.pdf'):
    
    fig = plt.figure()
    for model_family in data:
        sorted_data = np.sort(data[model_family])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.plot(sorted_data, yvals, marker='.', linestyle='None', label=model_family)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'figures/request_times/{filename}')

# Example Usage:
if __name__ == "__main__":
    # Replace this with your list of numbers
    data_to_plot = []
    df = pd.read_csv('logs/request_times.csv')

    model_families = df['model_family'].unique()

    waiting_times = {}
    execution_times = {}
    sojourn_times = {}

    for model_family in model_families:
        subset_df = df[df['model_family'] == model_family]

        waiting_times[model_family] = subset_df['waiting_time'].values
        execution_times[model_family] = subset_df['execution_time'].values
        sojourn_times[model_family] = waiting_times[model_family] + execution_times[model_family]
        

    plot_cdf(waiting_times, xlabel='Waiting Times (milliseconds)',
             title='Cumulative Distribution Function (CDF)',
             filename='waiting_times.pdf')
    
    plot_cdf(execution_times, xlabel='Execution Times (milliseconds)',
             title='Cumulative Distribution Function (CDF)',
             filename='execution_times.pdf')

    plot_cdf(sojourn_times, xlabel='Sojourn Times (milliseconds)',
             title='Cumulative Distribution Function (CDF)',
             filename='sojourn_times.pdf')
