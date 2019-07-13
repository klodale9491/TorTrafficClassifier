import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_ffnn_a_performance(datasets_data, score='accuracy'):
    for ds_data in datasets_data:
        dataframe = pd.read_csv(ds_data['path'], ',')
        x_col = dataframe.columns[0]
        x_data = np.asarray(dataframe[x_col])
        if score == 'accuracy':
            y_col = dataframe.columns[2]
        elif score == 'precision':
            y_col = dataframe.columns[3]
        elif score == 'recall':
            y_col = dataframe.columns[4]
        elif score == 'f1':
            y_col = dataframe.columns[5]
        y_data = np.asarray(dataframe[y_col])
        plt.plot(x_data, y_data, ds_data['color'], label=ds_data['label'])
    plt.xlabel('layers')
    plt.ylabel(score)
    plt.legend(loc='lower left')
    plt.show()


# plot tipo 1 dei risultati
datasets = [
    {'path': 'results/ffnn_A/TimeBasedFeatures-15s-TOR-NonTOR.csv',  'label': '15s', 'color': '-y'},
    {'path': 'results/ffnn_A/TimeBasedFeatures-30s-TOR-NonTOR.csv',  'label': '30s', 'color': '-g'},
    {'path': 'results/ffnn_A/TimeBasedFeatures-60s-TOR-NonTOR.csv',  'label': '60s', 'color': '-r'},
    {'path': 'results/ffnn_A/TimeBasedFeatures-120s-TOR-NonTOR.csv', 'label': '120s', 'color': '-b'},
]
plot_ffnn_a_performance(datasets, score='accuracy')
plot_ffnn_a_performance(datasets, score='precision')
plot_ffnn_a_performance(datasets, score='recall')
plot_ffnn_a_performance(datasets, score='f1')
