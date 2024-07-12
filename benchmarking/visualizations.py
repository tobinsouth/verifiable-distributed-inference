import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_accuracy_plot(data_path: str):
    df = pd.read_csv(data_path)

    nodes = [1, 2, 3, 4]
    reference_accuracy = [100, 100, 100, 100]
    goal_accuracy_actual_accuracy = [90, 80, 60, 50]
    goal_resources_actual_accuracy = [85, 75, 55, 45]

    df_resources = df[df['ezkl_optimization_goal'] == 'resources']
    df_accuracy = df[df['ezkl_optimization_goal'] == 'accuracy']

    plt.figure(figsize=(9, 5))
    #plt.scatter(df_resources['num_nodes'], df_resources['reference_accuracy_loss'], marker='o', color='orange', label='Reference')
    plt.scatter(df_resources['num_nodes'], df_resources['accuracy_loss'], color='black', label='Actual (optimized for resources)')
    #plt.scatter(df['num nodes'], goal_resources_actual_accuracy, color='blue', label='Actual (optimized for accuracy)')

    plt.xlabel('No. of nodes/shards')
    plt.ylabel('Cumulative RMSE Loss')
    plt.title('Accuracy Loss')
    plt.legend()

    #plt.yticks(df['accuracy_loss'])
    plt.xticks(nodes)
    plt.grid(False)

    #plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    visualize_accuracy_plot('./results/accuracy_benchmark.csv')
