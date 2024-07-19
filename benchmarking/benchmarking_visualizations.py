import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_accuracy_plot(data_path: str):
    df = pd.read_csv(data_path)

    df_resources = df[df['ezkl_optimization_goal'] == 'resources']
    df_accuracy = df[df['ezkl_optimization_goal'] == 'accuracy']

    plt.figure(figsize=(9, 5))
    plt.plot(
        df_resources['num_nodes'],
        df_resources['reference_accuracy_loss'],
        marker='o',
        color='orange',
        label='Optimal'
    )
    plt.scatter(
        df_resources['num_nodes'],
        df_resources['accuracy_loss'],
        color='black',
        label='Actual (optimized for resources)'
    )
    #plt.scatter(df['num nodes'], goal_resources_actual_accuracy, color='blue', label='Actual (optimized for accuracy)')

    plt.xlabel('No. of nodes/shards')
    plt.ylabel('Cumulative RMSE Loss')
    plt.title('Accuracy Loss')
    plt.legend()

    #plt.yticks(df['accuracy_loss'])
    plt.xticks(df_resources['num_nodes'])
    plt.grid(False)

    #plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
    plt.show()


def visualize_accuracy_plot_sns(data_path: str):
    df = pd.read_csv(data_path)

    sns.scatterplot(
        data=df,
        x='num_nodes',
        y='accuracy_loss',
        hue='model',
        style='model',
        s=120
    )
    # sns.lineplot(
    #     data=df,
    #     x='num_nodes',
    #     y='reference_accuracy_loss',
    #     c='grey'
    # )

    plt.xlabel('No. of nodes/shards', fontsize=10)
    plt.ylabel('Cumulative RMSE Loss', fontsize=10)
    plt.title('Accuracy Loss', fontsize=12)

    plt.xticks(df['num_nodes'].unique())
    plt.ylim(0, df['accuracy_loss'].max() + 0.00005)

    plt.tight_layout()
    # plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
    # plt.savefig('./plots/accuracy_plot.png', format='png')
    plt.show()


if __name__ == '__main__':
    # visualize_accuracy_plot('./results/accuracy_benchmark.csv')
    visualize_accuracy_plot_sns('results/final/accuracy_benchmark_all')
