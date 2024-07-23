import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def visualize_accuracy_old_1(data_path: str):
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


def visualize_accuracy(data_path: str):
    df = pd.read_csv(data_path)
    df = df[df.model != 'attention']
    df['model'] = df['model'].replace({'mlp': 'MLP', 'cnn': 'CNN'})

    plt.style.use('science')

    plt.figure(figsize=(11.69, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x='num_nodes',
        y='accuracy_loss',
        hue='model',
        # style='model',
        s=250
    )

    plt.xlabel('No. of nodes/shards', fontsize=20)
    plt.ylabel('Cumulative RMSE Loss', fontsize=20)
    plt.title('Accuracy Loss', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_nodes'].unique())
    plt.ylim(0, df['accuracy_loss'].max() + 0.00005)

    plt.legend(title='Model', title_fontsize=16, prop={'size': 16})

    plt.tight_layout()
    plt.savefig('./plots/accuracy-plot.pdf', format='pdf')
    # plt.savefig('./plots/accuracy-plot.png', format='png')
    plt.show()


def visualize_accuracy_alt(data_path: str):
    outlier_bound = 0.05

    df = pd.read_csv(data_path)

    no_outlier_df = df[df['accuracy_loss'] <= outlier_bound]

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=no_outlier_df,
        x='num_nodes',
        y='accuracy_loss',
        hue='model',
        style='model',
        s=120,
        ax=ax
    )

    ax.set_xlabel('No. of nodes/shards', fontsize=12)
    ax.set_ylabel('Cumulative RMSE Loss', fontsize=12)
    ax.set_title('Accuracy Loss', fontsize=16)

    ax.set_xticks(no_outlier_df['num_nodes'].unique())
    ax.set_ylim(0, no_outlier_df['accuracy_loss'].max() + 0.001)

    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper left",
                          bbox_to_anchor=(0.06, 0.25, 0.7, 0.7), bbox_transform=ax.transAxes)

    outlier_df = df[df['accuracy_loss'] > outlier_bound]

    hue_order = ax.legend_.get_texts()
    hue_order = [t.get_text() for t in hue_order]

    style_order = ax.legend_.get_lines()
    style_order = [l.get_label() for l in style_order]

    sns.scatterplot(
        data=outlier_df,
        x='num_nodes',
        y='accuracy_loss',
        hue='model',
        style='model',
        s=120,
        ax=ax_inset,
        hue_order=hue_order,
        style_order=style_order,
        legend=False
    )

    ax_inset.set_xlim(outlier_df['num_nodes'].min(), outlier_df['num_nodes'].max())
    ax_inset.set_ylim(outlier_df['accuracy_loss'].max() - 0.011, outlier_df['accuracy_loss'].max() + 0.011)

    ax_inset.set_xticks(outlier_df['num_nodes'].unique())

    ax_inset.set_title('Outlier(s)', fontsize=10)
    ax_inset.set_xlabel('', fontsize=10)
    ax_inset.set_ylabel('', fontsize=10)

    # Annotate the outlier in the main plot

    for idx, row in outlier_df.iterrows():
        ax.text(row['num_nodes'], row['accuracy_loss'], f'{row["accuracy_loss"]:.2e}',
                horizontalalignment='left', size='medium', color='black', weight='semibold')

    # Adjust layout and save the plot
    plt.tight_layout()
    # plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
    plt.savefig('./plots/accuracy_plot.png', format='png')
    plt.show()


if __name__ == '__main__':
    visualize_accuracy('results/final/accuracy_benchmark_all.csv')
