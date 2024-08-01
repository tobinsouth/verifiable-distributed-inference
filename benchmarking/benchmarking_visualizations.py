import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scienceplots
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter


def custom_byte_formatter(x, pos):
    if x == 0:
        return '0 B'
    elif x == 1:
        return '1 B'
    elif x == 10 ** 1:
        return '10 B'
    elif x == 10 ** 2:
        return '100 B'
    elif x == 10 ** 3:
        return '1 KB'
    elif x == 10 ** 4:
        return '10 KB'
    elif x == 10 ** 5:
        return '100 KB'
    elif x == 10 ** 6:
        return '1 MB'
    elif x == 10 ** 7:
        return '10 MB'
    elif x == 10 ** 8:
        return '100 MB'
    elif x == 10 ** 9:
        return '1 GB'
    elif x == 10 ** 10:
        return '10 GB'
    elif x == 10 ** 11:
        return '100 GB'
    elif x == 10 ** 12:
        return '1 TB'
    else:
        return ''


def custom_time_formatter(x, pos):
    if x == 0:
        return '0 s'
    if 0 <= x <= 60:
        return f'{x:.0f} s'
    elif 60 < x <= 60*60:
        return f'{(x/60):.1f} m'
    elif 60*60 < x <= 60*60*24:
        return f'{(x/(60*60)):.1f} h'
    else:
        return ''



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
    # plt.scatter(df['num nodes'], goal_resources_actual_accuracy, color='blue', label='Actual (optimized for accuracy)')

    plt.xlabel('No. of nodes/shards')
    plt.ylabel('Cumulative RMSE Loss')
    # plt.title('Accuracy Loss')
    plt.legend()

    # plt.yticks(df['accuracy_loss'])
    plt.xticks(df_resources['num_nodes'])
    plt.grid(False)

    # plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
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


def visualize_accuracy(data_path: str, save_pdf: bool = False):
    df = pd.read_csv(data_path)
    df = df[df.model != 'testing']
    df = df[df.model != 'attention']
    df['model'] = df['model'].replace({
        'mlp': 'MLP',
        'cnn': 'CNN',
        'mlp2': 'MLP2'
    })

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
    plt.ylabel('Cumulative RMSE loss', fontsize=20)
    # plt.title('Accuracy Loss', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_nodes'].unique())
    #plt.ylim(bottom=0)
    plt.yscale('log')

    plt.legend(title='Model',
               title_fontsize=16,
               prop={'size': 16},
               loc='upper left',
               bbox_to_anchor=(1, 1))

    plt.tight_layout()
    if save_pdf:
        plt.savefig('./plots/accuracy-plot.pdf', format='pdf')
    # plt.savefig('./plots/accuracy-plot.png', format='png')
    plt.show()


def visualize_proving_and_setup_times(data_path_proving: str, data_path_setup: str, save_pdf: bool = False):
    df = pd.read_csv(data_path_proving)
    df2 = pd.read_csv(data_path_setup)
    df = pd.merge(df, df2, on=['model_id', 'num_shards'])
    df = df[df['model_id'] != 'testing']
    df['model_id'] = df['model_id'].replace({
        'mlp': 'MLP',
        'cnn': 'CNN',
        'testing': 'Testing',
        'mlp2': 'MLP2'
    })

    df = pd.melt(df,
                 id_vars=["model_id", "num_shards"],
                 value_vars=["total_proof_generation_time", "total_setup_time"],
                 var_name="Step",
                 value_name="time"
                 )
    df['Step'] = df['Step'].replace({
        'total_proof_generation_time': 'Prove',
        'total_setup_time': 'Setup'
    })

    df = df.rename(columns={"model_id": "Model"})

    plt.style.use('science')

    plt.figure(figsize=(11.69, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x='num_shards',
        y='time',
        hue='Model',
        style='Step',
        s=250
    )

    plt.xlabel('No. of nodes/shards', fontsize=20)
    plt.ylabel('Cumulative proving time', fontsize=20)
    # plt.title('Setup and Proving Time', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_shards'].unique())

    plt.legend(title_fontsize=16,
               prop={'size': 16},
               loc='upper left',
               bbox_to_anchor=(1, 1))

    # plt.ylim(bottom=0)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_time_formatter))

    plt.tight_layout()
    if save_pdf:
        plt.savefig('./plots/proving-time-plot.pdf', format='pdf')
    plt.show()


def visualize_witness_times(data_path: str, save_pdf: bool = False):
    df = pd.read_csv(data_path)
    df = df[df['model_id'] != 'testing']
    df['model_id'] = df['model_id'].replace({
        'mlp': 'MLP',
        'cnn': 'CNN',
        'mlp2': 'MLP2'
    })

    plt.style.use('science')

    plt.figure(figsize=(11.69, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x='num_shards',
        y='total_witness_generation_time',
        hue='model_id',
        # style='model',
        s=250
    )

    plt.xlabel('No. of nodes/shards', fontsize=20)
    plt.ylabel('Cumulative witness generation time', fontsize=20)
    # plt.title('Added Overhead', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_shards'].unique())

    plt.legend(title='Model',
               title_fontsize=16,
               prop={'size': 16},
               loc='upper left',
               bbox_to_anchor=(1, 1))

    plt.yscale('log')
    plt.ylim(top=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_time_formatter))

    plt.tight_layout()
    if save_pdf:
        plt.savefig('./plots/witness-time-plot.pdf', format='pdf')
    plt.show()


def visualize_vk_and_pk_sizes(data_path: str, save_pdf: bool = False):
    df = pd.read_csv(data_path)
    df = df[df['model_id'] != 'testing']
    df['model_id'] = df['model_id'].replace({
        'mlp': 'MLP',
        'cnn': 'CNN',
        'mlp2': 'MLP2'
    })
    df = pd.melt(df,
                 id_vars=["model_id", "num_shards"],
                 value_vars=["total_vk_size", "total_pk_size"],
                 var_name="key_size_type", value_name="size"
    )
    df['key_size_type'] = df['key_size_type'].replace({
        'total_vk_size': 'vk',
        'total_pk_size': 'pk'
    })

    df = df.rename(columns={
        "model_id": "Model",
        "key_size_type": "Key"}
    )

    plt.style.use('science')

    plt.figure(figsize=(11.69, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x='num_shards',
        y='size',
        hue='Model',
        style='Key',
        s=250
    )

    plt.xlabel('No. of nodes/shards', fontsize=20)
    plt.ylabel('Cumulative artifact size', fontsize=20)
    # plt.title('Verification and Proving Key Sizes', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_shards'].unique())

    plt.yscale('log')
    plt.ylim(top=10**12, bottom=10**4)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_byte_formatter))

    plt.legend(title_fontsize=16,
               prop={'size': 16},
               bbox_to_anchor=(1, 1))


    plt.tight_layout()
    if save_pdf:
        plt.savefig('./plots/key-size-plot.pdf', format='pdf')
    plt.show()


def visualize_proof_and_witness_sizes(data_path: str, save_pdf: bool = False):
    df = pd.read_csv(data_path)
    df = df[df['model_id'] != 'testing']
    df['model_id'] = df['model_id'].replace({
        'mlp': 'MLP',
        'cnn': 'CNN',
        'mlp2': 'MLP2'
    })
    df = pd.melt(df,
                 id_vars=["model_id", "num_shards"],
                 value_vars=["total_proof_size", "total_witness_size"],
                 var_name="artifact_type",
                 value_name="size"
    )
    df['artifact_type'] = df['artifact_type'].replace({
        'total_proof_size': 'Proof',
        'total_witness_size': 'Witness'
    })

    df = df.rename(columns={
        "model_id": "Model",
        "artifact_type": "Artifact"}
    )

    plt.style.use('science')

    formatter = FuncFormatter(lambda x, pos: '%1.1fK' % (x * 1e-3))

    plt.figure(figsize=(11.69, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x='num_shards',
        y='size',
        hue='Model',
        style='Artifact',
        s=250
    )

    plt.xlabel('No. of nodes/shards', fontsize=20)
    plt.ylabel('Cumulative artifact size', fontsize=20)
    # plt.title('Proof and Witness Sizes', fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(df['num_shards'].unique())

    plt.yscale('log')
    plt.ylim(bottom=10**3, top=10**8)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_byte_formatter))

    plt.legend(title_fontsize=16,
               prop={'size': 16},
               loc='upper left',
               bbox_to_anchor=(1, 1))




    plt.tight_layout()
    if save_pdf:
        plt.savefig('./plots/proof-witness-size-plot.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    visualize_accuracy('results/accuracy_benchmark_all.csv', True)
    visualize_proving_and_setup_times('results/cumulative_proving_time.csv',
                                      'results/cumulative_setup_time.csv', True)
    visualize_witness_times('results/cumulative_witness_time.csv', True)
    visualize_vk_and_pk_sizes('results/file_sizes.csv', True)
    visualize_proof_and_witness_sizes('results/file_sizes.csv', True)

