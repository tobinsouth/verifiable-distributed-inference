import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_accuracy_plot():
    nodes = [1, 2, 3, 4]
    reference_accuracy = [100, 100, 100, 100]
    goal_accuracy_actual_accuracy = [90, 80, 60, 50]
    goal_resources_actual_accuracy = [85, 75, 55, 45]

    plt.figure(figsize=(7, 5))
    plt.scatter(nodes, reference_accuracy, marker='o', color='orange', label='Reference')
    plt.scatter(nodes, goal_accuracy_actual_accuracy, color='black', label='Actual (optimized for resources)')
    plt.scatter(nodes, goal_resources_actual_accuracy, color='blue', label='Actual (optimized for accuracy)')

    plt.xlabel('No. of nodes/shards')
    plt.ylabel('Accuracy in %')
    plt.title('Accuracy Loss')
    plt.legend()


    plt.yticks(list(range(0, 101, 10)))
    plt.xticks(nodes)
    plt.grid(False)

    plt.savefig('./plots/accuracy_plot.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    visualize_accuracy_plot()
