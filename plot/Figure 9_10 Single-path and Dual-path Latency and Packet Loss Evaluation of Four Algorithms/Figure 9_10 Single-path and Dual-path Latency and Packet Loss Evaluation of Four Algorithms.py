import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from matplotlib.font_manager import FontProperties

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams['font.family'] = 'Times New Roman'


def plot_bar_chart(ax, file_path, y_label,
                  colors=None, legend_location='upper right',
                  font_sizes=None, rotate_xticks=45,
                  show_value_labels=True,
                  percentage_columns=None, y_min=None, y_max=None,
                  group_spacing_factor=1.0, total_width=0.4):
    try:
        df = pd.read_csv(file_path, sep='\t')
        logging.info(f"Successfully read file '{file_path}'.")
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return

    if df.empty:
        logging.error(f"File '{file_path}' is empty or incorrectly formatted.")
        return

    is_percentage = '%' in y_label
    if is_percentage and percentage_columns:
        try:
            df[percentage_columns] = df[percentage_columns] * 100
            logging.info(f"Converted columns {percentage_columns} to percentages.")
        except KeyError as e:
            logging.error(f"Percentage columns not found in data: {e}")
            return

    if font_sizes is None:
        font_sizes = {
            'xlabel': 16,
            'ylabel': 16,
            'legend': 14,
            'x_ticks': 14,
            'y_ticks': 14,
            'values': 10
        }

    labels = df['Test'].astype(str).str.extract('(\d+)')[0].astype(str)
    data_columns = [col for col in df.columns if col != 'Test']
    num_columns = len(data_columns)
    x = np.arange(len(labels)) * group_spacing_factor
    width = total_width / num_columns

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if colors is None:
        colors = default_colors[:num_columns]
    elif len(colors) < num_columns:
        logging.warning("Provided colors are fewer than the number of data columns. Using default colors to fill.")
        colors += default_colors[len(colors):num_columns]

    for i, column in enumerate(data_columns):
        offset = (i - num_columns / 2) * width + width / 2
        rects = ax.bar(x + offset, df[column], width, label=column, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.8)
        if show_value_labels:
            autolabel(ax, rects, is_percentage=is_percentage, font_size=font_sizes['values'])

    ax.set_xlabel('Test', fontsize=font_sizes['xlabel'])
    ax.set_ylabel(y_label, fontsize=font_sizes['ylabel'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_xticks,
                       fontproperties=FontProperties(style='normal', weight='normal', family='Times New Roman'))

    ax.tick_params(axis='both', labelsize=font_sizes['x_ticks'], which='major')
    ax.tick_params(axis='y', labelsize=font_sizes['y_ticks'], which='major')

    if y_min is not None or y_max is not None:
        ax.set_ylim([
            y_min if y_min is not None else ax.get_ylim()[0],
            y_max if y_max is not None else ax.get_ylim()[1]
        ])

    ax.set_yscale('linear')

    ax.grid(False)

    ax.legend(loc=legend_location, fontsize=font_sizes['legend'], frameon=False, ncol=2)

    return ax


def autolabel(ax, rects, is_percentage=False, font_size=10):
    for rect in rects:
        height = rect.get_height()
        label = f'{height:.2f}%' if is_percentage else f'{height:.2f}'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=font_size,
                    fontweight='bold',
                    color='black')


def main():
    chart_groups = [
        {
            'charts': [
                {
                    'file_path': 'data_002_single.txt',
                    'y_label': 'Single Path latency (us)',
                    'colors': ['#4eab90', '#edddc3', '#834026', '#eebf6d'],
                    'legend_location': 'upper right',
                    'font_sizes': {
                        'xlabel': 16,
                        'ylabel': 16,
                        'legend': 14,
                        'x_ticks': 14,
                        'y_ticks': 14,
                        'values': 10
                    },
                    'rotate_xticks': 0,
                    'show_value_labels': False,
                    'percentage_columns': None,
                    'y_min': 0,
                    'y_max': 175.0,
                    'group_spacing_factor': 0.6,
                    'total_width': 0.4
                },
                {
                    'file_path': 'data_002_dual.txt',
                    'y_label': 'Dual Path latency (us)',
                    'colors': ['#db3124', '#90bee0', '#ffdf92', '#4b74b2'],
                    'legend_location': 'upper right',
                    'font_sizes': {
                        'xlabel': 16,
                        'ylabel': 16,
                        'legend': 14,
                        'x_ticks': 14,
                        'y_ticks': 14,
                        'values': 10
                    },
                    'rotate_xticks': 0,
                    'show_value_labels': False,
                    'percentage_columns': None,
                    'y_min': 0,
                    'y_max': 155.0,
                    'group_spacing_factor': 0.6,
                    'total_width': 0.4
                }
            ],
            'output_filenames': {
                'svg': 'plots/Latency_Evaluation_of_Four_Algorithms.svg',
                'png': 'plots/Latency_Evaluation_of_Four_Algorithms.png',
                'pdf': 'plots/Latency_Evaluation_of_Four_Algorithms.pdf',
                'eps': 'plots/Latency_Evaluation_of_Four_Algorithms.eps'
            }
        },
        {
            'charts': [
                {
                    'file_path': 'data_003_single.txt',
                    'y_label': 'Single Path Packet Loss (%)',
                    'colors': ['#4eab90', '#edddc3', '#834026', '#eebf6d'],
                    'legend_location': 'upper right',
                    'font_sizes': {
                        'xlabel': 16,
                        'ylabel': 16,
                        'legend': 14,
                        'x_ticks': 14,
                        'y_ticks': 14,
                        'values': 10
                    },
                    'rotate_xticks': 0,
                    'show_value_labels': False,
                    'percentage_columns': ['G-D3QN', 'DQN', 'ACO', 'OSPF'],
                    'y_min': 0.5,
                    'y_max': 1.75,
                    'group_spacing_factor': 0.6,
                    'total_width': 0.4
                },
                {
                    'file_path': 'data_003_dual.txt',
                    'y_label': 'Dual Path Packet Loss (%)',
                    'colors': ['#db3124', '#90bee0', '#ffdf92', '#4b74b2'],
                    'legend_location': 'upper right',
                    'font_sizes': {
                        'xlabel': 16,
                        'ylabel': 16,
                        'legend': 14,
                        'x_ticks': 14,
                        'y_ticks': 13,
                        'values': 10
                    },
                    'rotate_xticks': 0,
                    'show_value_labels': False,
                    'percentage_columns': ['G-D3QN', 'DQN', 'ACO', 'OSPF'],
                    'y_min': 0,
                    'y_max': 0.027,
                    'group_spacing_factor': 0.6,
                    'total_width': 0.4
                }
            ],
            'output_filenames': {
                'svg': 'plots/Packet_Loss_Evaluation_of_Four_Algorithms.svg',
                'png': 'plots/Packet_Loss_Evaluation_of_Four_Algorithms.png',
                'pdf': 'plots/Packet_Loss_Evaluation_of_Four_Algorithms.pdf',
                'eps': 'plots/Packet_Loss_Evaluation_of_Four_Algorithms.eps'
            }
        }
    ]

    for group in chart_groups:
        charts = group['charts']
        output_filenames = group['output_filenames']

        output_dirs = [
            os.path.dirname(output_filenames['svg']),
            os.path.dirname(output_filenames['png']),
            os.path.dirname(output_filenames['pdf']),
            os.path.dirname(output_filenames['eps'])
        ]
        for output_dir in output_dirs:
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created directory '{output_dir}' for saving charts.")

        fig_width = 16
        fig_height = 6
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        for ax, chart in zip(axes, charts):
            plot_bar_chart(
                ax=ax,
                file_path=chart['file_path'],
                y_label=chart['y_label'],
                colors=chart.get('colors'),
                legend_location=chart.get('legend_location', 'upper right'),
                font_sizes=chart.get('font_sizes'),
                rotate_xticks=chart.get('rotate_xticks', 45),
                show_value_labels=chart.get('show_value_labels', True),
                percentage_columns=chart.get('percentage_columns'),
                y_min=chart.get('y_min'),
                y_max=chart.get('y_max'),
                group_spacing_factor=chart.get('group_spacing_factor', 1.0),
                total_width=chart.get('total_width', 0.4)
            )

        plt.tight_layout()

        fig.savefig(output_filenames['svg'], format='svg', facecolor='white')
        fig.savefig(output_filenames['png'], format='png', dpi=600, facecolor='white')
        fig.savefig(output_filenames['pdf'], format='pdf', dpi=600, facecolor='white')
        fig.savefig(output_filenames['eps'], format='eps', facecolor='white')
        logging.info(f"Chart group saved as '{output_filenames['svg']}', '{output_filenames['png']}', '{output_filenames['pdf']}', and '{output_filenames['eps']}'.")

        plt.show(block=False)
        plt.pause(3)
        plt.close(fig)
        logging.info("Chart group window has been automatically closed.")

    print("All images have been displayed and saved, the program continues to run...")


if __name__ == "__main__":
    main()
