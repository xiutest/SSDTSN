import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from matplotlib.font_manager import FontProperties

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams['font.family'] = 'Times New Roman'

def plot_bar_chart(file_path, title, y_label, colors=None, legend_location='upper right',
                  font_sizes=None, rotate_xticks=45,
                  show_value_labels=True, display_time=3,
                  y_min=None, y_max=None,
                  group_spacing_factor=1.0, total_width=0.4,
                  ax=None):

    try:
        df = pd.read_csv(file_path, sep='\t')
        logging.info(f"Successfully read file '{file_path}'.")
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return

    if df.empty:
        logging.error(f"File '{file_path}' is empty or has incorrect format.")
        return

    if font_sizes is None:
        font_sizes = {
            'title': 20,
            'xlabel': 16,
            'ylabel': 16,
            'legend': 14,
            'x_ticks': 18,
            'y_ticks': 14,
            'values': 10
        }

    labels = df['Test'].astype(str).str.extract('(\d+)')[0].astype(str)
    data_columns = [col for col in df.columns if col != 'Test']
    num_columns = len(data_columns)
    x = np.arange(len(labels)) * group_spacing_factor
    width = total_width / num_columns

    own_fig = False
    if ax is None:
        fig_width = 8
        fig_height = 6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        own_fig = True

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if colors is None:
        colors = default_colors[:num_columns]
    elif len(colors) < num_columns:
        logging.warning("Provided colors less than number of data columns. Using default colors for the rest.")
        colors += default_colors[len(colors):num_columns]

    for i, column in enumerate(data_columns):
        offset = (i - num_columns / 2) * width + width / 2
        rects = ax.bar(x + offset, df[column], width, label=column, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.8)
        if show_value_labels:
            autolabel(ax, rects, is_percentage=False, font_size=font_sizes['values'])

    if title:
        ax.set_title(title, fontsize=font_sizes['title'], pad=15)

    ax.set_xlabel('Test', fontsize=font_sizes['xlabel'])
    ax.set_ylabel(y_label, fontsize=font_sizes['ylabel'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_xticks,
                       fontproperties=FontProperties(style='normal', weight='normal', family='Times New Roman'))

    ax.tick_params(axis='x', labelsize=font_sizes['x_ticks'], which='major')
    ax.tick_params(axis='y', labelsize=font_sizes['y_ticks'], which='major')

    if y_min is not None or y_max is not None:
        ax.set_ylim([
            y_min if y_min is not None else ax.get_ylim()[0],
            y_max if y_max is not None else ax.get_ylim()[1]
        ])

    ax.set_yscale('linear')
    ax.grid(False)
    ax.legend(loc=legend_location, fontsize=font_sizes['legend'], frameon=False, ncol=2)
    plt.tight_layout()

    if own_fig:
        svg_filename = f"{os.path.splitext(file_path)[0]}.svg"
        png_filename = f"{os.path.splitext(file_path)[0]}.png"
        pdf_filename = f"{os.path.splitext(file_path)[0]}.pdf"
        fig.savefig(svg_filename, format='svg', facecolor='white')
        fig.savefig(png_filename, format='png', dpi=600, facecolor='white')
        fig.savefig(pdf_filename, format='pdf', dpi=600, facecolor='white')
        logging.info(f"Chart saved as '{svg_filename}', '{png_filename}' and '{pdf_filename}'.")

        plt.show(block=False)
        plt.pause(display_time)
        plt.close(fig)
        logging.info("Chart window closed automatically.")


def autolabel(ax, rects, is_percentage=False, font_size=10):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    """
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
    bar_charts = [
        {
            'file_path': 'data_single.txt',
            'title': None,
            'y_label': 'Single Path Load CV',
            'colors': ['#4eab90', '#edddc3', '#834026', '#eebf6d'],
            'legend_location': 'upper right',
            'font_sizes': {
                'title': 18,
                'xlabel': 16,
                'ylabel': 16,
                'legend': 14,
                'x_ticks': 14,
                'y_ticks': 14,
                'values': 10
            },
            'rotate_xticks': 0,
            'show_value_labels': False,
            'display_time': 3,
            'y_min': 0,
            'y_max': 1.05,
            'group_spacing_factor': 0.6,
            'total_width': 0.4
        },
        {
            'file_path': 'data_dual.txt',
            'title': None,
            'y_label': 'Dual Path Load CV',
            'colors': ['#db3124', '#90bee0', '#ffdf92', '#4b74b2'],
            'legend_location': 'upper right',
            'font_sizes': {
                'title': 18,
                'xlabel': 16,
                'ylabel': 16,
                'legend': 14,
                'x_ticks': 14,
                'y_ticks': 13,
                'values': 10
            },
            'rotate_xticks': 0,
            'show_value_labels': False,
            'display_time': 3,
            'y_min': 0,
            'y_max': 1.05,
            'group_spacing_factor': 0.6,
            'total_width': 0.4
        }
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    logging.info("Created a figure with two subplots.")

    for i, bar_chart in enumerate(bar_charts):
        plot_bar_chart(
            file_path=bar_chart['file_path'],
            title=bar_chart['title'],
            y_label=bar_chart['y_label'],
            colors=bar_chart.get('colors'),
            legend_location=bar_chart.get('legend_location', 'upper right'),
            font_sizes=bar_chart.get('font_sizes'),
            rotate_xticks=bar_chart.get('rotate_xticks', 45),
            show_value_labels=bar_chart.get('show_value_labels', True),
            display_time=bar_chart.get('display_time', 3),
            y_min=bar_chart.get('y_min'),
            y_max=bar_chart.get('y_max'),
            group_spacing_factor=bar_chart.get('group_spacing_factor', 1.0),
            total_width=bar_chart.get('total_width', 0.4),
            ax=axes[i]
        )

    plt.tight_layout()

    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory '{output_dir}' for saving the combined plot.")

    combined_eps_filename = os.path.join(output_dir, 'combined_plots.eps')

    fig.savefig(combined_eps_filename, format='eps', dpi=1200, facecolor='white')
    logging.info(f"Combined chart saved as '{combined_eps_filename}'.")

    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)
    logging.info("Combined chart window closed automatically.")

if __name__ == "__main__":
    main()
