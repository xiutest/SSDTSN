import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from matplotlib.font_manager import FontProperties

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams['font.family'] = 'Times New Roman'


def plot_bar_chart(file_path, title, y_label, svg_filename, png_filename, pdf_filename,
                  colors=None, legend_location='upper right',
                  font_sizes=None, rotate_xticks=45,
                  show_value_labels=True, display_time=3,
                  percentage_columns=None, y_min=None, y_max=None,
                  group_spacing_factor=1.0, total_width=0.4,
                  y_scale='linear', ax=None):

    try:
        df = pd.read_csv(file_path, sep='\t')
        logging.info(f"Successfully read file '{file_path}'.")
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return

    if df.empty:
        logging.error(f"File '{file_path}' is empty or has incorrect format.")
        return

    is_percentage = '%' in y_label
    if is_percentage and percentage_columns:
        try:
            df[percentage_columns] = df[percentage_columns] * 100
            logging.info(f"Converted columns {percentage_columns} to percentages.")
        except KeyError as e:
            logging.error(f"Percentage column not found in data: {e}")
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

    labels = df['Test'].astype(str)
    data_columns = [col for col in df.columns if col != 'Test']
    num_columns = len(data_columns)
    x = np.arange(len(labels)) * group_spacing_factor
    width = total_width / num_columns

    output_dirs = []
    if svg_filename:
        output_dirs.append(os.path.dirname(svg_filename))
    if png_filename:
        output_dirs.append(os.path.dirname(png_filename))
    if pdf_filename:
        output_dirs.append(os.path.dirname(pdf_filename))
    for output_dir in output_dirs:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory '{output_dir}' for saving plots.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    if colors is None:
        colors = default_colors[:num_columns]
    elif len(colors) < num_columns:
        logging.warning("Provided colors are fewer than the number of data columns. Using default colors for the rest.")
        colors += default_colors[len(colors):num_columns]

    for i, column in enumerate(data_columns):
        offset = (i - num_columns / 2) * width + width / 2
        rects = ax.bar(x + offset, df[column], width, label=column, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.8)
        if show_value_labels:
            autolabel(ax, rects, is_percentage=is_percentage, font_size=font_sizes['values'])

    if title:
        ax.set_title(title, fontsize=font_sizes['title'], pad=15)

    ax.set_xlabel('Test', fontsize=font_sizes['xlabel'])
    ax.set_ylabel(y_label, fontsize=font_sizes['ylabel'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_xticks,
                       fontproperties=FontProperties(style='normal', weight='normal', family='Times New Roman'))

    ax.tick_params(axis='x', labelsize=font_sizes['x_ticks'])
    ax.tick_params(axis='y', labelsize=font_sizes['y_ticks'])

    if y_min is not None or y_max is not None:
        ax.set_ylim([
            y_min if y_min is not None else ax.get_ylim()[0],
            y_max if y_max is not None else ax.get_ylim()[1]
        ])

    ax.set_yscale(y_scale)

    ax.grid(False)

    ax.legend(loc=legend_location, fontsize=font_sizes['legend'], frameon=False, ncol=1)

    if created_fig:
        plt.tight_layout()

        if svg_filename or png_filename or pdf_filename:
            if svg_filename:
                fig.savefig(svg_filename, format='svg', facecolor='white')
            if png_filename:
                fig.savefig(png_filename, format='png', dpi=600, facecolor='white')
            if pdf_filename:
                fig.savefig(pdf_filename, format='pdf', dpi=600, facecolor='white')
            logging.info(f"Chart saved as '{svg_filename}', '{png_filename}' and '{pdf_filename}'.")

        if display_time > 0:
            plt.show(block=False)
            plt.pause(display_time)
            plt.close(fig)
            logging.info("Chart window closed automatically.")


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


def save_combined_eps(bar_charts, combined_eps_filename):

    num_charts = len(bar_charts)
    fig_width = 8 * num_charts
    fig_height = 6
    fig, axes = plt.subplots(1, num_charts, figsize=(fig_width, fig_height), constrained_layout=True)

    if num_charts == 1:
        axes = [axes]

    for ax, bar_chart in zip(axes, bar_charts):
        plot_bar_chart(
            file_path=bar_chart['file_path'],
            title=bar_chart.get('title'),
            y_label=bar_chart['y_label'],
            svg_filename=None,
            png_filename=None,
            pdf_filename=None,
            colors=bar_chart.get('colors'),
            legend_location=bar_chart.get('legend_location', 'upper right'),
            font_sizes=bar_chart.get('font_sizes'),
            rotate_xticks=bar_chart.get('rotate_xticks', 45),
            show_value_labels=bar_chart.get('show_value_labels', True),
            display_time=0,
            percentage_columns=bar_chart.get('percentage_columns'),
            y_min=bar_chart.get('y_min'),
            y_max=bar_chart.get('y_max'),
            group_spacing_factor=bar_chart.get('group_spacing_factor', 1.0),
            total_width=bar_chart.get('total_width', 0.4),
            y_scale=bar_chart.get('y_scale', 'linear'),
            ax=ax
        )

    fig.savefig(combined_eps_filename, format='eps', facecolor='white')
    logging.info(f"Combined EPS saved as '{combined_eps_filename}'.")
    plt.close(fig)


def main():
    palette1 = ['#4c9ac9', '#e2745e']
    palette2 = ['#9180ac', '#9dd0c7']
    palette3 = ['#d6afb9', '#7e9bb7']

    bar_charts = [
        {
            'file_path': 'd3qn_latebcy.txt',
            'title': None,
            'y_label': 'G-D3QN latency (us)',
            'svg_filename': 'plots/D3QN_Single-path_vs_Dual-path_Latency.svg',
            'png_filename': 'plots/D3QN_Single-path_vs_Dual-path_Latency.png',
            'pdf_filename': 'plots/D3QN_Single-path_vs_Dual-path_Latency.pdf',
            'colors': palette1,
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
            'y_max': 165.0,
            'group_spacing_factor': 0.6,
            'total_width': 0.4,
            'y_scale': 'linear'
        },
        {
            'file_path': 'd3qn_packet_loss.txt',
            'title': None,
            'y_label': 'G-D3QN Packet Loss (%)',
            'svg_filename': 'plots/D3QN_Single-path_vs_Dual-path_Packet_Loss.svg',
            'png_filename': 'plots/D3QN_Single-path_vs_Dual-path_Packet_Loss.png',
            'pdf_filename': 'plots/D3QN_Single-path_vs_Dual-path_Packet_Loss.pdf',
            'colors': palette2,
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
            'y_min': 1e-4,
            'y_max': 0.035,
            'group_spacing_factor': 0.6,
            'total_width': 0.4,
            'y_scale': 'log'
        },
        {
            'file_path': 'd3qn_loadCV.txt',
            'title': None,
            'y_label': 'G-D3QN LoadCV',
            'svg_filename': 'plots/D3QN_Single-path_vs_Dual-path_LoadCV.svg',
            'png_filename': 'plots/D3QN_Single-path_vs_Dual-path_LoadCV.png',
            'pdf_filename': 'plots/D3QN_Single-path_vs_Dual-path_LoadCV.pdf',
            'colors': palette3,
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
            'y_max': 0.8,
            'group_spacing_factor': 0.6,
            'total_width': 0.4,
            'y_scale': 'linear'
        },
    ]

    for bar_chart in bar_charts:
        plot_bar_chart(
            file_path=bar_chart['file_path'],
            title=bar_chart.get('title'),
            y_label=bar_chart['y_label'],
            svg_filename=bar_chart['svg_filename'],
            png_filename=bar_chart['png_filename'],
            pdf_filename=bar_chart['pdf_filename'],
            colors=bar_chart.get('colors'),
            legend_location=bar_chart.get('legend_location', 'upper right'),
            font_sizes=bar_chart.get('font_sizes'),
            rotate_xticks=bar_chart.get('rotate_xticks', 45),
            show_value_labels=bar_chart.get('show_value_labels', True),
            display_time=bar_chart.get('display_time', 3),
            percentage_columns=bar_chart.get('percentage_columns'),
            y_min=bar_chart.get('y_min'),
            y_max=bar_chart.get('y_max'),
            group_spacing_factor=bar_chart.get('group_spacing_factor', 1.0),
            total_width=bar_chart.get('total_width', 0.4),
            y_scale=bar_chart.get('y_scale', 'linear')
        )

    combined_eps_filename = 'plots/combined_plots.eps'
    save_combined_eps(bar_charts, combined_eps_filename)


if __name__ == "__main__":
    main()
