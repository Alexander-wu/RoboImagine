#!/usr/bin/env python3
"""Compact two-panel cost and PSNR chart with a single shared legend."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

INK = '#253141'
BLUE, PURPLE = '#225D91', '#9B84BC'
COLORS = [BLUE, PURPLE]
FONTS = HERE.parents[1]/'common/fonts'


def main():
    raw = (HERE / 'data.json').read_bytes()
    data = json.loads(raw)
    records = data['records']
    assert [r['method'] for r in records] == ['Continued SFT', 'RoboAlign-R2']
    assert all(r['updates'] == 500 for r in records)
    for r in records:
        assert math.isclose(r['train_seconds'] * r['gpus'] / 3600, r['gpu_hours'], rel_tol=1e-12)
    costs = [r['gpu_hours'] for r in records]
    gains = [r['psnr'] - data['baseline']['metrics']['psnr'] for r in records]
    for face in ['Regular', 'Bold', 'Italic', 'BoldItalic']:
        font_manager.fontManager.addfont(str(FONTS / f'EBGaramond-{face}.otf'))
    font_manager.findfont('EB Garamond', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'EB Garamond', 'font.size': 8.5,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': '#758294', 'xtick.labelsize': 7.2,
                         'svg.hashsalt': 'r2-cost-shared-legend-v4', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(3.5, 1.14))
    # One short legend frees the left margin and removes slanted row labels.
    legend = fig.legend(
        handles=[Patch(facecolor=color, label=('RoboImagine' if r['method']=='RoboAlign-R2' else r['method'])) for color, r in zip(COLORS, records)],
        loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False,
        borderaxespad=0, borderpad=0, handlelength=1.15, handleheight=.60,
        handletextpad=.45, columnspacing=1.8, prop={'family':'EB Garamond','size':8.5})
    for text, color in zip(legend.get_texts(), COLORS):
        text.set_color('#80639E' if color == PURPLE else color)
    bottom, height = .17, .50
    left = fig.add_axes([.045, bottom, .420, height])
    right = fig.add_axes([.565, bottom, .420, height])
    axes = [left, right]
    row_y = [1.15, -.15]
    ytop, ybottom = row_y
    fig.text(.045, .75, '(a) Cost (GPU-h)', fontsize=8.8, weight='bold', ha='left')
    fig.text(.565, .75, '(b) PSNR gain (dB)', fontsize=8.8, weight='bold', ha='left')
    for ax in axes:
        ax.set_ylim(-.60, 1.60)
        ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color('#C9D0DA')
        ax.spines['bottom'].set_linewidth(.55)
        ax.tick_params(axis='x', width=.5, length=2.5, pad=2)
        ax.grid(axis='x', color='#E8ECF1', linewidth=.55)
    left.set_xlim(0, 32)
    left.set_xticks([0, 15, 30])
    right.set_xlim(-.20, 1.40)
    right.set_xticks([0, .5, 1.0], ['0', '0.5', '1.0'])
    right.axvline(0, color='#A2ADBA', lw=.8, zorder=2)
    for ax, values in zip(axes, [costs, gains]):
        ax.barh(row_y, values, height=.52, color=COLORS, zorder=3)
    # Long-bar labels are white; small values sit beside their true endpoints.
    left.text(costs[1] / 2, ybottom, f'{costs[1]:.2f}', ha='center', va='center',
              fontsize=9.2, color=INK, weight='bold')
    left.annotate(f'{costs[0]:.2f}', (costs[0], ytop), xytext=(3, 0),
                  textcoords='offset points', ha='left', va='center',
                  color=BLUE, fontsize=9.2, weight='bold')
    right.text(gains[1] / 2, ybottom, f'+{gains[1]:.2f}', ha='center', va='center',
               fontsize=9.2, color=INK, weight='bold')
    right.scatter([gains[0]], [ytop], s=10, color=BLUE, edgecolor='white', linewidth=.4, zorder=4)
    right.annotate(f'{gains[0]:+.2f}'.replace('-', '−'), (0, ytop), xytext=(5, 0),
                   textcoords='offset points', ha='left', va='center',
                   color=BLUE, fontsize=9.2, weight='bold')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*legend.get_texts(), *fig.texts, *[t for ax in axes for t in [*ax.texts, *ax.get_xticklabels()]]]:
        bb = text.get_window_extent(renderer)
        assert bb.x0 >= -.5 and bb.y0 >= -.5 and bb.x1 <= fig.bbox.width + .5 and bb.y1 <= fig.bbox.height + .5, text.get_text()
    for suffix, mode in [('editable.svg', 'none'), ('svg', 'path')]:
        with plt.rc_context({'svg.fonttype': mode}):
            fig.savefig(HERE / f'figure.{suffix}', format='svg', metadata={'Date': None})
    plt.close(fig)
    sys.path.insert(0,str(HERE.parents[1]/"common"))
    from cairo_support import configure_cairo
    configure_cairo()
    import cairosvg
    cairosvg.svg2pdf(url=str(HERE/'figure.svg'),write_to=str(HERE/'figure.pdf'))
    cairosvg.svg2png(url=str(HERE/'figure.svg'),write_to=str(HERE/'figure.png'),output_width=2800)
    metadata = dict(data_sha256=hashlib.sha256(raw).hexdigest(), figure_inches=[3.5, 1.14],
                    cost_gpu_hours=costs, psnr_changes_db=gains,
                    cost_axis='linear, starts at zero', gain_axis='linear, includes negative values and visible zero',
                    methods=[r['method'] for r in records], font='EB Garamond', shared_legend=True, legend_position="top, one row", shared_method_label_rotation=0,
                    chart_width_fractions=[.420,.420],
                    shared_initialization=data['shared_initial_digest'], updates=500,
                    scope='Matched updates, not matched compute; same measured data as original Fig11',
                    error_bars='None: one measured run per method')
    (HERE / 'render_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (HERE / 'caption.txt').write_text((HERE / 'caption_source.txt').read_text())
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
