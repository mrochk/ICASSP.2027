import jax.numpy as jnp
from pathlib import Path
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

PLOTS_PATH = Path.cwd() / 'plots'

def setup():
    PLOTS_PATH.mkdir(exist_ok=True)

    sns.set_theme(context='paper', style='white', palette='dark')

    mpl.rcParams.update({
        'figure.figsize': (3.5, 1.9),
        'figure.dpi': 200,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'stixsans',
        'mathtext.default': 'regular',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'legend.frameon': False,
    })

    colors = sns.color_palette()
    return colors

def plot_before(sigmas1, sigmas2, errors, rconds, colors):
    _, axes = plt.subplots(2, 1, figsize=(3.5, 3))

    ax = axes[0]
    ax.semilogx(sigmas1, errors[:, 0], 'X', label=r'$y_1$', color=colors[0])
    ax.semilogx(sigmas1, errors[:, 1], 'X', label=r'$y_2$', color=colors[3])
    ax.set_ylabel('Mean Per-Output Error')
    ax.legend()

    ax = axes[1]
    ax.loglog(sigmas2, rconds[0], label=r'$W$', color=colors[2], linewidth=1.5)
    ax.loglog(sigmas2, rconds[1], label=r'$V$', color=colors[1], linewidth=1.5)
    ax.loglog(sigmas2, rconds[2], label=r'$H$', color=colors[0], linewidth=1.5)
    ax.loglog(sigmas2, rconds[3], label=r'$R$', color=colors[3], linewidth=1.5)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('Median RCond Number')
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'before.pdf')
    plt.show()

def plot_after(sigmas1, sigmas2, errors, rconds, colors):
    _, axes = plt.subplots(2, 1, figsize=(3.5, 2.9))

    ax = axes[0]
    ax.semilogx(sigmas1, errors[:, 0], 'X', label=r'$y_1$', color=colors[0])
    ax.semilogx(sigmas1, errors[:, 1], 'X', label=r'$y_2$', color=colors[3])
    ax.set_ylim((0.0, 10))
    ax.set_ylabel('Mean Per-Output Error')
    ax.legend()
    ax.set_yticks([0, 5, 10])

    ax = axes[1]
    ax.semilogx(sigmas2, rconds[0], label=r'$W$', color=colors[2], linewidth=2)
    ax.semilogx(sigmas2, rconds[1], label=r'$V$', color=colors[1], linewidth=2)
    ax.semilogx(sigmas2, rconds[2], label=r'$H$', color=colors[0], linewidth=2)
    ax.semilogx(sigmas2, rconds[3], label=r'$R$', color=colors[3], linewidth=2)
    ax.set_xlabel(r'$\sigma$')
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_ylabel('Median RCond Number')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'after.pdf')
    plt.show()

def plot_smoothing1(dofs, errors_nosmoothing, errors_smoothing, colors):
    _, ax = plt.subplots(figsize=(3.5, 1.9))
    m1 = '.'; m2 = '+'
    ax.scatter(dofs, errors_nosmoothing[:, 0], color=colors[3], marker=m1)
    ax.scatter(dofs, errors_nosmoothing[:, 1], color=colors[3], marker=m2)
    ax.scatter(dofs, errors_smoothing[:, 0],   color=colors[0], marker=m1)
    ax.scatter(dofs, errors_smoothing[:, 1],   color=colors[0], marker=m2)
    ax.set_xlabel(r'$df$')
    ax.set_ylabel('Mean Error')
    group_handles = [
        Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=colors[1],
           markeredgecolor=colors[1], markersize=8, label='No Smoothing (B-Splines)'),
        Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=colors[0],
           markeredgecolor=colors[0], markersize=8, label='With Smoothing (P-Splines)'),
    ]
    legend1 = ax.legend(handles=group_handles, loc='upper left', frameon=False)
    ax.add_artist(legend1)
    output_handles = [
        Line2D([0], [0], marker=m1, color='black', linestyle='None', markersize=6, label=r'$y_1$'),
        Line2D([0], [0], marker=m2, color='black', linestyle='None', markersize=6, label=r'$y_2$'),
    ]
    ax.legend(
        handles=output_handles,
        loc='upper right',
        bbox_to_anchor=(0.8, 1.0),
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'smoothing1.pdf')
    plt.show()

def plot_smoothing2(dofs, errors_nosmoothing, errors_smoothing, colors):
    edges = [10, 50, 60, 70, 80, 90, 101]
    groups = list(zip(edges[:-1], edges[1:]))
    ticks = [f'[{lo}-{hi-1}]' for lo, hi in groups]
    pos = jnp.arange(len(groups))

    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for errs, color, off in [
        (errors_nosmoothing, colors[3], +0.18),
        (errors_smoothing, colors[0], -0.18),
    ]: 

        vals = [errs[(dofs >= lo) & (dofs < hi)].ravel() for lo, hi in groups]
    
        bp = ax.boxplot(
            vals,
            positions=pos + off,
            widths=0.3,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=0.7),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            medianprops=dict(color='white', linewidth=0.7)
        )
    
        for patch in bp['boxes']: patch.set_facecolor(color)

    ax.axvline(x=0.5, color='black', linewidth=0.1,  zorder=0)
    ax.axvline(x=1.5, color='black', linewidth=0.1,  zorder=0)
    ax.axvline(x=2.5, color='black', linewidth=0.1,  zorder=0)
    ax.axvline(x=3.5, color='black', linewidth=0.1,  zorder=0)
    ax.axvline(x=4.5, color='black', linewidth=0.1,  zorder=0)

    ax.set_xticks(pos); ax.set_xticklabels(ticks)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0.7, 210)
    ax.axhline(y=10, color='black', linestyle='--', linewidth=1,  zorder=0,  alpha=0.7)
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1,  zorder=0, alpha=0.7)

    ax.legend(
        handles=[
            Patch(facecolor=colors[3], label='No Smoothing (B-Splines)'),
            Patch(facecolor=colors[0], label='With Smoothing (P-Splines)')
        ],
        frameon=True, 
        loc='upper left',
        fontsize=6,
        framealpha=1.0,
    )

    ax.set_xlabel(r'$df$', fontsize='small'); ax.set_ylabel('Mean Error')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    plt.savefig(PLOTS_PATH / 'smoothing2.pdf')
    plt.show()
