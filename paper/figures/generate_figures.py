#!/usr/bin/env python3
"""Generate publication-quality figures for the CNN robustness capstone paper."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# ── Style setup ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
})

# Colorblind-friendly palette (Wong 2011)
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_RED = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_CYAN = '#56B4E9'
CB_YELLOW = '#F0E442'

# ── Data paths ───────────────────────────────────────────────────────────────
BASE = '/home/possum/cv-robustness-thesis/results'
OUTDIR = '/home/possum/cv-robustness-thesis/paper/figures'

summary = pd.read_csv(f'{BASE}/analysis/analysis/summary_table.csv')

# Also load raw CSVs for per-seed data
raw_files = []
for f in ['baseline/baseline/resnet18_baseline_robustness.csv',
          'baseline/baseline/mobilenetv2_baseline_robustness.csv',
          'augmented/augmented/resnet18_mixed_robustness.csv',
          'augmented/augmented/mobilenetv2_mixed_robustness.csv',
          'augmented/augmented/resnet18_single_gaussian_blur_robustness.csv',
          'augmented/augmented/mobilenetv2_single_gaussian_blur_robustness.csv',
          'augmented/augmented/resnet18_single_gaussian_noise_robustness.csv',
          'augmented/augmented/mobilenetv2_single_gaussian_noise_robustness.csv',
          'augmented/augmented/resnet18_curriculum_gaussian_blur_robustness.csv',
          'augmented/augmented/mobilenetv2_curriculum_gaussian_blur_robustness.csv',
          'augmented/augmented/resnet18_curriculum_gaussian_noise_robustness.csv',
          'augmented/augmented/mobilenetv2_curriculum_gaussian_noise_robustness.csv']:
    raw_files.append(pd.read_csv(f'{BASE}/{f}'))
raw = pd.concat(raw_files, ignore_index=True)

# Readable degradation names
DEG_NAMES = {
    'gaussian_noise': 'Gaussian\nNoise',
    'gaussian_blur': 'Gaussian\nBlur',
    'motion_blur': 'Motion\nBlur',
    'jpeg': 'JPEG',
    'contrast': 'Contrast',
    'darkening': 'Darkening',
    'clean': 'Clean',
}
DEG_NAMES_ONELINE = {
    'gaussian_noise': 'Gauss. Noise',
    'gaussian_blur': 'Gauss. Blur',
    'motion_blur': 'Motion Blur',
    'jpeg': 'JPEG',
    'contrast': 'Contrast',
    'darkening': 'Darkening',
    'clean': 'Clean',
}
DEG_ORDER = ['gaussian_noise', 'gaussian_blur', 'motion_blur', 'jpeg', 'contrast', 'darkening']

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Baseline heatmaps
# ═══════════════════════════════════════════════════════════════════════════════
def fig_baseline_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

    for ax, model, title in zip(axes, ['resnet18', 'mobilenetv2'],
                                 ['ResNet-18', 'MobileNet-V2']):
        s = summary[(summary['model'] == model) & (summary['strategy'] == 'baseline')]

        # Build matrix: rows = degradation types (+ clean), cols = severity
        rows = ['clean'] + DEG_ORDER
        row_labels = ['Clean'] + [DEG_NAMES_ONELINE[d] for d in DEG_ORDER]
        cols = [0, 1, 2, 3]
        col_labels = ['Clean', 'Mild', 'Medium', 'Severe']

        matrix = np.full((len(rows), 4), np.nan)
        for i, deg in enumerate(rows):
            for j, sev in enumerate(cols):
                if deg == 'clean' and sev > 0:
                    continue
                if deg != 'clean' and sev == 0:
                    continue
                match = s[(s['eval_degradation'] == deg) & (s['eval_severity'] == sev)]
                if len(match) > 0:
                    matrix[i, j] = match['mean'].values[0]

        # For clean row, put value in first column only; for degraded rows, cols 1-3
        # Restructure: clean row has 1 value, degraded rows have 3 values
        # Better approach: rows = [clean] + 6 degs, columns = severity levels
        # Clean occupies a single row spanning all columns with same value

        # Simpler: make matrix with rows=degs, cols=severities, clean as separate row
        n_rows = 7  # clean + 6 degs
        n_cols = 3  # 3 severity levels
        mat = np.full((n_rows, n_cols), np.nan)

        # Clean row: same value across all 3 columns
        clean_val = s[(s['eval_degradation'] == 'clean')]['mean'].values[0]
        mat[0, :] = clean_val

        for i, deg in enumerate(DEG_ORDER):
            for j, sev in enumerate([1, 2, 3]):
                match = s[(s['eval_degradation'] == deg) & (s['eval_severity'] == sev)]
                if len(match) > 0:
                    mat[i + 1, j] = match['mean'].values[0]

        # Custom colormap: red to yellow to green
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'rg', ['#d73027', '#fee08b', '#1a9850'], N=256)

        im = ax.imshow(mat, cmap=cmap, vmin=10, vmax=100, aspect='auto')

        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 40 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color=color)

        row_labels = ['Clean'] + [DEG_NAMES_ONELINE[d] for d in DEG_ORDER]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(['Mild', 'Medium', 'Severe'])
        ax.set_title(title, fontweight='bold')

        # Light grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', size=0)
        # Remove default spines for heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Accuracy (%)')

    fig.savefig(f'{OUTDIR}/fig_baseline_heatmap.png')
    plt.close(fig)
    print('  Saved fig_baseline_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Tradeoff scatter
# ═══════════════════════════════════════════════════════════════════════════════
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # For each (model, strategy, deg_type_trained, seed): compute clean acc and mean robustness
    # Mean robustness = average accuracy across all 18 degraded conditions (6 types x 3 severities)
    records = []
    for (model, strategy, deg, seed), grp in raw.groupby(['model', 'strategy', 'deg_type_trained', 'seed']):
        clean = grp[grp['eval_degradation'] == 'clean']['accuracy'].values
        if len(clean) == 0:
            continue
        clean_acc = clean[0]
        degraded = grp[grp['eval_degradation'] != 'clean']['accuracy']
        if len(degraded) == 0:
            continue
        mean_rob = degraded.mean()

        # Label for strategy
        if strategy == 'baseline':
            label = 'Baseline'
        elif strategy == 'mixed':
            label = 'Mixed'
        elif strategy == 'single' and deg == 'gaussian_blur':
            label = 'Single (Blur)'
        elif strategy == 'single' and deg == 'gaussian_noise':
            label = 'Single (Noise)'
        elif strategy == 'curriculum' and deg == 'gaussian_blur':
            label = 'Curriculum (Blur)'
        elif strategy == 'curriculum' and deg == 'gaussian_noise':
            label = 'Curriculum (Noise)'
        else:
            label = f'{strategy}({deg})'

        records.append({
            'model': model, 'strategy_label': label, 'seed': seed,
            'clean_acc': clean_acc, 'mean_robustness': mean_rob
        })

    df = pd.DataFrame(records)

    # Markers for strategies, colors for models
    model_colors = {'resnet18': CB_BLUE, 'mobilenetv2': CB_ORANGE}
    model_names = {'resnet18': 'ResNet-18', 'mobilenetv2': 'MobileNet-V2'}
    strategy_markers = {
        'Baseline': 'o', 'Mixed': 's', 'Single (Blur)': '^',
        'Single (Noise)': 'v', 'Curriculum (Blur)': 'X', 'Curriculum (Noise)': 'D'
    }

    # Plot individual seeds (small, transparent)
    for (model, strat), grp in df.groupby(['model', 'strategy_label']):
        ax.scatter(grp['clean_acc'], grp['mean_robustness'],
                   c=model_colors[model], marker=strategy_markers.get(strat, 'o'),
                   s=20, alpha=0.3, linewidths=0)

    # Plot means (larger markers)
    means = df.groupby(['model', 'strategy_label']).agg(
        clean_mean=('clean_acc', 'mean'), clean_std=('clean_acc', 'std'),
        rob_mean=('mean_robustness', 'mean'), rob_std=('mean_robustness', 'std')
    ).reset_index()

    # Track handles for legend
    model_handles = {}
    strat_handles = {}

    for _, row in means.iterrows():
        m = row['model']
        s = row['strategy_label']
        marker = strategy_markers.get(s, 'o')
        color = model_colors[m]

        h = ax.scatter(row['clean_mean'], row['rob_mean'],
                       c=color, marker=marker, s=100, edgecolors='black',
                       linewidths=0.5, zorder=5)

        # Error bars
        ax.errorbar(row['clean_mean'], row['rob_mean'],
                     xerr=row['clean_std'], yerr=row['rob_std'],
                     fmt='none', ecolor=color, alpha=0.5, capsize=2, linewidth=1)

        if m not in model_handles:
            model_handles[m] = ax.scatter([], [], c=color, marker='o', s=60,
                                           label=model_names[m])
        if s not in strat_handles:
            strat_handles[s] = ax.scatter([], [], c='gray', marker=marker, s=60,
                                           label=s)

    # Diagonal reference line (clean = robustness)
    lims = [10, 100]
    ax.plot(lims, lims, '--', color='gray', alpha=0.4, linewidth=0.8, zorder=0)

    # "Ideal" annotation
    ax.annotate('Ideal', xy=(96, 96), fontsize=8, color='gray', alpha=0.6,
                ha='center')

    ax.set_xlabel('Clean Accuracy (%)')
    ax.set_ylabel('Mean Degraded Accuracy (%)')
    ax.set_xlim(5, 100)
    ax.set_ylim(5, 100)

    # Two-part legend
    legend1 = ax.legend(handles=list(model_handles.values()),
                         loc='upper left', title='Model', framealpha=0.9,
                         title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=list(strat_handles.values()),
              loc='lower right', title='Strategy', framealpha=0.9,
              title_fontsize=9, fontsize=8)

    fig.savefig(f'{OUTDIR}/fig_tradeoff.png')
    plt.close(fig)
    print('  Saved fig_tradeoff.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 & 4: Strategy comparison bar charts
# ═══════════════════════════════════════════════════════════════════════════════
def fig_strategy_comparison(model, model_name, filename):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Strategies to compare
    strategies = [
        ('baseline', 'none', 'Baseline'),
        ('mixed', 'none', 'Mixed'),
        ('single', 'gaussian_blur', 'Single (Blur)'),
        ('single', 'gaussian_noise', 'Single (Noise)'),
    ]

    colors = [CB_BLUE, CB_GREEN, CB_ORANGE, CB_RED]
    n_degs = len(DEG_ORDER)
    n_strats = len(strategies)
    width = 0.18
    x = np.arange(n_degs)

    for i, (strat, deg, label) in enumerate(strategies):
        means = []
        stds = []
        for d in DEG_ORDER:
            # Average across severities
            sub = summary[(summary['model'] == model) &
                          (summary['strategy'] == strat) &
                          (summary['deg_type_trained'] == deg) &
                          (summary['eval_degradation'] == d)]
            if len(sub) > 0:
                means.append(sub['mean'].mean())
                # Pooled std: use mean of stds (approximate)
                stds.append(sub['std'].mean())
            else:
                means.append(0)
                stds.append(0)

        offset = (i - (n_strats - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=label,
                       color=colors[i], capsize=2, error_kw={'linewidth': 0.8},
                       edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([DEG_NAMES_ONELINE[d] for d in DEG_ORDER])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.set_title(f'{model_name}: Strategy Comparison by Degradation Type', fontweight='bold')

    # Light horizontal reference lines
    for y in [25, 50, 75]:
        ax.axhline(y, color='gray', alpha=0.15, linewidth=0.5, zorder=0)

    fig.savefig(f'{OUTDIR}/{filename}')
    plt.close(fig)
    print(f'  Saved {filename}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Curriculum collapse
# ═══════════════════════════════════════════════════════════════════════════════
def fig_curriculum_collapse():
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # All (model, strategy, deg_type_trained) combos, clean accuracy
    combos = [
        ('resnet18', 'baseline', 'none', 'Baseline'),
        ('resnet18', 'mixed', 'none', 'Mixed'),
        ('resnet18', 'single', 'gaussian_blur', 'Single\n(Blur)'),
        ('resnet18', 'single', 'gaussian_noise', 'Single\n(Noise)'),
        ('resnet18', 'curriculum', 'gaussian_noise', 'Curric.\n(Noise)'),
        ('resnet18', 'curriculum', 'gaussian_blur', 'Curric.\n(Blur)'),
        ('mobilenetv2', 'baseline', 'none', 'Baseline'),
        ('mobilenetv2', 'mixed', 'none', 'Mixed'),
        ('mobilenetv2', 'single', 'gaussian_blur', 'Single\n(Blur)'),
        ('mobilenetv2', 'single', 'gaussian_noise', 'Single\n(Noise)'),
        ('mobilenetv2', 'curriculum', 'gaussian_noise', 'Curric.\n(Noise)'),
        ('mobilenetv2', 'curriculum', 'gaussian_blur', 'Curric.\n(Blur)'),
    ]

    resnet_data = [(l, m, st, dt) for m, st, dt, l in combos if m == 'resnet18']
    mobile_data = [(l, m, st, dt) for m, st, dt, l in combos if m == 'mobilenetv2']

    labels_r = [x[0] for x in resnet_data]
    labels_m = [x[0] for x in mobile_data]

    n = len(resnet_data)
    x = np.arange(n)
    width = 0.35

    means_r, stds_r, means_m, stds_m = [], [], [], []
    for label, model, strat, deg in resnet_data:
        row = summary[(summary['model'] == model) & (summary['strategy'] == strat) &
                       (summary['deg_type_trained'] == deg) &
                       (summary['eval_degradation'] == 'clean')]
        means_r.append(row['mean'].values[0] if len(row) > 0 else 0)
        stds_r.append(row['std'].values[0] if len(row) > 0 else 0)

    for label, model, strat, deg in mobile_data:
        row = summary[(summary['model'] == model) & (summary['strategy'] == strat) &
                       (summary['deg_type_trained'] == deg) &
                       (summary['eval_degradation'] == 'clean')]
        means_m.append(row['mean'].values[0] if len(row) > 0 else 0)
        stds_m.append(row['std'].values[0] if len(row) > 0 else 0)

    bars_r = ax.bar(x - width/2, means_r, width, yerr=stds_r, label='ResNet-18',
                     color=CB_BLUE, capsize=3, error_kw={'linewidth': 0.8},
                     edgecolor='white', linewidth=0.3)
    bars_m = ax.bar(x + width/2, means_m, width, yerr=stds_m, label='MobileNet-V2',
                     color=CB_ORANGE, capsize=3, error_kw={'linewidth': 0.8},
                     edgecolor='white', linewidth=0.3)

    # Highlight the collapse bars (curriculum blur = index 5)
    collapse_idx = 5
    bars_r[collapse_idx].set_edgecolor(CB_RED)
    bars_r[collapse_idx].set_linewidth(2)
    bars_m[collapse_idx].set_edgecolor(CB_RED)
    bars_m[collapse_idx].set_linewidth(2)

    # Annotate collapse values
    ax.annotate(f'{means_r[collapse_idx]:.1f}%',
                xy=(collapse_idx - width/2, means_r[collapse_idx] + stds_r[collapse_idx] + 1),
                ha='center', va='bottom', fontsize=8, color=CB_RED, fontweight='bold')
    ax.annotate(f'{means_m[collapse_idx]:.1f}%',
                xy=(collapse_idx + width/2, means_m[collapse_idx] + stds_m[collapse_idx] + 1),
                ha='center', va='bottom', fontsize=8, color=CB_RED, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_r, fontsize=9)
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('Clean Accuracy by Training Strategy', fontweight='bold')

    # Reference line at chance level
    ax.axhline(10, color=CB_RED, alpha=0.3, linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n - 0.5, 11, 'Chance', fontsize=7, color=CB_RED, alpha=0.5, ha='right')

    fig.savefig(f'{OUTDIR}/fig_curriculum_collapse.png')
    plt.close(fig)
    print('  Saved fig_curriculum_collapse.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Transfer effects
# ═══════════════════════════════════════════════════════════════════════════════
def fig_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

    model = 'resnet18'
    train_types = [('gaussian_blur', 'Trained on Blur'), ('gaussian_noise', 'Trained on Noise')]

    for ax, (train_deg, title) in zip(axes, train_types):
        # Get single-type trained data
        s = summary[(summary['model'] == model) &
                     (summary['strategy'] == 'single') &
                     (summary['deg_type_trained'] == train_deg) &
                     (summary['eval_degradation'] != 'clean')]

        # Also get baseline
        b = summary[(summary['model'] == model) &
                     (summary['strategy'] == 'baseline') &
                     (summary['eval_degradation'] != 'clean')]

        # Build matrices
        n_degs = len(DEG_ORDER)
        n_sevs = 3
        mat_single = np.full((n_degs, n_sevs), np.nan)
        mat_baseline = np.full((n_degs, n_sevs), np.nan)

        for i, deg in enumerate(DEG_ORDER):
            for j, sev in enumerate([1, 2, 3]):
                row_s = s[(s['eval_degradation'] == deg) & (s['eval_severity'] == sev)]
                row_b = b[(b['eval_degradation'] == deg) & (b['eval_severity'] == sev)]
                if len(row_s) > 0:
                    mat_single[i, j] = row_s['mean'].values[0]
                if len(row_b) > 0:
                    mat_baseline[i, j] = row_b['mean'].values[0]

        # Show the DIFFERENCE (single - baseline)
        diff = mat_single - mat_baseline

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'div', ['#d73027', '#ffffff', '#1a9850'], N=256)

        vmax = np.nanmax(np.abs(diff))
        im = ax.imshow(diff, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        # Annotate
        for i in range(n_degs):
            for j in range(n_sevs):
                val = diff[i, j]
                if not np.isnan(val):
                    sign = '+' if val > 0 else ''
                    color = 'black'
                    if abs(val) > vmax * 0.7:
                        color = 'white'
                    ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                            fontsize=7.5, fontweight='bold', color=color)

        ax.set_yticks(range(n_degs))
        ax.set_yticklabels([DEG_NAMES_ONELINE[d] for d in DEG_ORDER])
        ax.set_xticks(range(n_sevs))
        ax.set_xticklabels(['Mild', 'Medium', 'Severe'])
        ax.set_title(title, fontweight='bold')

        # Highlight the trained-on row
        trained_idx = DEG_ORDER.index(train_deg)
        ax.add_patch(plt.Rectangle((-0.5, trained_idx - 0.5), n_sevs, 1,
                                    fill=False, edgecolor='black', linewidth=2))

        # Grid
        ax.set_xticks(np.arange(-0.5, n_sevs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_degs, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', size=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Accuracy Change vs. Baseline (pp)')

    fig.suptitle('ResNet-18: Transfer Effects of Single-Type Training', fontweight='bold', y=1.02)
    fig.savefig(f'{OUTDIR}/fig_transfer.png')
    plt.close(fig)
    print('  Saved fig_transfer.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures...')
    fig_baseline_heatmap()
    fig_tradeoff()
    fig_strategy_comparison('resnet18', 'ResNet-18', 'fig_strategy_comparison.png')
    fig_strategy_comparison('mobilenetv2', 'MobileNet-V2', 'fig_strategy_comparison_mobilenet.png')
    fig_curriculum_collapse()
    fig_transfer()
    print('Done!')
