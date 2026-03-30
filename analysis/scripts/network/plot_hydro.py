import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
 
# 1. Load data
file_path = '/p/tmp/lorenzop/pypsa-earth-ivan/data/hydro_capacities.csv'
col_names = ['Country', 'Code', 'P_nom', 'P_store', 'E_store', 'Inflow']
df = pd.read_csv(file_path, names=col_names, header=None)
 
for col in ['P_nom', 'P_store', 'E_store', 'Inflow']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['P_nom', 'E_store'])
 
# 2. Filter Africa (GW/MW bug) and micro-systems
african_iso2 = [
    'DZ','AO','BJ','BW','BF','BI','CV','CM','CF','TD','KM','CD','CG','CI','DJ',
    'EG','GQ','ER','SZ','ET','GA','GM','GH','GN','GW','KE','LS','LR','LY','MG',
    'MW','ML','MR','MU','MA','MZ','NA','NE','NG','RW','ST','SN','SC','SL','SO',
    'ZA','SS','SD','TZ','TG','TN','UG','ZM','ZW'
]
df = df[~df['Code'].isin(african_iso2)]
df = df[df['P_nom'] > 0.05].copy()
 
# 3. Storage duration: E_store [TWh] / P_nom [GW] * 1000 → hours
df['Storage_Hours'] = (df['E_store'] / df['P_nom']) * 1000
 
# 4. Figure
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10, 5000)
ax.set_ylim(0.1, 1000)
 
# 5. Iso-energy curves
# On a log-log plot with equal decades per unit length, E=P*H/1000
# lines have slope -1 → display angle is always -45°.
# We pre-compute the angle correction for the actual figure aspect ratio
# using a purely geometric approach (no canvas.draw needed).
fig_w, fig_h = fig.get_size_inches()          # inches
ax_pos = ax.get_position()                    # fractional axes size
ax_w = ax_pos.width  * fig_w                  # axes width  in inches
ax_h = ax_pos.height * fig_h                  # axes height in inches
x_decades = np.log10(5000) - np.log10(10)     # 2.7
y_decades = np.log10(1000) - np.log10(0.1)    # 4.0
# pixels per decade on each axis
px_per_dec_x = ax_w / x_decades
px_per_dec_y = ax_h / y_decades
# slope in display space: dy/dx in display coords for a -1 log-log slope
angle_rad = np.arctan2(-px_per_dec_y, px_per_dec_x)
iso_angle  = np.degrees(angle_rad)            # typically around -50° to -55°
 
x_iso = np.logspace(np.log10(10), np.log10(5000), 300)
iso_twh_levels = [1, 5, 20, 50, 100, 200, 500]
 
for e_twh in iso_twh_levels:
    y_iso = (e_twh * 1000) / x_iso
    mask  = (y_iso >= 0.1) & (y_iso <= 1000)
    if not mask.any():
        continue
    ax.plot(x_iso[mask], y_iso[mask], '--', color='#bdbdbd', alpha=0.7,
            linewidth=1.0, zorder=1)
 
    # Label at ~15% along the visible curve (upper-left end)
    x_vis, y_vis = x_iso[mask], y_iso[mask]
    i = int(len(x_vis) * 0.15)
    ax.text(x_vis[i], y_vis[i], f'{e_twh} TWh',
            color='#777777', fontsize=8.5, ha='left', va='bottom',
            rotation=iso_angle, rotation_mode='anchor',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8),
            zorder=2)
 
# 6. Scatter
china  = df[df['Code'] == 'CN']
others = df[df['Code'] != 'CN']
 
ax.scatter(others['Storage_Hours'], others['P_nom'],
           color='#2b8cbe', alpha=0.85, edgecolor='white', s=70, linewidth=0.8, zorder=3)
if not china.empty:
    ax.scatter(china['Storage_Hours'], china['P_nom'],
               color='#de2d26', alpha=0.95, edgecolor='white', s=100, linewidth=0.8, zorder=4)
 
# 7. Country labels
# Offset in log-space multipliers (x_mult, y_mult, ha, va)
offsets = {
    'CN': (1.10, 1.00, 'left',  'center'),
    'NO': (1.10, 1.00, 'left',  'center'),
    'SE': (1.10, 1.00, 'left',  'center'),
    'FR': (1.10, 1.12, 'left',  'bottom'),
    'IT': (0.90, 1.12, 'right', 'bottom'),
    'AT': (0.90, 1.00, 'right', 'center'),
    'CH': (1.10, 1.00, 'left',  'center'),
    'ES': (1.10, 1.00, 'left',  'center'),
    'DE': (1.10, 1.00, 'left',  'center'),
    'PT': (0.90, 1.00, 'right', 'center'),
    'RO': (1.10, 1.00, 'left',  'center'),
    'FI': (1.10, 1.00, 'left',  'center'),
}
default = (1.10, 1.00, 'left', 'center')
 
for _, row in df.iterrows():
    code = row['Code']
    ox, oy, ha, va = offsets.get(code, default)
    ax.text(row['Storage_Hours'] * ox, row['P_nom'] * oy,
            'China (CN)' if code == 'CN' else code,
            color='#de2d26' if code == 'CN' else '#333333',
            fontsize=11 if code == 'CN' else 9,
            fontweight='bold' if code == 'CN' else 'normal',
            ha=ha, va=va, zorder=5)
 
# 8. Labels and style
ax.set_title('Global Hydroelectric Profiles: Storage Duration vs. Installed Capacity',
             fontsize=16, fontweight='bold', pad=20, color='#111111')
ax.set_xlabel('Storage Duration [Equivalent Hours]', fontsize=13)
ax.set_ylabel('Installed Capacity [GW]', fontsize=13)
ax.grid(True, which='major', ls='-',  alpha=0.3, color='#cccccc')
ax.grid(True, which='minor', ls=':', alpha=0.2, color='#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
# 9. Save
output_file = Path(__file__).parent / 'hydro_archetypes_clean.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Grafico salvato in: {output_file}")