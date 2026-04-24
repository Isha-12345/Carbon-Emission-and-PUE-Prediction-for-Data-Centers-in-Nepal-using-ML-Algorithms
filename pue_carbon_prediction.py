# =============================================================================
# Carbon Emission and PUE Prediction for Data Centers in Nepal
# Using Random Forest Regression
#
# Author  : Isha Adhikari
# Dataset : Data Center Cold Source Control Dataset (Kaggle)
# Source  : https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset
#
# This script follows the methodology described in the paper:
#   Section 4.1  - Dataset loading and description
#   Section 4.2  - Target variable derivation (PUE and CO2e)
#   Section 4.3  - Feature engineering
#   Section 4.4  - Random Forest model training and evaluation
#   Section 4.5  - Nepal projection methodology (Scenario A and B)
#   Section 5    - Results, figures, and feature importance
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

DATASET_PATH     = 'cold_source_control_dataset.csv'
OUTPUT_DIR       = '.'          # folder where figures are saved
RANDOM_STATE     = 42
TEST_SIZE        = 0.20         # 80/20 split as stated in the paper
N_FOLDS          = 5            # 5-fold cross-validation
N_ESTIMATORS     = 200          # number of trees in the forest
MAX_DEPTH        = None         # let trees grow fully
MIN_SAMPLES_LEAF = 2

# Nepal-specific constants (Section 4.5)
CI_NEPAL         = 23           # gCO2/kWh - Nepal grid carbon intensity (hydropower)
TEMP_SCENARIO_A  = 17.0         # Kathmandu annual mean ambient temperature (degrees C)
TEMP_SCENARIO_B  = 30.0         # Kathmandu peak summer temperature (degrees C)

# Reference peak IT power (kW) — normalization baseline (Section 4.2)
P_MAX_IT = 1.0

# Colour palette — used consistently in all figures
C_BLUE   = '#1a6faf'
C_RED    = '#c0392b'
C_GREEN  = '#1e8449'
C_ORANGE = '#d35400'
C_GREY   = '#7f8c8d'
C_LIGHT  = '#eaf3fb'

print("=" * 65)
print("  PUE and Carbon Emission Prediction — Nepal Data Center Study")
print("=" * 65)

# =============================================================================
# 1. LOAD DATASET  (Section 4.1)
# =============================================================================

print("\n[1] Loading dataset ...")
df = pd.read_csv(DATASET_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Rename columns to clean Python-friendly names
df.columns = [
    'Timestamp', 'Server_Workload', 'Inlet_Temp', 'Outlet_Temp',
    'Ambient_Temp', 'Cooling_Power', 'Chiller_Usage', 'AHU_Usage',
    'Energy_Cost', 'Temp_Deviation', 'Strategy_Action', 'Output'
]

print(f"   Records loaded  : {len(df):,}")
print(f"   Date range      : {df['Timestamp'].min().date()} to {df['Timestamp'].max().date()}")
print(f"   Columns         : {list(df.columns)}")

# =============================================================================
# 2. DERIVE TARGET VARIABLES  (Section 4.2)
# =============================================================================

print("\n[2] Deriving PUE and CO2e targets ...")

# IT power — Equation 1 in the paper
# P_IT = (Server_Workload / 100) * P_MAX_IT
df['IT_Power'] = (df['Server_Workload'] / 100.0) * P_MAX_IT

# Total facility power — Equation 2
# P_total = P_IT + P_cooling
df['Total_Power'] = df['IT_Power'] + df['Cooling_Power']

# PUE — Equation 3
# PUE = P_total / P_IT  =  1 + (P_cooling / P_IT)
# Replace zero IT_Power to avoid division by zero
df['PUE'] = np.where(
    df['IT_Power'] > 0,
    df['Total_Power'] / df['IT_Power'],
    np.nan
)

# Hourly CO2e in grams — Equation 4
# CO2e = P_total (kW) * 1 hr * CI_region (gCO2/kWh)
df['CO2e_g'] = df['Total_Power'] * 1.0 * CI_NEPAL

print(f"   PUE  — min: {df['PUE'].min():.3f}  mean: {df['PUE'].mean():.3f}  "
      f"max: {df['PUE'].max():.3f}  std: {df['PUE'].std():.3f}")
print(f"   CO2e — min: {df['CO2e_g'].min():.2f}  mean: {df['CO2e_g'].mean():.2f}  "
      f"max: {df['CO2e_g'].max():.2f}  std: {df['CO2e_g'].std():.2f}  g/hr")

# =============================================================================
# 3. FEATURE ENGINEERING  (Section 4.3)
# =============================================================================

print("\n[3] Building features ...")

# Time features
df['Hour']        = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek    # 0 = Monday
df['Month']       = df['Timestamp'].dt.month
df['Is_Weekend']  = (df['Day_of_Week'] >= 5).astype(int)

# Thermal ratio features
df['DeltaT_ratio']        = df['Outlet_Temp'] / df['Inlet_Temp'].replace(0, np.nan)
df['Cooling_to_IT_ratio'] = df['Cooling_Power'] / df['IT_Power'].replace(0, np.nan)

# Lag features — previous hour values
df = df.sort_values('Timestamp').reset_index(drop=True)
df['Cooling_Power_lag1']   = df['Cooling_Power'].shift(1)
df['Server_Workload_lag1'] = df['Server_Workload'].shift(1)

# Rolling average features — 3-hour window
df['Cooling_Power_roll3']   = df['Cooling_Power'].rolling(window=3).mean()
df['Server_Workload_roll3'] = df['Server_Workload'].rolling(window=3).mean()

# One-hot encode cooling strategy action
strategy_dummies = pd.get_dummies(df['Strategy_Action'], prefix='Strategy')
df = pd.concat([df, strategy_dummies], axis=1)

# Drop rows with NaN (from lag and rolling operations)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"   Rows after feature engineering : {len(df):,}")

# Define feature columns (18+ features matching the paper)
FEATURE_COLS = [
    # Raw sensor features
    'Server_Workload', 'Inlet_Temp', 'Outlet_Temp',
    'Ambient_Temp', 'Cooling_Power', 'Chiller_Usage',
    'AHU_Usage', 'Energy_Cost', 'Temp_Deviation',
    # Time features
    'Hour', 'Day_of_Week', 'Month', 'Is_Weekend',
    # Thermal ratio features
    'DeltaT_ratio', 'Cooling_to_IT_ratio',
    # Lag features
    'Cooling_Power_lag1', 'Server_Workload_lag1',
    # Rolling average features
    'Cooling_Power_roll3', 'Server_Workload_roll3',
]

# Add one-hot strategy columns (exclude the original string column 'Strategy_Action')
strategy_cols = [c for c in df.columns if c.startswith('Strategy_') and c != 'Strategy_Action']
FEATURE_COLS += strategy_cols

print(f"   Total features used            : {len(FEATURE_COLS)}")
print(f"   Feature list                   : {FEATURE_COLS}")

X      = df[FEATURE_COLS]
y_pue  = df['PUE']
y_co2  = df['CO2e_g']

# =============================================================================
# 4. TRAIN / TEST SPLIT  (Section 4.6)
# =============================================================================

print("\n[4] Splitting data 80% train / 20% test ...")
X_train, X_test, y_pue_train, y_pue_test = train_test_split(
    X, y_pue, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"   Training samples : {len(X_train):,}")
print(f"   Testing samples  : {len(X_test):,}")

# =============================================================================
# 5. RANDOM FOREST — PUE MODEL  (Section 4.4)
# =============================================================================

print("\n[5] Training Random Forest for PUE ...")

rf_pue = RandomForestRegressor(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    random_state     = RANDOM_STATE,
    n_jobs           = -1
)
rf_pue.fit(X_train, y_pue_train)

# 5-fold cross-validation on the training set
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_r2_pue = cross_val_score(rf_pue, X_train, y_pue_train, cv=kf, scoring='r2')
print(f"   CV R2 scores (5-fold) : {cv_r2_pue.round(4)}")
print(f"   CV R2 mean +/- std    : {cv_r2_pue.mean():.4f} +/- {cv_r2_pue.std():.4f}")

# Test set evaluation
pue_pred = rf_pue.predict(X_test)
pue_rmse = np.sqrt(mean_squared_error(y_pue_test, pue_pred))
pue_mae  = mean_absolute_error(y_pue_test, pue_pred)
pue_r2   = r2_score(y_pue_test, pue_pred)

print(f"\n   --- PUE Test Results ---")
print(f"   RMSE : {pue_rmse:.4f}")
print(f"   MAE  : {pue_mae:.4f}")
print(f"   R2   : {pue_r2:.4f}")

# =============================================================================
# 6. RANDOM FOREST — CO2e MODEL  (Section 4.4)
# =============================================================================

print("\n[6] Training Random Forest for CO2e ...")

rf_co2 = RandomForestRegressor(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    random_state     = RANDOM_STATE,
    n_jobs           = -1
)
rf_co2.fit(X_train, y_co2_train)

cv_r2_co2 = cross_val_score(rf_co2, X_train, y_co2_train, cv=kf, scoring='r2')
print(f"   CV R2 scores (5-fold) : {cv_r2_co2.round(4)}")
print(f"   CV R2 mean +/- std    : {cv_r2_co2.mean():.4f} +/- {cv_r2_co2.std():.4f}")

co2_pred = rf_co2.predict(X_test)
co2_rmse = np.sqrt(mean_squared_error(y_co2_test, co2_pred))
co2_mae  = mean_absolute_error(y_co2_test, co2_pred)
co2_r2   = r2_score(y_co2_test, co2_pred)

print(f"\n   --- CO2e Test Results ---")
print(f"   RMSE : {co2_rmse:.4f} g/hr")
print(f"   MAE  : {co2_mae:.4f} g/hr")
print(f"   R2   : {co2_r2:.4f}")

# =============================================================================
# 7. FEATURE IMPORTANCE  (Section 5.2 / Figure 4)
# =============================================================================

print("\n[7] Extracting feature importance scores ...")

importances_pue = rf_pue.feature_importances_
feat_imp = pd.DataFrame({
    'Feature'    : FEATURE_COLS,
    'Importance' : importances_pue
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n   Top 10 features by importance (PUE model):")
print(feat_imp.head(10).to_string(index=False))

# =============================================================================
# 8. NEPAL PROJECTIONS  (Section 4.5 / Section 5.3)
# =============================================================================

print("\n[8] Generating Nepal-specific projections ...")

def build_nepal_scenario(ambient_temp, workload_range, df_ref, feature_cols):
    """
    Build a synthetic input DataFrame for a Nepal scenario.
    All features are set to their training-set means except
    Ambient_Temp and Server_Workload, which are set explicitly.
    """
    mean_vals = df_ref[feature_cols].mean().to_dict()
    rows = []
    for wl in workload_range:
        row = mean_vals.copy()
        row['Ambient_Temp']    = ambient_temp
        row['Server_Workload'] = wl
        # Recalculate derived features that depend on workload
        it_pow = wl / 100.0 * P_MAX_IT
        row['Cooling_to_IT_ratio'] = row['Cooling_Power'] / it_pow if it_pow > 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)[feature_cols]

workload_range = np.arange(10, 101, 1)   # 10% to 100% in 1% steps

# Scenario A — Kathmandu annual average (17 degrees C)
scen_a_X   = build_nepal_scenario(TEMP_SCENARIO_A, workload_range, df, FEATURE_COLS)
scen_a_pue = rf_pue.predict(scen_a_X)
scen_a_co2 = rf_co2.predict(scen_a_X)

# Scenario B — Kathmandu peak summer (30 degrees C)
scen_b_X   = build_nepal_scenario(TEMP_SCENARIO_B, workload_range, df, FEATURE_COLS)
scen_b_pue = rf_pue.predict(scen_b_X)
scen_b_co2 = rf_co2.predict(scen_b_X)

# Training dataset mean (24 degrees C) — used as a reference line in Figure 3
scen_m_X   = build_nepal_scenario(23.99, workload_range, df, FEATURE_COLS)
scen_m_pue = rf_pue.predict(scen_m_X)

# Projected annual CO2e at mean workload (65%)
mean_wl_idx = np.argmin(np.abs(workload_range - 65))

annual_co2_scenA  = scen_a_co2[mean_wl_idx] * 8760 / 1000   # kg/year
annual_co2_scenB  = scen_b_co2[mean_wl_idx] * 8760 / 1000   # kg/year
seasonal_penalty  = (annual_co2_scenB - annual_co2_scenA) / annual_co2_scenB * 100

print(f"   Scenario A — PUE at 65% workload : {scen_a_pue[mean_wl_idx]:.3f}")
print(f"   Scenario B — PUE at 65% workload : {scen_b_pue[mean_wl_idx]:.3f}")
print(f"   Annual CO2e — Scenario A         : {annual_co2_scenA:.2f} kg/yr")
print(f"   Annual CO2e — Scenario B         : {annual_co2_scenB:.2f} kg/yr")
print(f"   Seasonal emission penalty        : {seasonal_penalty:.1f}%")

pue_improvement_pct = (scen_a_pue[mean_wl_idx] - 1.8) / scen_a_pue[mean_wl_idx] * 100
print(f"   PUE improvement (to 1.8)         : {pue_improvement_pct:.1f}% CO2e reduction")

# =============================================================================
# 9. FIGURE 1 — DATASET OVERVIEW  (Section 4.1)
# =============================================================================

print("\n[9] Generating Figure 1 — Dataset Overview ...")

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# Panel (a) — PUE distribution
ax1 = fig.add_subplot(gs[0, 0])
pue_clip = df['PUE'].clip(upper=5)
ax1.hist(pue_clip, bins=45, color=C_BLUE, edgecolor='white',
         linewidth=0.4, alpha=0.88)
ax1.axvline(df['PUE'].mean(), color=C_RED, lw=1.8, ls='--',
            label=f'Mean = {df["PUE"].mean():.2f}')
ax1.axvline(1.58, color=C_GREEN, lw=1.8, ls=':', label='Global avg = 1.58')
ax1.set_xlabel('Derived PUE', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('(a) PUE Distribution', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, framealpha=0.85)
ax1.set_facecolor(C_LIGHT)
ax1.tick_params(labelsize=9)

# Panel (b) — Ambient Temperature vs PUE scatter (coloured by cooling power)
ax2 = fig.add_subplot(gs[0, 1])
sc = ax2.scatter(
    df['Ambient_Temp'], df['PUE'].clip(upper=5),
    c=df['Cooling_Power'], cmap='YlOrRd', s=6, alpha=0.55, linewidths=0
)
cb = fig.colorbar(sc, ax=ax2, pad=0.02)
cb.set_label('Cooling Power (kW)', fontsize=8)
cb.ax.tick_params(labelsize=7)
z = np.polyfit(df['Ambient_Temp'], df['PUE'].clip(upper=5), 1)
p = np.poly1d(z)
xs = np.linspace(df['Ambient_Temp'].min(), df['Ambient_Temp'].max(), 100)
ax2.plot(xs, p(xs), color=C_BLUE, lw=1.8, ls='--', label='Trend')
ax2.set_xlabel('Ambient Temperature (degrees C)', fontsize=10)
ax2.set_ylabel('Derived PUE', fontsize=10)
ax2.set_title('(b) Ambient Temp vs PUE', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_facecolor(C_LIGHT)
ax2.tick_params(labelsize=9)

# Panel (c) — Server Workload vs Cooling Power (coloured by chiller usage)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(
    df['Server_Workload'], df['Cooling_Power'],
    c=df['Chiller_Usage'], cmap='Blues', s=6, alpha=0.55, linewidths=0
)
sm = plt.cm.ScalarMappable(
    cmap='Blues',
    norm=plt.Normalize(df['Chiller_Usage'].min(), df['Chiller_Usage'].max())
)
cb2 = fig.colorbar(sm, ax=ax3, pad=0.02)
cb2.set_label('Chiller Usage (%)', fontsize=8)
cb2.ax.tick_params(labelsize=7)
z2 = np.polyfit(df['Server_Workload'], df['Cooling_Power'], 1)
p2 = np.poly1d(z2)
xs2 = np.linspace(df['Server_Workload'].min(), df['Server_Workload'].max(), 100)
ax3.plot(xs2, p2(xs2), color=C_RED, lw=1.8, ls='--', label='Trend')
ax3.set_xlabel('Server Workload (%)', fontsize=10)
ax3.set_ylabel('Cooling Power (kW)', fontsize=10)
ax3.set_title('(c) Workload vs Cooling Power', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.set_facecolor(C_LIGHT)
ax3.tick_params(labelsize=9)

# Panel (d) — Cooling strategy distribution
ax4 = fig.add_subplot(gs[1, 1])
strategy_counts = df['Strategy_Action'].value_counts()
colors_bar = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_GREY]
bars = ax4.bar(
    range(len(strategy_counts)), strategy_counts.values,
    color=colors_bar, edgecolor='white', linewidth=0.5, width=0.65
)
ax4.set_xticks(range(len(strategy_counts)))
ax4.set_xticklabels(
    [s.replace(' ', '\n') for s in strategy_counts.index], fontsize=8
)
ax4.set_ylabel('Record Count', fontsize=10)
ax4.set_title('(d) Cooling Strategy Distribution',
              fontsize=11, fontweight='bold')
for bar, val in zip(bars, strategy_counts.values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(val), ha='center', va='bottom', fontsize=8)
ax4.set_facecolor(C_LIGHT)
ax4.tick_params(labelsize=9)
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle(
    'Figure 1: Dataset Overview — Cold Source Control Dataset (3,498 hourly records)',
    fontsize=11, fontweight='bold', y=0.98
)
path_fig1 = os.path.join(OUTPUT_DIR, 'fig1_dataset_overview.png')
plt.savefig(path_fig1, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved -> {path_fig1}")

# =============================================================================
# 10. FIGURE 2 — TEMPORAL PATTERNS AND PUE DRIVERS  (Section 5.1)
# =============================================================================

print("\n[10] Generating Figure 2 — Temporal Patterns and PUE Drivers ...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.patch.set_facecolor('white')

# Panel (a) — Diurnal PUE pattern by hour
hourly_pue = df.groupby('Hour')['PUE'].agg(['mean', 'std']).reset_index()
axes[0].plot(hourly_pue['Hour'], hourly_pue['mean'],
             color=C_BLUE, lw=2.2, marker='o', ms=4)
axes[0].fill_between(
    hourly_pue['Hour'],
    hourly_pue['mean'] - hourly_pue['std'],
    hourly_pue['mean'] + hourly_pue['std'],
    alpha=0.18, color=C_BLUE, label='+/- 1 std'
)
axes[0].axhline(1.58, color=C_GREEN, lw=1.5, ls=':', label='Global avg 1.58')
axes[0].set_xlabel('Hour of Day', fontsize=10)
axes[0].set_ylabel('Mean PUE', fontsize=10)
axes[0].set_title('(a) Diurnal PUE Pattern', fontsize=11, fontweight='bold')
axes[0].set_xticks(range(0, 24, 3))
axes[0].legend(fontsize=8)
axes[0].set_facecolor(C_LIGHT)
axes[0].tick_params(labelsize=9)

# Panel (b) — Monthly mean CO2e bar chart
monthly_co2 = df.groupby('Month')['CO2e_g'].mean()
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
colors_month = [
    C_RED if v >= monthly_co2.mean() else C_BLUE
    for v in monthly_co2.values
]
bars_m = axes[1].bar(
    month_labels, monthly_co2.values,
    color=colors_month, edgecolor='white', linewidth=0.5, width=0.6
)
axes[1].axhline(monthly_co2.mean(), color='black', lw=1.3, ls='--',
                label=f'Mean = {monthly_co2.mean():.1f} g/hr')
for bar, val in zip(bars_m, monthly_co2.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)
axes[1].set_xlabel('Month', fontsize=10)
axes[1].set_ylabel('Mean CO2e (g/hr)', fontsize=10)
axes[1].set_title('(b) Monthly Mean CO2e\n(Nepal CI = 23 gCO2/kWh)',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].set_facecolor(C_LIGHT)
axes[1].tick_params(labelsize=9)

# Panel (c) — Temperature deviation vs PUE box plots
bins_dev   = [0, 1, 2, 4, 6, 11]
labels_dev = ['0-1', '1-2', '2-4', '4-6', '6+']
df['Dev_Bin'] = pd.cut(df['Temp_Deviation'], bins=bins_dev, labels=labels_dev)
groups = [
    df[df['Dev_Bin'] == b]['PUE'].clip(upper=5).dropna().values
    for b in labels_dev
]
bp = axes[2].boxplot(
    groups, patch_artist=True, widths=0.55,
    medianprops=dict(color=C_RED, lw=2),
    whiskerprops=dict(color=C_BLUE),
    capprops=dict(color=C_BLUE),
    flierprops=dict(marker='.', color=C_BLUE, alpha=0.3, ms=3)
)
for patch in bp['boxes']:
    patch.set_facecolor(C_LIGHT)
    patch.set_edgecolor(C_BLUE)
axes[2].set_xticklabels(labels_dev)
axes[2].set_xlabel('Temperature Deviation (degrees C)', fontsize=10)
axes[2].set_ylabel('PUE', fontsize=10)
axes[2].set_title('(c) Temp. Deviation vs PUE', fontsize=11, fontweight='bold')
axes[2].set_facecolor(C_LIGHT)
axes[2].tick_params(labelsize=9)

fig.suptitle(
    'Figure 2: Temporal Patterns and PUE Drivers',
    fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout()
path_fig2 = os.path.join(OUTPUT_DIR, 'fig2_temporal_pue_drivers.png')
plt.savefig(path_fig2, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved -> {path_fig2}")

# =============================================================================
# 11. FIGURE 3 — NEPAL PROJECTIONS  (Section 5.3)
# =============================================================================

print("\n[11] Generating Figure 3 — Nepal Projections ...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.patch.set_facecolor('white')

# Panel (a) — Projected PUE vs server workload for three temperature scenarios
axes[0].plot(workload_range, scen_a_pue, color=C_BLUE, lw=2.2, ls='-',
             label=f'Kathmandu Annual Avg ({TEMP_SCENARIO_A} C)')
axes[0].plot(workload_range, scen_m_pue, color=C_ORANGE, lw=2.2, ls='--',
             label='Training Dataset Mean (24 C)')
axes[0].plot(workload_range, scen_b_pue, color=C_RED, lw=2.2, ls='-.',
             label=f'Kathmandu Peak Summer ({TEMP_SCENARIO_B} C)')
axes[0].axhline(1.58, color=C_GREEN, lw=1.5, ls=':', label='Global avg PUE 1.58')
axes[0].set_xlabel('Server Workload (%)', fontsize=10)
axes[0].set_ylabel('Projected PUE', fontsize=10)
axes[0].set_title('(a) Projected PUE vs Workload\n(Nepal Scenarios)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].set_facecolor(C_LIGHT)
axes[0].tick_params(labelsize=9)
axes[0].set_xlim(10, 100)

# Panel (b) — Annual CO2e comparison across four PUE levels
scenario_labels = [
    'Global\nBest\n(PUE 1.2)',
    'Nepal\nWinter\n(PUE 2.0)',
    'Nepal\nAnnual\n(PUE 2.3)',
    'Nepal\nSummer\n(PUE 2.7)'
]
pue_levels           = [1.2, 2.0, 2.3, 2.7]
ref_it_kwh           = 8760   # kWh/yr per reference IT kW
annual_co2_scenarios = [
    pue * ref_it_kwh * CI_NEPAL / 1000
    for pue in pue_levels
]                              # kg CO2e per year
colors_scen = [C_GREEN, C_BLUE, C_ORANGE, C_RED]

bars_s = axes[1].bar(
    scenario_labels, annual_co2_scenarios,
    color=colors_scen, edgecolor='white', linewidth=0.5, width=0.6
)
for bar, val in zip(bars_s, annual_co2_scenarios):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f'{val:.0f} kg', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

axes[1].set_ylabel('Annual CO2e (kg/yr)\nper reference IT kW', fontsize=10)
axes[1].set_title('(b) Annual CO2e by PUE Scenario\n(Nepal CI = 23 gCO2/kWh)',
                  fontsize=11, fontweight='bold')
axes[1].set_facecolor(C_LIGHT)
axes[1].tick_params(labelsize=9)

# Arrow showing reduction potential between summer and winter scenarios
axes[1].annotate(
    '', xy=(1, annual_co2_scenarios[1]),
    xytext=(3, annual_co2_scenarios[3]),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.3)
)
diff_pct = (
    (annual_co2_scenarios[3] - annual_co2_scenarios[1])
    / annual_co2_scenarios[3] * 100
)
axes[1].text(
    2, (annual_co2_scenarios[1] + annual_co2_scenarios[3]) / 2 + 15,
    f'{diff_pct:.0f}% reduction\npotential',
    ha='center', fontsize=8, color='black', style='italic'
)

fig.suptitle(
    'Figure 3: Nepal-Specific PUE and Carbon Emission Projections',
    fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout()
path_fig3 = os.path.join(OUTPUT_DIR, 'fig3_nepal_projections.png')
plt.savefig(path_fig3, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved -> {path_fig3}")

# =============================================================================
# 12. FIGURE 4 — FEATURE IMPORTANCE  (Section 5.2)
# =============================================================================

print("\n[12] Generating Figure 4 — Feature Importance ...")

top_n     = 10
top_feats = feat_imp.head(top_n).copy()

colors_fi = []
for imp in top_feats['Importance']:
    if imp > 0.15:
        colors_fi.append(C_RED)
    elif imp > 0.06:
        colors_fi.append(C_BLUE)
    else:
        colors_fi.append(C_GREY)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')

bars_fi = ax.barh(
    top_feats['Feature'][::-1].values,
    top_feats['Importance'][::-1].values,
    color=colors_fi[::-1], edgecolor='white', linewidth=0.4, height=0.6
)
for bar, val in zip(bars_fi, top_feats['Importance'][::-1].values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)

ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)', fontsize=10)
ax.set_title(
    'Figure 4: Random Forest Feature Importance for PUE Prediction\n'
    '(Top 10 features — computed from trained model)',
    fontsize=11, fontweight='bold'
)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=9)
ax.set_xlim(0, top_feats['Importance'].max() * 1.18)

legend_elements = [
    Patch(facecolor=C_RED,  label='Primary drivers (above 15%)'),
    Patch(facecolor=C_BLUE, label='Secondary drivers (6-15%)'),
    Patch(facecolor=C_GREY, label='Minor drivers (below 6%)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
path_fig4 = os.path.join(OUTPUT_DIR, 'fig4_feature_importance.png')
plt.savefig(path_fig4, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved -> {path_fig4}")

# =============================================================================
# 13. SUMMARY RESULTS TABLE
# =============================================================================

print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"""
  Dataset
  -------
  Total records          : {len(df):,}
  Features used          : {len(FEATURE_COLS)}
  Training samples       : {len(X_train):,}
  Testing samples        : {len(X_test):,}

  Model Performance — PUE
  -----------------------
  CV R2 (mean +/- std)   : {cv_r2_pue.mean():.4f} +/- {cv_r2_pue.std():.4f}
  Test RMSE              : {pue_rmse:.4f}
  Test MAE               : {pue_mae:.4f}
  Test R2                : {pue_r2:.4f}

  Model Performance — CO2e
  ------------------------
  CV R2 (mean +/- std)   : {cv_r2_co2.mean():.4f} +/- {cv_r2_co2.std():.4f}
  Test RMSE              : {co2_rmse:.4f} g/hr
  Test MAE               : {co2_mae:.4f} g/hr
  Test R2                : {co2_r2:.4f}

  Top 3 PUE Feature Importances
  ------------------------------
  1. {feat_imp.iloc[0]['Feature']:35s} : {feat_imp.iloc[0]['Importance']:.4f}
  2. {feat_imp.iloc[1]['Feature']:35s} : {feat_imp.iloc[1]['Importance']:.4f}
  3. {feat_imp.iloc[2]['Feature']:35s} : {feat_imp.iloc[2]['Importance']:.4f}

  Nepal Projections (at 65 percent server workload)
  ---------------------------------------------------
  Scenario A — Kathmandu Average ({TEMP_SCENARIO_A} C)
    Projected PUE            : {scen_a_pue[mean_wl_idx]:.3f}
    Projected CO2e           : {scen_a_co2[mean_wl_idx]:.2f} g/hr
    Annual CO2e (per IT kW)  : {annual_co2_scenA:.2f} kg/yr

  Scenario B — Kathmandu Peak Summer ({TEMP_SCENARIO_B} C)
    Projected PUE            : {scen_b_pue[mean_wl_idx]:.3f}
    Projected CO2e           : {scen_b_co2[mean_wl_idx]:.2f} g/hr
    Annual CO2e (per IT kW)  : {annual_co2_scenB:.2f} kg/yr
    Seasonal emission penalty: {seasonal_penalty:.1f} percent

  Figures saved
  -------------
  fig1_dataset_overview.png
  fig2_temporal_pue_drivers.png
  fig3_nepal_projections.png
  fig4_feature_importance.png
""")
print("=" * 65)
print("  Done.")
print("=" * 65)
