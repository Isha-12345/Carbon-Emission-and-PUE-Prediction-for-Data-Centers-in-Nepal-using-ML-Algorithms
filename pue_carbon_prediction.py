# =============================================================================
# Carbon Emission and PUE Prediction for Data Centers in Nepal
# Algorithms: Random Forest | XGBoost | Support Vector Regression (SVR)
#
# Author  : Isha Adhikari
# Dataset : Data Center Cold Source Control Dataset (Kaggle)
# Source  : https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset




import warnings
warnings.filterwarnings('ignore')

import os
import time
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATASET_PATH     = 'cold_source_control_dataset.csv'
OUTPUT_DIR       = '.'
RANDOM_STATE     = 42
TEST_SIZE        = 0.20       # 80/20 split
N_FOLDS          = 5
P_MAX_IT         = 1.0        # reference peak IT power (kW)
CI_NEPAL         = 23         # Nepal grid carbon intensity (gCO2/kWh)
TEMP_SCENARIO_A  = 17.0       # Kathmandu annual mean ambient temp (°C)
TEMP_SCENARIO_B  = 30.0       # Kathmandu peak summer temp (°C)

# Colour palette
C_BLUE   = '#1a6faf'
C_RED    = '#c0392b'
C_GREEN  = '#1e8449'
C_ORANGE = '#d35400'
C_GREY   = '#7f8c8d'
C_PURPLE = '#6c3483'
C_LIGHT  = '#eaf3fb'

ALG_COLORS = {
    'Random Forest': C_BLUE,
    'XGBoost':       C_ORANGE,
    'SVR':           C_PURPLE,
}

DPI = 220   # resolution for all saved figures


def save(fig, filename):
    """Save a figure and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {filename}')


# =============================================================================
#  LOAD DATASET
# =============================================================================

print('=' * 60)
print('SECTION 1: Loading dataset')
print('=' * 60)

df = pd.read_csv(DATASET_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df.columns = [
    'Timestamp', 'Server_Workload', 'Inlet_Temp', 'Outlet_Temp',
    'Ambient_Temp', 'Cooling_Power', 'Chiller_Usage', 'AHU_Usage',
    'Energy_Cost', 'Temp_Deviation', 'Strategy_Action', 'Output',
]

print(f'  Records   : {len(df):,}')
print(f'  Date range: {df["Timestamp"].min().date()} to {df["Timestamp"].max().date()}')
print()

# =============================================================================
#  DERIVE TARGET VARIABLES (PUE and CO2e)
# =============================================================================

print('=' * 60)
print('SECTION 2: Deriving PUE and CO2e')
print('=' * 60)

df['IT_Power']    = (df['Server_Workload'] / 100.0) * P_MAX_IT
df['Total_Power'] = df['IT_Power'] + df['Cooling_Power']
df['PUE']         = np.where(
    df['IT_Power'] > 0, df['Total_Power'] / df['IT_Power'], np.nan
)
df['CO2e_g'] = df['Total_Power'] * CI_NEPAL  # g CO2e per hour

print(f'  PUE  — min={df["PUE"].min():.3f}  '
      f'mean={df["PUE"].mean():.3f}  '
      f'max={df["PUE"].max():.3f}')
print(f'  CO2e — min={df["CO2e_g"].min():.2f}  '
      f'mean={df["CO2e_g"].mean():.2f}  '
      f'max={df["CO2e_g"].max():.2f} g/hr')
print()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print('=' * 60)
print('SECTION 3: Feature engineering')
print('=' * 60)

df['Hour']        = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Month']       = df['Timestamp'].dt.month
df['Is_Weekend']  = (df['Day_of_Week'] >= 5).astype(int)

df['DeltaT_ratio']        = df['Outlet_Temp'] / df['Inlet_Temp'].replace(0, np.nan)
df['Cooling_to_IT_ratio'] = df['Cooling_Power'] / df['IT_Power'].replace(0, np.nan)

df = df.sort_values('Timestamp').reset_index(drop=True)
df['Cooling_Power_lag1']    = df['Cooling_Power'].shift(1)
df['Server_Workload_lag1']  = df['Server_Workload'].shift(1)
df['Cooling_Power_roll3']   = df['Cooling_Power'].rolling(window=3).mean()
df['Server_Workload_roll3'] = df['Server_Workload'].rolling(window=3).mean()

strat_dummies = pd.get_dummies(df['Strategy_Action'], prefix='Strategy')
df = pd.concat([df, strat_dummies], axis=1)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURE_COLS = [
    'Server_Workload', 'Inlet_Temp', 'Outlet_Temp', 'Ambient_Temp',
    'Cooling_Power', 'Chiller_Usage', 'AHU_Usage', 'Energy_Cost',
    'Temp_Deviation', 'Hour', 'Day_of_Week', 'Month', 'Is_Weekend',
    'DeltaT_ratio', 'Cooling_to_IT_ratio',
    'Cooling_Power_lag1', 'Server_Workload_lag1',
    'Cooling_Power_roll3', 'Server_Workload_roll3',
] + [c for c in df.columns if c.startswith('Strategy_') and c != 'Strategy_Action']

print(f'  Rows after engineering : {len(df):,}')
print(f'  Total features         : {len(FEATURE_COLS)}')
print()

X     = df[FEATURE_COLS].astype(float)
y_pue = df['PUE']
y_co2 = df['CO2e_g']

# =============================================================================
# TRAIN / TEST SPLIT (80/20)
# =============================================================================

print('=' * 60)
print('SECTION 4: Train-test split 80/20')
print('=' * 60)

X_train, X_test, y_pue_train, y_pue_test = train_test_split(
    X, y_pue, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f'  Training samples : {len(X_train):,}')
print(f'  Testing  samples : {len(X_test):,}')
print()

# =============================================================================
#  MODEL DEFINITIONS
# =============================================================================

# Random Forest – 200 trees, min 2 samples per leaf
rf_pue = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_co2 = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)

# XGBoost – 300 rounds, learning rate 0.05, max depth 6
xgb_pue = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
xgb_co2 = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)

# SVR – RBF kernel inside a StandardScaler Pipeline (prevents data leakage)
svr_pue = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.05, gamma='scale'))
])
svr_co2 = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.05, gamma='scale'))
])

# =============================================================================
# TRAIN AND EVALUATE ALL MODELS
# =============================================================================

print('=' * 60)
print('SECTION 6: Training and evaluating all models')
print('=' * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def train_evaluate(model, X_tr, y_tr, X_te, y_te, name, target):
    """Fit model, run 5-fold CV, evaluate on test set."""
    print(f'\n  [{name}] target={target}')
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)

    print(f'    CV R2 : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')
    print(f'    RMSE  : {rmse:.4f}   MAE: {mae:.4f}   R2: {r2:.4f}')
    print(f'    Time  : {elapsed:.1f}s')

    return {
        'model':      model,
        'preds':      preds,
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std':  cv_r2.std(),
        'rmse':       rmse,
        'mae':        mae,
        'r2':         r2,
        'train_time': elapsed,
    }


print('\n  ─── PUE Models ───')
results_pue = {
    'Random Forest': train_evaluate(rf_pue, X_train, y_pue_train, X_test, y_pue_test, 'Random Forest', 'PUE'),
    'XGBoost':       train_evaluate(xgb_pue, X_train, y_pue_train, X_test, y_pue_test, 'XGBoost',       'PUE'),
    'SVR':           train_evaluate(svr_pue, X_train, y_pue_train, X_test, y_pue_test, 'SVR',           'PUE'),
}

print('\n  ─── CO2e Models ───')
results_co2 = {
    'Random Forest': train_evaluate(rf_co2, X_train, y_co2_train, X_test, y_co2_test, 'Random Forest', 'CO2e'),
    'XGBoost':       train_evaluate(xgb_co2, X_train, y_co2_train, X_test, y_co2_test, 'XGBoost',       'CO2e'),
    'SVR':           train_evaluate(svr_co2, X_train, y_co2_train, X_test, y_co2_test, 'SVR',           'CO2e'),
}
print()

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print('=' * 60)
print('SECTION 7: Feature importance')
print('=' * 60)

fi_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'RF':      rf_pue.feature_importances_,
    'XGBoost': xgb_pue.feature_importances_,
}).sort_values('RF', ascending=False).reset_index(drop=True)

print(fi_df.head(10)[['Feature', 'RF', 'XGBoost']].to_string(index=False))
print()

# Readable display labels for feature names
FEAT_LABELS = {
    'Cooling_to_IT_ratio':    'Cooling-to-IT Power Ratio',
    'Server_Workload':        'Server Workload (%)',
    'AHU_Usage':              'AHU Usage (%)',
    'Cooling_Power':          'Cooling Power (kW)',
    'Cooling_Power_lag1':     'Cooling Power (lag 1h)',
    'Chiller_Usage':          'Chiller Usage (%)',
    'Outlet_Temp':            'Outlet Temperature (°C)',
    'Temp_Deviation':         'Temp. Deviation (°C)',
    'Ambient_Temp':           'Ambient Temperature (°C)',
    'DeltaT_ratio':           'ΔT Ratio (outlet/inlet)',
    'Inlet_Temp':             'Inlet Temperature (°C)',
    'Energy_Cost':            'Total Energy Cost ($)',
    'Server_Workload_lag1':   'Server Workload (lag 1h)',
    'Cooling_Power_roll3':    'Cooling Power (3h avg)',
    'Server_Workload_roll3':  'Server Workload (3h avg)',
    'Hour':                   'Hour of Day',
    'Month':                  'Month',
    'Day_of_Week':            'Day of Week',
    'Is_Weekend':             'Weekend Indicator',
}


def feat_label(name):
    return FEAT_LABELS.get(name, name)


# =============================================================================
# NEPAL PROJECTIONS
# =============================================================================

print('=' * 60)
print('SECTION 8: Nepal projections')
print('=' * 60)


def build_scenario(ambient_temp, workload_range, df_ref, feature_cols):
    """Build scenario rows for Nepal projection."""
    base = df_ref[feature_cols].astype(float).mean().to_dict()
    rows = []
    for wl in workload_range:
        row = base.copy()
        row['Ambient_Temp']           = ambient_temp
        row['Server_Workload']        = wl
        it_pow                        = (wl / 100.0) * P_MAX_IT
        row['Cooling_to_IT_ratio']    = row['Cooling_Power'] / it_pow if it_pow > 0 else 0
        row['Server_Workload_lag1']   = wl   # consistent steady-state assumption
        row['Server_Workload_roll3']  = wl
        rows.append(row)
    return pd.DataFrame(rows)[feature_cols].astype(float)


workload_range = np.arange(10, 101, 1)
mean_wl_idx    = np.argmin(np.abs(workload_range - 65))

proj_pue, proj_co2 = {}, {}
for name in ['Random Forest', 'XGBoost', 'SVR']:
    sc_a = build_scenario(TEMP_SCENARIO_A, workload_range, df, FEATURE_COLS)
    sc_b = build_scenario(TEMP_SCENARIO_B, workload_range, df, FEATURE_COLS)
    proj_pue[name] = {
        'A': results_pue[name]['model'].predict(sc_a),
        'B': results_pue[name]['model'].predict(sc_b),
    }
    proj_co2[name] = {
        'A': results_co2[name]['model'].predict(sc_a),
        'B': results_co2[name]['model'].predict(sc_b),
    }

print('  Projections at 65% server workload:')
for name in ['Random Forest', 'XGBoost', 'SVR']:
    pue_a = proj_pue[name]['A'][mean_wl_idx]
    pue_b = proj_pue[name]['B'][mean_wl_idx]
    co2_a = proj_co2[name]['A'][mean_wl_idx] * 8760 / 1000
    co2_b = proj_co2[name]['B'][mean_wl_idx] * 8760 / 1000
    print(f'  {name}: ScenA PUE={pue_a:.3f} CO2e={co2_a:.2f} kg/yr | '
          f'ScenB PUE={pue_b:.3f} CO2e={co2_b:.2f} kg/yr')
print()

# =============================================================================
# GENERATE ALL FIGURES  
# =============================================================================

print('=' * 60)
print('SECTION 9: Generating figures')
print('=' * 60)

ALG_NAMES = ['Random Forest', 'XGBoost', 'SVR']

# ─────────────────────────────────────────────────────────────────────────────
# FIG 01 – PUE Distribution (histogram)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
ax.hist(df['PUE'].clip(upper=5), bins=45, color=C_BLUE,
        edgecolor='white', linewidth=0.4, alpha=0.88)
ax.axvline(df['PUE'].mean(), color=C_RED, lw=1.8, ls='--',
           label=f'Dataset mean = {df["PUE"].mean():.2f}')
ax.axvline(1.58, color=C_GREEN, lw=1.8, ls=':',
           label='Global average = 1.58')
ax.set_xlabel('Derived PUE', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('PUE Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig01_pue_distribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 02 – Ambient Temperature vs PUE (scatter, coloured by cooling power)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
sc = ax.scatter(df['Ambient_Temp'], df['PUE'].clip(upper=5),
                c=df['Cooling_Power'], cmap='YlOrRd',
                s=6, alpha=0.55, linewidths=0)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('Cooling Power (kW)', fontsize=9)
z  = np.polyfit(df['Ambient_Temp'], df['PUE'].clip(upper=5), 1)
xs = np.linspace(df['Ambient_Temp'].min(), df['Ambient_Temp'].max(), 100)
ax.plot(xs, np.poly1d(z)(xs), color=C_BLUE, lw=1.8, ls='--', label='Trend')
ax.set_xlabel('Ambient Temperature (°C)', fontsize=11)
ax.set_ylabel('Derived PUE', fontsize=11)
ax.set_title('Ambient Temperature vs PUE', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig02_ambient_vs_pue.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 03 – Server Workload vs Cooling Power (scatter, coloured by chiller)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
ax.scatter(df['Server_Workload'], df['Cooling_Power'],
           c=df['Chiller_Usage'], cmap='Blues',
           s=6, alpha=0.55, linewidths=0)
sm  = plt.cm.ScalarMappable(
    cmap='Blues',
    norm=plt.Normalize(df['Chiller_Usage'].min(), df['Chiller_Usage'].max())
)
cb2 = fig.colorbar(sm, ax=ax, pad=0.02)
cb2.set_label('Chiller Usage (%)', fontsize=9)
z2  = np.polyfit(df['Server_Workload'], df['Cooling_Power'], 1)
xs2 = np.linspace(10, 100, 100)
ax.plot(xs2, np.poly1d(z2)(xs2), color=C_RED, lw=1.8, ls='--', label='Trend')
ax.set_xlabel('Server Workload (%)', fontsize=11)
ax.set_ylabel('Cooling Power (kW)', fontsize=11)
ax.set_title('Server Workload vs Cooling Power', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig03_workload_vs_cooling.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 04 – Cooling Strategy Distribution (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
strat_counts = df['Output'].value_counts().sort_index()
strat_names  = {0: 'Increase\nChiller', 1: 'Reduce\nAHU',
                2: 'Maintain', 3: 'Boost All', 4: 'Eco Mode'}
labels_d     = [strat_names.get(i, str(i)) for i in strat_counts.index]
colors_d     = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_GREY]
bars_d       = ax.bar(range(len(strat_counts)), strat_counts.values,
                      color=colors_d, edgecolor='white', linewidth=0.5, width=0.65)
ax.set_xticks(range(len(strat_counts)))
ax.set_xticklabels(labels_d, fontsize=9)
ax.set_ylabel('Record Count', fontsize=11)
ax.set_title('Cooling Strategy Action Distribution', fontsize=12, fontweight='bold')
for bar, val in zip(bars_d, strat_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4, str(val),
            ha='center', va='bottom', fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
save(fig, 'fig04_strategy_distribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 05 – Diurnal PUE Pattern (mean PUE by hour of day)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('white')
df['Hour_col'] = df['Timestamp'].dt.hour
hourly_pue = df.groupby('Hour_col')['PUE'].agg(['mean', 'std']).reset_index()
ax.plot(hourly_pue['Hour_col'], hourly_pue['mean'],
        color=C_BLUE, lw=2.2, marker='o', ms=4)
ax.fill_between(hourly_pue['Hour_col'],
                hourly_pue['mean'] - hourly_pue['std'],
                hourly_pue['mean'] + hourly_pue['std'],
                alpha=0.18, color=C_BLUE, label='±1 std')
ax.axhline(1.58, color=C_GREEN, lw=1.5, ls=':', label='Global avg PUE 1.58')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Mean PUE', fontsize=11)
ax.set_title('Diurnal PUE Pattern', fontsize=12, fontweight='bold')
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig05_diurnal_pue.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 06 – Monthly Mean CO2e (Nepal CI)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
df['Month_col']  = df['Timestamp'].dt.month
df['CO2e_Nepal'] = df['Total_Power'] * CI_NEPAL
monthly_co2      = df.groupby('Month_col')['CO2e_Nepal'].mean()
month_labels     = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May'}
mlabels          = [month_labels.get(m, str(m)) for m in monthly_co2.index]
colors_m         = [C_BLUE if v < monthly_co2.mean() else C_RED
                    for v in monthly_co2.values]
bars_m = ax.bar(mlabels, monthly_co2.values, color=colors_m,
                edgecolor='white', linewidth=0.5, width=0.6)
ax.axhline(monthly_co2.mean(), color='black', lw=1.3, ls='--',
           label=f'Mean = {monthly_co2.mean():.1f} g/hr')
for bar, val in zip(bars_m, monthly_co2.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Mean CO₂e (g/hr)', fontsize=11)
ax.set_title('Monthly Mean CO₂e  (Nepal CI = 23 gCO₂/kWh)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig06_monthly_co2e.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 07 – Temperature Deviation vs PUE (box plot)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
df['Dev_Bin'] = pd.cut(df['Temp_Deviation'],
                        bins=[0, 1, 2, 4, 6, 11],
                        labels=['0–1', '1–2', '2–4', '4–6', '6+'])
groups = [df[df['Dev_Bin'] == b]['PUE'].clip(upper=5).dropna().values
          for b in ['0–1', '1–2', '2–4', '4–6', '6+']]
bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                medianprops=dict(color=C_RED, lw=2),
                whiskerprops=dict(color=C_BLUE),
                capprops=dict(color=C_BLUE),
                flierprops=dict(marker='.', color=C_BLUE, alpha=0.3, ms=3))
for patch in bp['boxes']:
    patch.set_facecolor(C_LIGHT)
    patch.set_edgecolor(C_BLUE)
ax.set_xticklabels(['0–1', '1–2', '2–4', '4–6', '6+'], fontsize=10)
ax.set_xlabel('Temperature Deviation (°C)', fontsize=11)
ax.set_ylabel('PUE', fontsize=11)
ax.set_title('Temperature Deviation vs PUE', fontsize=12, fontweight='bold')
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
plt.tight_layout()
save(fig, 'fig07_tempdev_vs_pue.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 08 – RF Feature Importance (MDI, top 10, horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
top10     = fi_df.head(10).copy()
imp_vals  = top10['RF'].values[::-1]
feat_lbls = [feat_label(f) for f in top10['Feature'].values[::-1]]
colors_fi = [C_RED if v > 0.15 else (C_BLUE if v > 0.06 else C_GREY)
             for v in imp_vals]
bars_fi = ax.barh(feat_lbls, imp_vals, color=colors_fi,
                  edgecolor='white', linewidth=0.4, height=0.6)
for bar, val in zip(bars_fi, imp_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
ax.set_title('Random Forest Feature Importance (PUE Model)',
             fontsize=12, fontweight='bold')
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=9)
ax.set_xlim(0, max(imp_vals) * 1.18)
legend_els = [
    Patch(facecolor=C_RED,  label='Primary  (>15%)'),
    Patch(facecolor=C_BLUE, label='Secondary (6–15%)'),
    Patch(facecolor=C_GREY, label='Minor    (<6%)'),
]
ax.legend(handles=legend_els, fontsize=8, loc='lower right')
plt.tight_layout()
save(fig, 'fig08_rf_feature_importance.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 09 – RF vs XGBoost Feature Importance (grouped horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')
top10_fi  = fi_df.head(10).copy()
feat_rev  = [feat_label(f) for f in top10_fi['Feature'].values[::-1]]
x         = np.arange(len(feat_rev))
width     = 0.38
ax.barh(x - width / 2, top10_fi['RF'].values[::-1],
        width, color=C_BLUE, edgecolor='white', linewidth=0.4, label='Random Forest')
ax.barh(x + width / 2, top10_fi['XGBoost'].values[::-1],
        width, color=C_ORANGE, edgecolor='white', linewidth=0.4, label='XGBoost')
ax.set_yticks(x)
ax.set_yticklabels(feat_rev, fontsize=9)
ax.set_xlabel('Feature Importance Score', fontsize=11)
ax.set_title('Feature Importance: RF vs XGBoost (PUE Model)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=9)
plt.tight_layout()
save(fig, 'fig09_rf_xgb_feature_compare.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIGS 10–15 – Metric comparison bar charts (one per metric per target)
# ─────────────────────────────────────────────────────────────────────────────
bar_colors = [ALG_COLORS[n] for n in ALG_NAMES]

metric_configs = [
    # (filename,     target_label, results_dict, metric_key, y_label,      title)
    ('fig10_pue_rmse_comparison.png',  'PUE',       results_pue, 'rmse', 'RMSE',      'RMSE Comparison — PUE Prediction'),
    ('fig11_pue_mae_comparison.png',   'PUE',       results_pue, 'mae',  'MAE',       'MAE Comparison — PUE Prediction'),
    ('fig12_pue_r2_comparison.png',    'PUE',       results_pue, 'r2',   'R²',        'R² Comparison — PUE Prediction'),
    ('fig13_co2e_rmse_comparison.png', 'CO₂e',      results_co2, 'rmse', 'RMSE (g/hr)', 'RMSE Comparison — CO₂e Prediction'),
    ('fig14_co2e_mae_comparison.png',  'CO₂e',      results_co2, 'mae',  'MAE (g/hr)',  'MAE Comparison — CO₂e Prediction'),
    ('fig15_co2e_r2_comparison.png',   'CO₂e',      results_co2, 'r2',   'R²',          'R² Comparison — CO₂e Prediction'),
]

for fname, target_lbl, results, metric_key, y_label, title in metric_configs:
    vals = [results[n][metric_key] for n in ALG_NAMES]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    bars = ax.bar(ALG_NAMES, vals, color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.012,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    if metric_key == 'r2':
        ax.axhline(0.90, color=C_GREEN, lw=1.3, ls='--', label='R²=0.90 threshold')
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor(C_LIGHT)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    save(fig, fname)

# ─────────────────────────────────────────────────────────────────────────────
# FIGS 16–18 – Predicted vs Actual PUE (one figure per algorithm)
# ─────────────────────────────────────────────────────────────────────────────
pred_configs = [
    ('fig16_pred_actual_rf.png',  'Random Forest'),
    ('fig17_pred_actual_xgb.png', 'XGBoost'),
    ('fig18_pred_actual_svr.png', 'SVR'),
]

for fname, name in pred_configs:
    preds = results_pue[name]['preds']
    r2    = results_pue[name]['r2']
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.scatter(y_pue_test, preds, color=ALG_COLORS[name],
               alpha=0.35, s=8, linewidths=0)
    lim_min = min(float(y_pue_test.min()), float(preds.min())) - 0.1
    lim_max = max(float(y_pue_test.max()), float(preds.max())) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='black', lw=1.5, ls='--', label='Perfect fit')
    ax.set_xlabel('Actual PUE', fontsize=11)
    ax.set_ylabel('Predicted PUE', fontsize=11)
    ax.set_title(f'Predicted vs Actual PUE — {name}\n(R² = {r2:.4f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor(C_LIGHT)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    save(fig, fname)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 19 – Nepal Scenario A — all three models (17°C ambient)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('white')
for name in ALG_NAMES:
    ax.plot(workload_range, proj_pue[name]['A'],
            color=ALG_COLORS[name], lw=2.0, label=name)
ax.axhline(1.58, color=C_GREEN, lw=1.5, ls=':', label='Global avg PUE 1.58')
ax.set_xlabel('Server Workload (%)', fontsize=11)
ax.set_ylabel('Projected PUE', fontsize=11)
ax.set_title(f'Nepal PUE Projection — Scenario A  '
             f'(Kathmandu Annual Avg {TEMP_SCENARIO_A}°C)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
ax.set_xlim(10, 100)
plt.tight_layout()
save(fig, 'fig19_nepal_scenario_a.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 20 – Nepal Scenario B — all three models (30°C ambient)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('white')
for name in ALG_NAMES:
    ax.plot(workload_range, proj_pue[name]['B'],
            color=ALG_COLORS[name], lw=2.0, label=name)
ax.axhline(1.58, color=C_GREEN, lw=1.5, ls=':', label='Global avg PUE 1.58')
ax.set_xlabel('Server Workload (%)', fontsize=11)
ax.set_ylabel('Projected PUE', fontsize=11)
ax.set_title(f'Nepal PUE Projection — Scenario B  '
             f'(Kathmandu Peak Summer {TEMP_SCENARIO_B}°C)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(C_LIGHT)
ax.tick_params(labelsize=10)
ax.set_xlim(10, 100)
plt.tight_layout()
save(fig, 'fig20_nepal_scenario_b.png')

# =============================================================================
# SECTION 10 – SUMMARY REPORT
# =============================================================================

print()
print('=' * 60)
print('FINAL SUMMARY')
print('=' * 60)

print('\n  PUE Model Results (Test Set):')
print(f'  {"Model":<16} {"R2":>8} {"RMSE":>8} {"MAE":>8} {"CV_R2":>8}')
print('  ' + '-' * 46)
for name in ALG_NAMES:
    r = results_pue[name]
    print(f'  {name:<16} {r["r2"]:>8.4f} {r["rmse"]:>8.4f} '
          f'{r["mae"]:>8.4f} {r["cv_r2_mean"]:>8.4f}')

print('\n  CO2e Model Results (Test Set):')
print(f'  {"Model":<16} {"R2":>8} {"RMSE":>8} {"MAE":>8} {"CV_R2":>8}')
print('  ' + '-' * 46)
for name in ALG_NAMES:
    r = results_co2[name]
    print(f'  {name:<16} {r["r2"]:>8.4f} {r["rmse"]:>8.4f} '
          f'{r["mae"]:>8.4f} {r["cv_r2_mean"]:>8.4f}')

print('\n  Nepal Projections (65% workload):')
for name in ALG_NAMES:
    pue_a = proj_pue[name]['A'][mean_wl_idx]
    pue_b = proj_pue[name]['B'][mean_wl_idx]
    co2_a = proj_co2[name]['A'][mean_wl_idx] * 8760 / 1000
    co2_b = proj_co2[name]['B'][mean_wl_idx] * 8760 / 1000
    print(f'  {name}:')
    print(f'    Scenario A (17°C)  PUE={pue_a:.3f}  CO2e={co2_a:.2f} kg/yr')
    print(f'    Scenario B (30°C)  PUE={pue_b:.3f}  CO2e={co2_b:.2f} kg/yr')

print('\n  Figures saved (20 individual PNG files):')
for i, name in enumerate([
    'fig01_pue_distribution.png', 'fig02_ambient_vs_pue.png',
    'fig03_workload_vs_cooling.png', 'fig04_strategy_distribution.png',
    'fig05_diurnal_pue.png', 'fig06_monthly_co2e.png',
    'fig07_tempdev_vs_pue.png', 'fig08_rf_feature_importance.png',
    'fig09_rf_xgb_feature_compare.png', 'fig10_pue_rmse_comparison.png',
    'fig11_pue_mae_comparison.png', 'fig12_pue_r2_comparison.png',
    'fig13_co2e_rmse_comparison.png', 'fig14_co2e_mae_comparison.png',
    'fig15_co2e_r2_comparison.png', 'fig16_pred_actual_rf.png',
    'fig17_pred_actual_xgb.png', 'fig18_pred_actual_svr.png',
    'fig19_nepal_scenario_a.png', 'fig20_nepal_scenario_b.png',
], 1):
    print(f'    {i:02d}. {name}')

print('\nAll done.')

# Figure index
# ─────────────────────────────────────────────────────────────────
#  fig01_pue_distribution.png          PUE histogram
#  fig02_ambient_vs_pue.png            Ambient temp vs PUE scatter
#  fig03_workload_vs_cooling.png       Workload vs cooling power scatter
#  fig04_strategy_distribution.png     Cooling strategy bar chart
#  fig05_diurnal_pue.png               Mean PUE by hour of day
#  fig06_monthly_co2e.png              Monthly mean CO2e
#  fig07_tempdev_vs_pue.png            Temp deviation vs PUE box plot
#  fig08_rf_feature_importance.png     RF feature importance (MDI)
#  fig09_rf_xgb_feature_compare.png    RF vs XGBoost feature importance
#  fig10_pue_rmse_comparison.png       RMSE comparison – PUE models
#  fig11_pue_mae_comparison.png        MAE comparison – PUE models
#  fig12_pue_r2_comparison.png         R2 comparison – PUE models
#  fig13_co2e_rmse_comparison.png      RMSE comparison – CO2e models
#  fig14_co2e_mae_comparison.png       MAE comparison – CO2e models
#  fig15_co2e_r2_comparison.png        R2 comparison – CO2e models
#  fig16_pred_actual_rf.png            Predicted vs actual PUE – RF
#  fig17_pred_actual_xgb.png           Predicted vs actual PUE – XGBoost
#  fig18_pred_actual_svr.png           Predicted vs actual PUE – SVR
#  fig19_nepal_scenario_a.png          Nepal projection – Scenario A (17°C)
#  fig20_nepal_scenario_b.png          Nepal projection – Scenario B (30°C)
# ─────────────────────────────────────────────────────────────────
