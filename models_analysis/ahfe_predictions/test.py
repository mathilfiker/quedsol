import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import Ridge, Lasso  # use Ridge to reduce overfitting
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import joblib  # to save model
from sklearn.decomposition import PCA

def ensemble_predict(models, X_new):
    """
    Returns:
        mean_pred  : averaged prediction across all models  (shape: N,)
        std_pred   : std across models — use as uncertainty estimate (shape: N,)
        all_preds  : raw predictions from each model        (shape: n_models x N)
    """
    all_preds = np.array([m.predict(X_new) for m in models])
    return all_preds.mean(axis=0), all_preds.std(axis=0), all_preds#

selected_features = ['H_donors','delta_eat','gap','alpha','delta_mu','logP','rotatable_bonds','delta_gap','mol_mass','mbd']
target = "dG"

print('Results on FreeSolv')
df_test = pd.read_csv("freesolv_test.csv")
x_test = df_test[selected_features].values
y_test = df_test[target].values
pipe = joblib.load("../../scripts/ahfe_models/polynomial_ahfe.joblib")
y_pred, std_pred, all_preds = ensemble_predict(pipe, x_test)

mae_svr = mae(y_test, y_pred)
rmse_svr = rmse(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
pearson, _ = pearsonr(y_test,y_pred)
spearman, _ = spearmanr(y_test,y_pred)
kendall, _ = kendalltau(y_test,y_pred)


print(f'Standard deviation {std_pred.mean():.4f}')
print(f'MAE: SVR: {mae_svr}')
print('RMSE: SVR: ',rmse_svr)
print('R^2: SVR: ',r2)
print("Pearson: SVR: ",pearson)
print("Spearman: SVR: ",spearman)
print("Kendall : SVR", kendall)

# PLOT
y_min = min(min(y_test),min(y_pred))
y_max = max(max(y_test),max(y_pred))
x = y_test
#metrics
abs_err = mae(x,y_pred)
err = rmse(x,y_pred)
tau = kendalltau(x, y_pred).statistic
pearson = pearsonr(x, y_pred).statistic
r2 = r2_score(x, y_pred)
spear = spearmanr(x,y_pred).statistic

fig, ax = plt.subplots(figsize=(16, 16))

# Split predictions into confident vs uncertain
confident_mask   = std_pred < 0.5   # threshold is tunable
uncertain_mask   = std_pred >= 0.5

# Scatter plot with error bars - CONFIDENT predictions (circles)
ax.errorbar(x[confident_mask], y_pred[confident_mask], 
            yerr=std_pred[confident_mask],
            fmt='o', color='tab:blue', alpha=0.8, markersize=20,
            label='Confident')

# Scatter plot with error bars - UNCERTAIN predictions (x markers)
ax.errorbar(x[uncertain_mask], y_pred[uncertain_mask], 
            yerr=std_pred[uncertain_mask],
            fmt='x', color='tab:orange', alpha=0.8, markersize=20,
            markeredgewidth=3, label='Uncertain')

x_range = np.linspace(y_min, y_max, 400)

# Plot the diagonal line (y = x)
ax.plot(x_range, x_range, 'k-')
# Shaded confidence regions (example)
ax.fill_between(x_range, x_range-2, x_range+2, color="gray", alpha=0.3)
ax.fill_between(x_range, x_range-1, x_range+1, color="gray", alpha=0.5)

stats_text = (f"MAE = {abs_err:.2f} kcal/mol\n"
              f"RMSE = {err:.2f} kcal/mol\n"
              f"τ = {tau:.2f}\n"
              f"r = {pearson:.2f}\n"
              f"R² = {r2:.2f}\n"
              r"$\rho$" +f"  = {spear:.2f}")
ax.legend(title=stats_text, loc="upper left",title_fontsize=45)
ax.set_ylim(y_min-1,y_max+1)
ax.set_xlim(y_min-1,y_max+1)
# Labels and title
ax.set_xlabel("Experimental ΔG (kcal/mol)",fontsize=45)
ax.set_ylabel("Computed ΔG (kcal/mol)",fontsize=45)
ax.tick_params(axis='x',labelsize=40)
ax.tick_params(axis='y',labelsize=40)
ax.set_title("Evaluation on FreeSolv",fontsize=45)
# Show plot
plt.savefig('results/freesolv_eval.png', bbox_inches='tight',pad_inches = 0)
plt.show()
plt.clf()
mean_pred, std_pred, _ = ensemble_predict(pipe, x_test)

# Split predictions into confident vs uncertain
confident_mask   = std_pred < 0.5   # threshold is tunable
uncertain_mask   = std_pred >= 0.5

mae_confident = np.mean(np.abs(mean_pred[confident_mask] - y_test[confident_mask]))
mae_uncertain = np.mean(np.abs(mean_pred[uncertain_mask] - y_test[uncertain_mask]))

print(f"MAE on confident predictions  ({confident_mask.sum()} molecules): {mae_confident:.4f}")
print(f"MAE on uncertain predictions  ({uncertain_mask.sum()} molecules): {mae_uncertain:.4f}")
# Save CSV for Freesolv Ensemble A
results_df = pd.DataFrame({
    'molecule': df_test['molecule'],
    'real_value': y_test,
    'prediction': y_pred,
    'uncertainty': std_pred,
    'confident': (std_pred < 0.5).astype(int)
})
results_df.to_csv('results/freesolv_eval.csv', index=False)
print("Saved: freesolveval.csv")


print('Results on FlexiSol')
# Error on test set
df_test = pd.read_csv("flexisol_all.csv")
x_test = df_test[selected_features].values
y_test = df_test[target].values

pipe = joblib.load("../../scripts/ahfe_models/polynomial_ahfe.joblib")
y_pred, std_pred, all_preds = ensemble_predict(pipe, x_test)

mae_svr = mae(y_test, y_pred)
rmse_svr = rmse(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
pearson, _ = pearsonr(y_test,y_pred)
spearman, _ = spearmanr(y_test,y_pred)
kendall, _ = kendalltau(y_test,y_pred)
err = y_pred-y_test
np.save('ensemble_err',err)
print(f'Standard deviation {std_pred.mean():.4f}')
print(f'MAE: SVR: {mae_svr}')
print('RMSE: SVR: ',rmse_svr)
print('R^2: SVR: ',r2)
print("Pearson: SVR: ",pearson)
print("Spearman: SVR: ",spearman)
print("Kendall : SVR", kendall)
mean_pred, std_pred, _ = ensemble_predict(pipe, x_test)
# PLOT
y_min = min(min(y_test),min(y_pred))
y_max = max(max(y_test),max(y_pred))

#XGB regression
x = y_test
x_err = 0.6 
#metrics
abs_err = mae(x,y_pred)
err = rmse(x,y_pred)
tau = kendalltau(x, y_pred).statistic
pearson = pearsonr(x, y_pred).statistic
r2 = r2_score(x, y_pred)
spear = spearmanr(x,y_pred).statistic

fig, ax = plt.subplots(figsize=(16, 16))

# Split predictions into confident vs uncertain
confident_mask   = std_pred < 0.6   # threshold is tunable
uncertain_mask   = std_pred >= 0.6

# Scatter plot with error bars - CONFIDENT predictions (circles)
ax.errorbar(x[confident_mask], y_pred[confident_mask], 
            xerr=x_err, yerr=std_pred[confident_mask],
            fmt='o', color='tab:blue', alpha=0.8, markersize=20)

# Scatter plot with error bars - UNCERTAIN predictions (x markers)
ax.errorbar(x[uncertain_mask], y_pred[uncertain_mask], 
            xerr=x_err, yerr=std_pred[uncertain_mask],
            fmt='x', color='tab:orange', alpha=0.8, markersize=20,
            markeredgewidth=3)

x_range = np.linspace(y_min, y_max, 400)

# Plot the diagonal line (y = x)
ax.plot(x_range, x_range, 'k-')
# Shaded confidence regions (example)
ax.fill_between(x_range, x_range-2, x_range+2, color="gray", alpha=0.3)
ax.fill_between(x_range, x_range-1, x_range+1, color="gray", alpha=0.5)

stats_text = (f"MAE = {abs_err:.2f} kcal/mol\n"
              f"RMSE = {err:.2f} kcal/mol\n"
              f"τ = {tau:.2f}\n"
              f"r = {pearson:.2f}\n"
              f"R² = {r2:.2f}\n"
              r"$\rho$" +f"  = {spear:.2f}")
ax.legend(title=stats_text, loc="upper left",title_fontsize=45)
ax.set_ylim(y_min-1,y_max+1)
ax.set_xlim(y_min-1,y_max+1)
# Labels and title
ax.set_xlabel("Experimental ΔG (kcal/mol)",fontsize=45)
ax.set_ylabel("Computed ΔG (kcal/mol)",fontsize=45)
ax.tick_params(axis='x',labelsize=40)
ax.tick_params(axis='y',labelsize=40)
ax.set_title("Evaluation on FlexiSol",fontsize=45)
# Show plot
plt.savefig('results/flexisol_eval.png', bbox_inches='tight',pad_inches = 0)
plt.show()
plt.clf()

# Split predictions into confident vs uncertain
confident_mask   = std_pred < 0.6   # threshold is tunable
uncertain_mask   = std_pred >= 0.6

mae_confident = np.mean(np.abs(mean_pred[confident_mask] - y_test[confident_mask]))
mae_uncertain = np.mean(np.abs(mean_pred[uncertain_mask] - y_test[uncertain_mask]))

print(f"MAE on confident predictions  ({confident_mask.sum()} molecules): {mae_confident:.4f}")
print(f"MAE on uncertain predictions  ({uncertain_mask.sum()} molecules): {mae_uncertain:.4f}")
results_df = pd.DataFrame({
    'molecule': df_test['molecule'],
    'real_value': y_test,
    'prediction': y_pred,
    'uncertainty': std_pred,
    'confident': (std_pred < 0.6).astype(int)
})
results_df.to_csv('results/flexisol_eval.csv', index=False)
print("Saved: flexisol_eval.csv")