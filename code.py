import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import matplotlib.pyplot as plt


data = pd.read_csv('spinal_bmd_data.csv')
data = data.sort_values('age')
age = data['age'].values
spnbmd = data['spnbmd'].values

# Fitting a cubic smooth spline with cross-validation to find the optimal smoothing parameter
kf = KFold(n_splits=10, shuffle=True, random_state=1)
best_spline = None
best_score = float('inf')

for train_index, test_index in kf.split(age):
    age_train, age_test = age[train_index], age[test_index]
    spnbmd_train, spnbmd_test = spnbmd[train_index], spnbmd[test_index]
    
    spline = UnivariateSpline(age_train, spnbmd_train, k=3)
    # SMOOTHING FACTOR
    smoothing_factor = np.sum((spline(age_train) - spnbmd_train) ** 2) / len(spnbmd_train)
    spline.set_smoothing_factor(smoothing_factor)
    
    test_score = np.mean((spline(age_test) - spnbmd_test) ** 2)
    
    if test_score < best_score:
        best_spline = spline
        best_score = test_score

# Value Prediction 
age_grid = np.linspace(age.min(), age.max(), 500)
spnbmd_pred = best_spline(age_grid)

# Calculate 90% confidence bands
residuals = spnbmd - best_spline(age)
residual_std = np.std(residuals)
conf_band = 1.645 * residual_std * np.sqrt(1 + 1/len(spnbmd))

upper_band = spnbmd_pred + conf_band
lower_band = spnbmd_pred - conf_band

# Computing the posterior mean and covariance using Gaussian Process Regression
kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=residual_std**2)

gp.fit(age.reshape(-1, 1), spnbmd)
y_pred, y_std = gp.predict(age_grid.reshape(-1, 1), return_std=True)
posterior_upper = y_pred + 1.645 * y_std
posterior_lower = y_pred - 1.645 * y_std


n_bootstraps = 100
bootstraps = np.zeros((n_bootstraps, len(age_grid)))

for i in range(n_bootstraps):
    sample_indices = np.random.choice(len(age), len(age), replace=True)
    age_sample = age[sample_indices]
    spnbmd_sample = spnbmd[sample_indices]
    
    sorted_indices = np.argsort(age_sample)
    age_sample = age_sample[sorted_indices]
    spnbmd_sample = spnbmd_sample[sorted_indices]
    
    print(f"Bootstrap iteration {i}:")
    print("age_sample:", age_sample)
    print("spnbmd_sample:", spnbmd_sample)
    
    try:
        bootstrap_spline = UnivariateSpline(age_sample, spnbmd_sample, k=3)
        bootstrap_spline.set_smoothing_factor(len(age_sample) * np.var(spnbmd_sample))
        bootstraps[i] = bootstrap_spline(age_grid)
    except Exception as e:
        print(f"Error in bootstrap iteration {i}: {e}")
        bootstraps[i] = np.nan 

bootstrap_mean = np.nanmean(bootstraps, axis=0)
bootstrap_upper = np.nanpercentile(bootstraps, 95, axis=0)
bootstrap_lower = np.nanpercentile(bootstraps, 5, axis=0)

# Plotting the 90% CI and Posterior Ban
plt.figure(figsize=(12, 8))
plt.plot(age, spnbmd, 'o', label='Data')
plt.plot(age_grid, spnbmd_pred, label='Cubic Spline', color='blue')
plt.fill_between(age_grid, lower_band, upper_band, color='blue', alpha=0.2, label='Spline 90% Confidence Band')
plt.plot(age_grid, y_pred, label='Posterior Mean', color='red')
plt.fill_between(age_grid, posterior_lower, posterior_upper, color='red', alpha=0.2, label='Posterior 90% Confidence Band')
plt.xlabel('Age')
plt.ylabel('Relative Spinal BMD')
plt.legend()
plt.show()

# Plotting the Bootstrap Results
plt.figure(figsize=(12, 8))
plt.plot(age, spnbmd, 'o', label='Data')
plt.plot(age_grid, bootstrap_mean, label='Bootstrap Mean', color='green')
plt.fill_between(age_grid, bootstrap_lower, bootstrap_upper, color='green', alpha=0.2, label='Bootstrap 90% Confidence Band')
plt.xlabel('Age')
plt.ylabel('Relative Spinal BMD')
plt.legend()
plt.show()

#SUMMARIZE RESULTS
print("Spline 90% Confidence Band:")
print("Lower Band:", lower_band)
print("Upper Band:", upper_band)
print("\nPosterior Mean and Covariance:")
print("Posterior Mean:", y_pred)
print("Posterior Covariance (variance):", y_std**2)
print("\nBootstrap 90% Confidence Band:")
print("Bootstrap Mean:", bootstrap_mean)
print("Bootstrap Lower Band:", bootstrap_lower)
print("Bootstrap Upper Band:", bootstrap_upper)
