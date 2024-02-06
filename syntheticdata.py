import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from tabulate import tabulate
from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel

# Set the random seed
np.random.seed(42)
# Set the sample sizes and sex
sample_size = [100, 300, 600]
sex_values = [0, 1]  # 1: female, 0: Male
# set the age range and std dev
age_mean = 72.5
age_std = 6
# Set the hippo_vol range and std dev
hippo_mean = 3400
hippo_stddev = 545
# Set the gray_vol range and std dev
gray_mean = 575000
gray_std = 52000
# add offset var
offsets = {
    'SITE_A': {'age': 5, 'hippo_vol': -400, 'gray_vol': -70},
    'SITE_C': {'age': -5, 'hippo_vol': 400, 'gray_vol': 200}
}
# Set the correlated coefficients
corr_coeff = 0.7

# Create the correlated matrix
corr_matrix = np.array([[1.0, corr_coeff, corr_coeff],
                       [corr_coeff, 1.0, 0.0],
                       [corr_coeff, 0.0, 1.0]])

# Perform choleskey decomposition to get the lower triangular matrix
lower_cholesky = np.linalg.cholesky(corr_matrix)

sample_sizes = [100, 300, 600]
sites = ['SITE_A', 'SITE_B', 'SITE_C']
#============================================================= Simulate the confounded data ======================================================================================================
site_data_confounded = pd.DataFrame()
for site, sample_size in zip(sites, sample_sizes):
    site_effect = np.random.normal(0, 100, sample_size)
    site_age_mean = age_mean + offsets.get(site, {'age': 0})['age']
    site_hippo_mean = hippo_mean + offsets.get(site, {'hippo_vol': 0})['hippo_vol'] + site_effect
    site_gray_mean = gray_mean + offsets.get(site, {'gray_vol': 0})['gray_vol'] + site_effect

    # Generate the uncorrelated data
    uncorr_data = np.random.normal(size = (sample_size, 3))

    # transform the uncorrelated date to be correlated
    correlated_data = np.dot(uncorr_data, lower_cholesky.T)

    # Add site specific means
    age = correlated_data[:, 0] + site_age_mean
    hippo_vol = correlated_data[:, 1] + site_hippo_mean
    gray_vol = correlated_data[:, 2] + site_gray_mean

    sex = np.random.choice(sex_values, sample_size)

    site_spec_data = pd.DataFrame({
        'site': [site] * sample_size,
        'age': age,
        'sex': sex,
        'H_V': hippo_vol,
        'G_V': gray_vol
    })
    site_data_confounded = site_data_confounded.append(site_spec_data, ignore_index=True)

#print(site_data_confounded.head())
# Introducing confounding effect of age on site and volume
age_site_effect = {'SITE_A': -2, 'SITE_B': 0, 'SITE_C': 2}
age_hippo_vol_effect = -0.1
age_gray_vol_effect = -0.2

for site in site_data_confounded['site'].unique():
    site_indices = site_data_confounded[site_data_confounded['site'] == site].index
    site_data_confounded.loc[site_indices, 'age'] += age_site_effect[site]
    site_data_confounded.loc[site_indices, 'H_V'] += site_data_confounded.loc[
                                                         site_indices, 'age'] * age_hippo_vol_effect
    site_data_confounded.loc[site_indices, 'G_V'] += site_data_confounded.loc[site_indices, 'age'] * age_gray_vol_effect

print(site_data_confounded.describe())
# print(site_data_nonconfounded.describe())

#================================================================= Extract data for ANOVA from confounded data ======================================================================================
hippo_vol_data_confounded = []
gray_vol_data_confounded = []

for site in site_data_confounded['site'].unique():
    site_subset = site_data_confounded[site_data_confounded['site'] == site]
    hippo_vol_data_confounded.append(site_subset['H_V'])
    gray_vol_data_confounded.append(site_subset['G_V'])

#==================================================================== Perform ANOVA for hippocampal volume (confounded data) =========================================================================
hippo_vol_anova_non_confounded = f_oneway(*hippo_vol_data_confounded)
print("ANOVA results for Hippo Volume (Confounded Data):")
print("F-statistic:", hippo_vol_anova_non_confounded.statistic)
print("p-value:", hippo_vol_anova_non_confounded.pvalue)

#=================================================================== Perform ANOVA for gray matter volume (confounded data) ===========================================================================
gray_vol_anova_non_confounded = f_oneway(*gray_vol_data_confounded)
print("\nANOVA results for Gray Matter Volume (Confounded Data):")
print("F-statistic:", gray_vol_anova_non_confounded.statistic)
print("p-value:", gray_vol_anova_non_confounded.pvalue)

g = sns.kdeplot(x='G_V', hue='site', data=site_data_confounded, fill=True)
g.axes.set_xlabel('Gray Matter Volume', fontsize=15)
g.axes.set_ylabel('Denisty', fontsize=15)
g.axes.set_title('Gray Matter Volume distribution across sites', fontsize=20)
plt.show()

#==================================================================== Get the data ready for harmonizationLearn ========================================================================================
covars = site_data_confounded[['site', 'age']]
covars = covars.rename(columns={'site': 'SITE'})
data = site_data_confounded['H_V'].values.reshape(-1, 1)
hv_model, adjusted_hv = harmonizationLearn(data, covars, eb=False)
site_data_confounded['adjusted_HV'] = adjusted_hv
gv_data = site_data_confounded['G_V'].values.reshape(-1, 1)
gv_model, adjusted_gv = harmonizationLearn(gv_data, covars, eb=False)
site_data_confounded['adjusted_GV'] = adjusted_gv

#=========================================================================== Plotting the before and after harmonization box plots ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data=site_data_confounded, x='site', y='H_V', autorange=True, ax=axes[0],
            meanline=True, showmeans=True, meanprops={'color': 'yellow'})
axes[0].set_ylabel('Hippocampus volume', fontsize=15)
axes[0].set_xlabel('Site', fontsize=15)
axes[0].set_title('Without Harmonization', fontsize=20, style='oblique')
# plt.show()

sns.boxplot(data=site_data_confounded, x='site', y='adjusted_HV', autorange=True, ax=axes[1],
            meanline=True, showmeans=True, meanprops={'color': 'yellow'})
# sns.stripplot(x=site_data['siteid'], y=site_data['adjusted_gmvol'], size=4, color='.3', linewidth=0, ax=axes[1])
axes[1].set_ylabel('Adjusted Hippocampus volume', fontsize=15)
axes[1].set_xlabel('Site', fontsize=15)
axes[1].set_title('With Harmonization', fontsize=20, style='oblique')
plt.tight_layout()
plt.show()

#======================================================================== Plotting the before and after harmonization box plots =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data=site_data_confounded, x='site', y='G_V', autorange=True, ax=axes[0],
            meanline=True, showmeans=True, meanprops={'color': 'yellow'})
axes[0].set_title('Without Harmonization', fontsize=20, style='oblique')
axes[0].set_ylabel("Gray matter Volume")
axes[0].set_xlabel('Site', fontsize=15)
# axes[0].set_show()

sns.boxplot(data=site_data_confounded, x='site', y='adjusted_GV', autorange=True, ax=axes[1],
            meanline=True, showmeans=True, meanprops={'color': 'yellow'})
# sns.stripplot(x=site_data['siteid'], y=site_data['adjusted_gmvol'], size=4, color='.3', linewidth=0, ax=axes[1])
axes[1].set_title('With Harmonization', fontsize=20, style='oblique')
axes[1].set_ylabel('Adjusted Gray matter volume', fontsize=15)
axes[1].set_xlabel('Site', fontsize=15)
plt.tight_layout()
plt.show()

grouped_data = site_data_confounded.groupby('site').agg(
    {'G_V': ['mean', 'var', 'median'], 'adjusted_GV': ['mean', 'var', 'median'], 'H_V': ['mean', 'var', 'median'],
     'adjusted_HV': ['mean', 'var', 'median'], 'age': ['mean', 'var', 'median']})
grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
desc_table = tabulate(grouped_data, headers='keys', tablefmt='psql')
print(desc_table)

#======================================================================= Distribution plots for hippocampus volume before and after harmonization ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.kdeplot(x='adjusted_HV', hue='site', data=site_data_confounded, fill=True, ax=axes[0])
axes[0].set_xlabel('Adjusted Hippocampus Volume', fontsize=12)
axes[0].set_ylabel('Denisty', fontsize=12)
axes[0].set_title('HippocampusVolume distribution across sites after Combat', fontsize=15)

sns.kdeplot(x='adjusted_GV', hue='site', data=site_data_confounded, fill=True, ax=axes[1])
axes[1].set_xlabel('Adjusted Gray Volume', fontsize=12)
axes[1].set_ylabel('Denisty', fontsize=12)
axes[1].set_title('Gray matter volume distribution across sites after applying Combat', fontsize=15)
plt.tight_layout()
plt.show()

#========================================================================== Extract data for ANOVA from non-confounded data ==================================================================================
adj_hippo_vol_data_confounded = []
adj_gray_vol_data_confounded = []

for site in site_data_confounded['site'].unique():
    site_subset = site_data_confounded[site_data_confounded['site'] == site]
    adj_hippo_vol_data_confounded.append(site_subset['adjusted_HV'])
    adj_gray_vol_data_confounded.append(site_subset['adjusted_GV'])

#============================================================================ Perform ANOVA for hippocampal volume (confounded data) ==========================================================================
hippo_vol_anova_non_confounded = f_oneway(*adj_hippo_vol_data_confounded)
print("ANOVA results for Adjusted Hippo Volume (Confounded Data):")
print("F-statistic:", hippo_vol_anova_non_confounded.statistic)
print("p-value:", hippo_vol_anova_non_confounded.pvalue)

#============================================================================== Perform ANOVA for gray matter volume (confounded data) ========================================================================
gray_vol_anova_non_confounded = f_oneway(*adj_gray_vol_data_confounded)
print("\nANOVA results for Adjusted Gray Matter Volume (Confounded Data):")
print("F-statistic:", gray_vol_anova_non_confounded.statistic)
print("p-value:", gray_vol_anova_non_confounded.pvalue)
