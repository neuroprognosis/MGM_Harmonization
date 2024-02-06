import os
import csv
import numpy as np
import pandas as pd
from glob import glob
import nibabel as nib
from neuroHarmonize import harmonizationLearn
from neuroHarmonize.harmonizationNIFTI import flattenNIFTIs, createMaskNIFTI


def load(path, delimiter=",", usecols=None, encoding='utf=8'):
    if "full_recent" in path:
        data = pd.read_csv(path, delimiter=delimiter, encoding=encoding,
                           usecols=usecols)
        data['site_labels'] = data['siteid'].map({
            2: 'Berlin-2', 5: 'Bonn', 8: 'Göttingen', 10: 'Magdeburg', 11: 'München-1',
            13: 'Rostock', 14: 'Tübingen', 16: 'München-2', 17: 'Berlin-1', 18: 'Köln'})
        data['diag_labels'] = data['prmdiag'].map({0: 'CN', 1: 'SCD', 2: 'MCI', 5: 'AD', 100: 'ADR'})
        #data['sex_labels'] = data['sex_bin'].map({1: 'female', 0: 'male'})
    else:
        data = pd.read_csv(path, delimiter=delimiter, usecols=usecols)
        if 'Measure:volume' in data.columns:
            mask = data['Measure:volume'].str.contains('-M0')
            M0_data = data[mask]
            M0_data[['Subject', 'temp']] = M0_data['Measure:volume'].str.split('-', expand=True)
            M0_data.drop(columns=['Measure:volume', 'temp'], inplace=True)
            M0_data = M0_data.rename(
                columns={'Left-Hippocampus': 'left_hippocampus', 'Right-Hippocampus': 'right_hippocampus'})
            return M0_data

    return data


def create_brain_images_csv(path, csv_path):
    subject_folders = glob(path + 'M00/*')
    with open(csv_path + 'brain_images_path.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject', 'PATH'])
        for subject_folder in subject_folders:
            subject_id = os.path.basename(subject_folder)
            nifti_images = glob(os.path.join(subject_folder, 'mri', 'mwmwp1*.nii'))
            for image_path in nifti_images:
                writer.writerow([subject_id, image_path])


def merge_and_encode(df1, df2):
    merged_df = pd.merge(df1, df2, on='Subject')
    diag_onehot = pd.get_dummies(merged_df['prmdiag'], prefix='diag')
    sex_onehot = pd.get_dummies(merged_df['sex'], prefix='sex')
    return pd.concat([merged_df, diag_onehot, sex_onehot], axis=1)


def apply_harmonization(data, covariate_columns, harmonize_factors):
    covars = data[covariate_columns]
    label = ""
    if 'siteid' in covariate_columns:
        covars = covars.rename(columns={'siteid': 'SITE'})
        # Exclude 'siteid' from the labeling process
        covariate_columns = [col for col in covariate_columns if col != 'siteid']

    # Generate the label
    for col in covariate_columns:
        label += col[0].upper()
        if col[0].lower() == 'd':  # Check if the first character is 'd'
            label += col[-1]

    # If harmonize_data_or_columns is direct data (e.g., NumPy array or list)
    if isinstance(harmonize_factors, np.ndarray):
        _, adjusted_data = harmonizationLearn(harmonize_factors, covars)
        return data, adjusted_data
    # If harmonize_data_or_columns is list of column names
    elif isinstance(harmonize_factors, list):
        adjusted_volumes = {}
        for column in harmonize_factors:
            data[f'adjusted_{column}{label}'] = np.nan
            volume_data = data[column].values.reshape(-1, 1)
            _, adjusted_data = harmonizationLearn(volume_data, covars, eb=False)
            data[f'adjusted_{column}{label}'] = adjusted_data
        return data

def generate_image_name_and_column(covariate_columns):
    # If it's only 'siteid', then just return the basic name
    if covariate_columns == ['siteid']:
        return 'mwmwp1rt1w_denoised_adjusted.nii', 'Data_Har_Path'
    else:
        # Generate the name based on extra covariates, skipping 'siteid'
        extra_covars = "_".join([covar for covar in covariate_columns if covar != 'siteid'])
        image_name = f'mwmwp1rt1w_denoised_adjusted_{extra_covars}.nii'
        column_name = f'Data_Har_{extra_covars}_Path'
        return image_name, column_name

def adjust_and_save_nifti_images(df, nifti_array_adj, mask_path, save_path, covariate_columns):
    img_mask = nib.load(mask_path).get_fdata().round().astype(int) == 1
    image_name, column_name = generate_image_name_and_column(covariate_columns)
    for i in range(df.shape[0]):
        n_image_i = nib.load(df.PATH[i])
        n_data_i = n_image_i.get_fdata()
        nifti_new = np.zeros_like(n_data_i)
        nifti_new[img_mask] = nifti_array_adj[i]
        subject_name = df.Subject[i]
        subject_folder = os.path.join(save_path, subject_name, 'mri')
        output_path = os.path.join(subject_folder, image_name)
        df.at[i, column_name] = output_path
        os.makedirs(subject_folder, exist_ok=True)
        nifti_img = nib.Nifti1Image(nifti_new, affine=n_image_i.affine, header=n_image_i.header)
        #print('OUTPUT PATH IS:', output_path, 'image_name is:', image_name )
        nib.save(nifti_img, output_path)


# Main Execution
features_path = '/storage/users/tummala/full_recent'
path = '/storage/ASSESSMENT/DELCODE-CAT12Long/features/'
csv_path = '/storage/users/tummala/'
save_path = '/storage/users/tummala/adjusted_M00/'
mask_path = 'mask.nii'

# Read all the needed data
features_cols = ['Subject', 'sex', 'sex_bin', 'siteid', 'age_baseline', 'age_M00', 'prmdiag']
features = load(features_path, delimiter=" ", usecols=features_cols, encoding="latin-1")
#print(features.head())
roi_cols = ['Measure:volume', 'Left-Hippocampus', 'Right-Hippocampus', 'TotalGrayVol']
roi_volume = load(csv_path + 'aseg_stats_2022-07-06_long.csv', delimiter=";", usecols=roi_cols)
create_brain_images_csv(path, csv_path)
nifti_list = pd.read_csv(csv_path + 'brain_images_path.csv')

# prep image dataset for harmonization
#nifti_avg, nifti_mask, affine, hdr0 = createMaskNIFTI(nifti_list, threshold=0)
nifti_array = flattenNIFTIs(nifti_list, mask_path)

# merge data
subject_image_data = merge_and_encode(nifti_list, features)
subject_volume_data = merge_and_encode(roi_volume, features)

# define covras and oolumns to be harmonized
covars = ['siteid']
covars_used = ['siteid', 'diag_5', 'age_baseline']
covars_mci_only = ['siteid','diag_2']
covars_mci = ['siteid', 'diag_2', 'age_baseline']
covars_wsex = ['siteid', 'diag_5', 'sex_f', 'sex_m', 'age_baseline']
harmonize_cols = ['left_hippocampus', 'right_hippocampus', 'TotalGrayVol']

# apply comBat on volumetric data with and without covariates of interest
subject_volume_data = apply_harmonization(subject_volume_data, covars, harmonize_cols)
subject_volume_data = apply_harmonization(subject_volume_data, covars_used, harmonize_cols)
subject_volume_data = apply_harmonization(subject_volume_data, covars_wsex, harmonize_cols)
subject_volume_data = apply_harmonization(subject_volume_data, covars_mci, harmonize_cols)
subject_volume_data = apply_harmonization(subject_volume_data, covars_mci_only, harmonize_cols)

# apply combat on modulated gray matter images with and without covariates of interest
_, nifti_array_adj = apply_harmonization(subject_image_data, covars, nifti_array)
adjust_and_save_nifti_images(subject_image_data, nifti_array_adj, mask_path, save_path, covars)

_, nifti_array_adj = apply_harmonization(subject_image_data, covars_used, nifti_array)
adjust_and_save_nifti_images(subject_image_data, nifti_array_adj, mask_path, save_path, covars_used)

_, nifti_array_adj = apply_harmonization(subject_image_data, covars_wsex, nifti_array)
adjust_and_save_nifti_images(subject_image_data, nifti_array_adj, mask_path, save_path, covars_wsex)

_, nifti_array_adj = apply_harmonization(subject_image_data, covars_mci, nifti_array)
adjust_and_save_nifti_images(subject_image_data, nifti_array_adj, mask_path, save_path, covars_mci)

_, nifti_array_adj = apply_harmonization(subject_image_data, covars_mci_only, nifti_array)
adjust_and_save_nifti_images(subject_image_data, nifti_array_adj, mask_path, save_path, covars_mci_only)

# save the csv files with harmonized data for further processing
subject_image_data.to_csv('my_image_data.csv', index=False)
subject_volume_data.to_csv('my_volume_data.csv', index=False)
