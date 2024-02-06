import os
from scipy.stats import t, f
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as stats
from nilearn.glm import OLSModel, SimpleRegressionResults
from statsmodels.stats.multitest import multipletests


def compute_masked_voxels(data_files, mask_path):
    if mask_path:
        mask = (nib.load(mask_path).get_fdata().round().astype(int) == 1)
        active_voxels = mask.sum()
        total_images = len(data_files)

        # Initialize array size of total_images x size of flattened images
        nifti_array = np.zeros((total_images, active_voxels))

        # Iterate over images and fill the container
        for i in range(total_images):
            nifti_data = nib.load(data_files[i]).get_fdata()
            nifti_array[i, :] = nifti_data[mask]

        return nifti_array


def generate_general_contrast(data):
    """Generate contrast vectors for each site combination."""
    # Extract unique sites and their original order
    sites = data['site_labels'].unique().tolist()
    num_sites = len(sites)

    # Fill in the contrast matrix for consecutive site comparisons
    contrasts = []
    for i in range(num_sites):
        contrast_vector = np.zeros(num_sites)
        if i < num_sites - 1:
            contrast_vector[i] = 1
            contrast_vector[i + 1] = -1
            contrast_name = f'{sites[i]} vs {sites[i + 1]}'
        else:
            contrast_vector[i] = 1
            contrast_vector[0] = -1
            contrast_name = f'{sites[i]} vs {sites[0]}'
        contrasts.append((contrast_name, contrast_vector))
    print(contrasts)
    return contrasts


def glm_model(data, covariates, contrasts):
    # Precompute design matrix
    design_matrix = pd.get_dummies(covariates)

    # Fit OLS model
    model = OLSModel(design_matrix)
    fit = model.fit(data)
    results = SimpleRegressionResults(fit)

    # Initialize lists to store results
    f_maps = []
    p_values = []

    for contrast_name, contrast_vector in contrasts:
        # Compute t-statistic maps for each contrast
        contrast_maps = results.Fcontrast(contrast_vector)

        # Store t-statistics
        f_map = contrast_maps.F
        f_maps.append((contrast_name, f_map))
        # print(f_map.shape)

        # Compute p-values
        df_num, df_den = contrast_maps.df_num, contrast_maps.df_den
        if np.isscalar(df_den):
            p_value_map = f.sf(f_map, df_num, df_den)  # one-sided p-value
        else:
            # Ensure df_den matches the length of f_map
            df_den_tiled = np.full_like(f_map, df_den[0])
            print(df_den_tiled.shape, df_num, f_map.shape)
            p_value_map = f.sf(f_map, df_num, df_den_tiled)

        p_values.append((contrast_name, p_value_map))

    # Flatten the list of p-values for correction
    all_p_values = [p for _, p_map in p_values for p in p_map.flatten()]

    # Apply multiple comparison correction (e.g., FDR)
    corrected_p_values = multipletests(all_p_values, method='fdr_bh')[1]

    # Reshape corrected p-values back to the original map shape
    corrected_p_value_maps = []
    offset = 0
    for contrast_name, p_map in p_values:
        # shape = p_values[idx][1].shape
        # print("SHAPE", shape)
        num_elements = p_map.size
        print(num_elements)

        # Extract the corresponding corrected p-values and reshape
        corrected_map = corrected_p_values[offset:offset + num_elements].reshape(p_map.shape)
        corrected_p_value_maps.append((contrast_name, corrected_map))
        print(corrected_map)

        # Update the offset
        offset += num_elements
        print(offset)
    return f_maps, p_values, corrected_p_value_maps


def plot(p_maps, mask_path, var_name):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    # img_name = "123"
    dir_name = f"{var_name}_CORRECTED_P_MAPS"
    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for contrast_name, p_values in p_maps:
        img_name = f"{var_name}_{contrast_name}"

        if np.any(np.isnan(mask_data)):
            mask_data[np.isnan(mask_data)] = 0

        # Create an empty array with the same shape as the mask
        p_map = np.zeros_like(mask_data)

        # Assuming corrected_p is a flattened array of p-values corresponding to non-zero elements of the mask
        non_zero_indices = np.nonzero(mask_data)
        p_map[non_zero_indices] = p_values
        img = nib.Nifti1Image(p_map, mask_img.affine)
        nib.save(img, os.path.join(dir_name, f"{img_name}.nii.gz"))

        # view = plotting.view_img(img, bg_img=mask_img, black_bg=True, symmetric_cmap=False, colorbar=True, cmap='cold_hot', title=img_name)
        # view.open_in_browser() # Display the image

    return "All plots generated successfully!"


def main_pipeline(data_files, data, mask_path, var_name):
    # 1. Compute masked voxels
    nifti_array = compute_masked_voxels(data_files, mask_path)

    # 2. Generate contrast vector based on user input
    contrasts = generate_general_contrast(data)

    # Using the 'siteid' column as the covariate
    covariates = data[['site_labels', 'diag_2']]

    # 3. Fit the GLM model
    f_maps, p_maps, corrected_p_maps = glm_model(nifti_array, covariates, contrasts)
    print('p_maps', p_maps)
    print('corrected_p_maps', corrected_p_maps)

    # 4. Plot the effects
    html_view = plot(corrected_p_maps, mask_path, var_name)

    return html_view


mask_path = 'mask.nii'
data_path = 'my_image_data.csv'
data = pd.read_csv(data_path)
har_data = data['Data_Har_diag_2_Path'].to_list()

# Generate and display results
# html_output_raw = main_pipeline(raw_data, data, mask_path, "RAW_DATA")
html_output_har = main_pipeline(har_data, data, mask_path, "HAR_MCI_DATA")
# print(html_output_raw)
print(html_output_har)
