import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import OLSModel, SimpleRegressionResults
from nilearn import plotting



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
    """Generate contrast matrix of site combination."""
    # Extract unique sites and their original order
    sites = data['siteid'].unique().tolist()
    num_sites = len(sites)

    # Create a mapping from site to its index in the original order
    site_to_index = {site: i for i, site in enumerate(sites)}
    # Initialize the contrast matrix
    contrast_matrix = np.zeros((num_sites-1, num_sites))

    # Fill in the contrast matrix based on the original order of sites
    for i in range(num_sites):
        if i < num_sites - 1:
            # For all rows except the last one
            current_site_index = site_to_index[sites[i]]
            next_site_index = site_to_index[sites[i + 1]]

            contrast_matrix[i, current_site_index] = -1
            contrast_matrix[i, next_site_index] = 1
        else:
            # For the last row, compare site 1 (index 0) and site 10 (index 9)
            contrast_matrix[i, 0] = 1  # Site 1
            contrast_matrix[i, num_sites - 1] = -1  # Site 10
    print(contrast_matrix)
    return contrast_matrix


def glm_model(data, covariates, contrast_matrix, var_name):
    # Precompute design matrix
    design_matrix = pd.get_dummies(covariates)


    # Fit OLS model
    model = OLSModel(design_matrix)
    fit = model.fit(data)
    results = SimpleRegressionResults(fit)

    # Compute contrast maps and corrections
    corrected_p_values = []
    f_maps = []
    contrast_map = results.Fcontrast(contrast_matrix)
    '''for contrast_name, contrast_vector in contrasts:
        # Compute contrast maps for each contrast
        contrast_maps = results.Fcontrast(contrast_matrix)

        # Store F-maps
        f_map = contrast_maps.F
        f_maps.append((contrast_name, f_map))'''
    f_map = contrast_map.F
    f_maps.append((var_name, f_map))
    return f_maps

def plot(f_maps, mask_path):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    #img_name = "123"

    for var_name, f_values in f_maps:
        img_name = f"{var_name}"

        if np.any(np.isnan(mask_data)):
            mask_data[np.isnan(mask_data)] = 0

        # Create an empty array with the same shape as the mask
        f_map = np.zeros_like(mask_data)

        # Assuming corrected_p is a flattened array of p-values corresponding to non-zero elements of the mask
        non_zero_indices = np.nonzero(mask_data)
        f_map[non_zero_indices] = f_values
        img = nib.Nifti1Image(f_map, mask_img.affine)
        nib.save(img, f"{img_name}.nii.gz")

        view = plotting.view_img(img, bg_img=mask_img, black_bg=True, symmetric_cmap=False, colorbar=True, cmap='cold_hot', title=img_name)
        view.open_in_browser() # Display the image

    return "All plots generated successfully!"


def main_pipeline(data_files, data, mask_path, var_name):
    # 1. Compute masked voxels
    nifti_array = compute_masked_voxels(data_files, mask_path)

    # 2. Generate contrast vector based on user input
    contrast_matrix = generate_general_contrast(data)

    # Using the 'siteid' column as the covariate
    covariates = data['siteid']

    # 3. Fit the GLM model
    f_maps = glm_model(nifti_array, covariates, contrast_matrix, var_name)
    print(f_maps)

    # 4. Plot the effects
    html_view = plot(f_maps, mask_path)

    return html_view


mask_path = 'mask.nii'
data_path = 'my_image_data.csv'
data = pd.read_csv(data_path)
raw_data = data['PATH'].to_list()
har_data = data['Data_Har_Path'].to_list()

# Generate and display results
html_output_raw = main_pipeline(raw_data, data, mask_path, "glm_raw_data_circularsitecomp")
html_output_har = main_pipeline(har_data, data, mask_path, "glm_har_data_circularsitecomp")
print(html_output_raw)
print(html_output_har)
