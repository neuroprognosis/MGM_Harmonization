**Abstract**
Alzheimer's Disease (AD) is a progressive neurodegenerative condition diagnosed using neuroimaging techniques like magnetic resonance imaging (MRI), revealing brain structural changes. Research on AD often involves multi-site data collection with diverse scanners, introducing site or scanner effects. Our objective is to reduce these effects while preserving data consistency. We applied the Combat harmonization method to DZNE DELCODE cohort data, initially focusing on region of interest (ROI) and modulated gray matter (MGM) data, then incorporating mild cognitive impairment (MCI) diagnosis as a covariate. We evaluated harmonization's impact through statistical analysis and support vector machine (SVM) classification. Our findings underscore the delicate balance between reducing site-related biases and maintaining data's biological and clinical relevance. Comparing three datasets—MGM images, harmonized MGM images, and harmonized MGM images with retained MCI diagnosis—yielded valuable insights. Initially, the model achieved 76\% accuracy, a 73\% F1-score, and an 81\% area under curve (AUC) score on MGM data. After harmonization, accuracy slightly decreased to 73\%, with a F1 score of 70\% and AUC score of 75\%. Interestingly, when preserving MCI diagnosis during harmonization, accuracy and AUC score both returned to 76\% and 81\%, mirroring performance on original MGM images. Our findings highlight Combat's effectiveness in mitigating scanner biases, enhancing data reliability without compromising clinical significance.

What you can find here:
**Harmonization Process:** Detailed methodology on how ROI data and gray matter images are harmonized, taking into account different variables to ensure accurate and reliable analysis.

**GLM Analysis (glm_analysis):** An in-depth look into the GLM analysis performed, including the construction of contrast vectors to generate T and P maps for VBM. This analysis helps in understanding the structural variations in the brain.

**F-map Generation (glm_analysis_2):** A continuation of the GLM analysis, focusing on building a contrast matrix to derive the F-map. The F-map provides insights into the statistical significance of the observed changes in brain structure.

**Synthetic data generation (syntheticdata):** To enhance our analysis and test the efficacy of combatting site effects, we have introduced synthetic data generation. This includes:

Site Effects Introduction: Synthetic data where site-specific effects are deliberately introduced to test the harmonization process.
Age to Volume Relation: Addition of synthetic data that simulates the relationship between age and brain volume, aiding in the analysis of age-related volumetric changes.
