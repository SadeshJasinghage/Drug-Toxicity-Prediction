# Drug Toxicity Profiling and Classification Using Explainable Machine Learning

## Overview
This project leverages machine learning and toxicogenomic data to predict drug-induced toxicity, specifically focusing on liver and kidney toxicity. By utilizing the Open TG-GATEs database, we developed predictive models that classify compound toxicity into four severity levels: minimal, slight, moderate, and severe. The Random Forest model emerged as the most effective, achieving 97.32% accuracy, offering a viable alternative to traditional animal testing methods.

## Project Description
Drug toxicity remains a significant challenge in pharmaceutical development, often leading to clinical trial failures and market withdrawals. Traditional toxicity testing methods, such as animal studies, are not only ethically contentious but also costly and time-consuming. This project addresses these issues by employing machine learning on toxicogenomic data to predict toxicity levels, thereby reducing reliance on animal testing and accelerating drug development.

### Key Objectives
1. **Develop a Computational Model**: Create a model to predict drug toxicity levels based on biochemical and clinical parameters for liver and kidney endpoints.
2. **Data Preprocessing**: Clean, merge, and preprocess the dataset to ensure high-quality input for model training.
3. **Model Training and Evaluation**: Train and evaluate machine learning models (Random Forest, SVM, XGBoost) to identify the best-performing algorithm.
4. **Interpretability**: Ensure the model provides explainable predictions to support regulatory and industrial adoption.

## Dataset
The Open TG-GATEs database was used, which includes:
- **Gene Expression Data**: From rat and human liver and kidney cells exposed to 170 chemical compounds.
- **Toxicological Assessments**: Pathological evaluations, hematology, biochemistry data, and organ/body weight measurements.
- **Experimental Design**: Standardized protocols for in vivo (rat) and in vitro (human and rat hepatocytes) studies.

### Data Preprocessing
1. **Initial Cleaning**: 
   - Merged datasets using a unique ID generated from 'EXP_ID', 'GROUP_ID', 'INDIVIDUAL_ID', and 'organ_label'.
   - Removed duplicates and irrelevant entries (e.g., GRADE_TYPE = 'P').
   - Encoded categorical variables (e.g., 'ORGAN' as binary: 0 for kidney, 1 for liver).

2. **Handling Missing Values**: 
   - Applied mean imputation for numerical columns using Scikit-learn's `SimpleImputer`.

3. **Feature Scaling**: 
   - Standardized numerical features using `StandardScaler` to ensure equal contribution to the model.

### Feature Selection
1. **Categorical Features**: 
   - Used Cramér’s V to measure association with the target variable (GRADE_TYPE).
   - Retained `COMPOUND_NAME` due to its strong association.

2. **Numerical Features**: 
   - Selected the top 16 numerical variables based on Pearson correlation with GRADE_TYPE.

3. **Class Imbalance**: 
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset across toxicity grades.

## Machine Learning Models
Three models were trained and evaluated:
1. **Random Forest**: Achieved the highest accuracy (97.32%) and F1-score (0.97).
2. **Support Vector Machine (SVM)**: Lower accuracy (58.86%) due to sensitivity to feature scaling and class imbalance.
3. **XGBoost**: Competitive performance (76.00%) but slightly inferior to Random Forest.

### Model Performance
- **Random Forest**:
  - **Precision**: 0.98 (Grade 0), 0.97 (Grade 1), 0.91 (Grade 2), 0.96 (Grade 3).
  - **Recall**: High for all classes, with minor challenges in moderate/severe cases due to class imbalance.
  - **Confusion Matrix**: Showed excellent classification accuracy across all toxicity grades.

## Results and Discussion
- **Interpretability**: Random Forest's feature importance provided insights into key predictors (e.g., compound identity, biochemical assays).
- **Clinical Relevance**: Predictions aligned well with histopathological outcomes, particularly for minimal and slight toxicity grades.
- **Challenges**: Moderate and severe toxicity grades had fewer samples, impacting model performance for these classes.

## Conclusion
This project demonstrates the potential of machine learning in computational toxicology, offering accurate and interpretable predictions of drug toxicity. By integrating gene expression and biochemical data, the model reduces reliance on animal testing and supports faster, safer drug development. Future work could expand to multi-omics data and other organ systems to further enhance predictive capabilities.

## Repository Structure
```
├── data/                   # Processed and raw datasets
├── notebooks/              # Jupyter notebooks for data preprocessing and model training
├── models/                 # Trained model files
├── results/                # Evaluation metrics, plots, and reports
├── README.md               # Project overview
```


## Dependencies
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn.

## Acknowledgments
- Dr. U.P. Liyanage for supervision.
- Open TG-GATEs consortium for the dataset.
- Contributors to the Scikit-learn and XGBoost libraries.
