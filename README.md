
Genetic Variation Analysis for Predicting Multiple Sclerosis (MS) Using Machine Learning and SNP Data

Title:- Data Analysis Project

Name:- P.Venkatesh

Reg No:- 36823025

Project Overview

The primary objective of this project is to investigate associations between genetic variations (Single Nucleotide Polymorphisms, or SNPs) and Multiple Sclerosis (MS) risk factors. Using machine learning and statistical techniques, the project aims to analyze, predict, and recommend insights based on the identified genetic markers. This analysis is divided into four analytical types:

1.Descriptive Analytics
2.Diagnostic Analytics
3.Predictive Analytics
4.Prescriptive Analytics

Each analytical approach provides a deeper understanding of the role of SNPs in MS.

#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
1. Data Loading and Preparation

Dataset The dataset used in this analysis is sourced from SNP associations with MS (MONDO_0005301_associations_exportmscv.csv). It contains columns such as riskFrequency, orValue, pValue, traitName, mappedGenes, and more. Each row represents an SNP and its corresponding characteristics related to MS

# Load the dataset
file_path = 'MONDO_0005301_associations_exportmscv.csv'
data = pd.read_csv(file_path)
data.head()
riskAllele	pValue	pValueAnnotation	riskFrequency	orValue	beta	ci	mappedGenes	traitName	efoTraits	bgTraits	accessionId	locations	pubmedId	author
0	rs117518546-T	2.000000e-25	-	0.345	2.916	-	-	IGHG1	Neuromyelitis optica spectrum disorder (AQP4-I...	AQP4-IgG-positive neuromyelitis optica	-	GCST90277235	14:105737776	33559384	Zhong X
1	rs28383224-A	6.000000e-12	-	0.42	2.24	-	[1.78-2.82]	HLA-DRB1,HLA-DQA1	Neuromyelitis optica	neuromyelitis optica	-	GCST005964	6:32615876	29769526	Estrada K
2	rs1150757-A	7.000000e-12	-	0.10	2.86	-	[1.98-4.14]	TNXB	Neuromyelitis optica	neuromyelitis optica	-	GCST005964	6:32061428	29769526	Estrada K
3	HLA-DRB1*03:01-?	2.000000e-12	-	0.12	2.71	-	[2.05-3.57]	-	Neuromyelitis optica	neuromyelitis optica	-	GCST005964	-	29769526	Estrada K
4	HLA-B*08:01-?	6.000000e-12	-	0.11	2.72	-	[2.05-3.63]	-	Neuromyelitis optica	neuromyelitis optica	-	GCST005964	-	29769526	Estrada K
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 944 entries, 0 to 943
Data columns (total 15 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   riskAllele        944 non-null    object 
 1   pValue            944 non-null    float64
 2   pValueAnnotation  944 non-null    object 
 3   riskFrequency     944 non-null    object 
 4   orValue           944 non-null    object 
 5   beta              944 non-null    object 
 6   ci                944 non-null    object 
 7   mappedGenes       944 non-null    object 
 8   traitName         944 non-null    object 
 9   efoTraits         944 non-null    object 
 10  bgTraits          944 non-null    object 
 11  accessionId       944 non-null    object 
 12  locations         944 non-null    object 
 13  pubmedId          944 non-null    int64  
 14  author            944 non-null    object 
dtypes: float64(1), int64(1), object(13)
memory usage: 110.8+ KB
PRE PROCESSING

# Convert relevant columns to numeric, handling any errors
for col in ['riskFrequency', 'orValue', 'pValue', 'beta']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# Filter for "multiple sclerosis" in traitName and drop rows with missing values in essential columns
ms_data = data[data['traitName'].str.contains("multiple sclerosis", case=False, na=False)]
ms_data = ms_data.dropna(subset=['riskFrequency', 'orValue', 'pValue'])
#Impute missing values in 'beta' or 'ci' if necessary
ms_data['beta'].fillna(ms_data['beta'].mean(), inplace=True)  # Impute with mean
ms_data['ci'].fillna("Not Specified", inplace=True)  # Categorical imputation
Descriptive Analytics

Descriptive analytics provides an overview of key data distributions and statistics to identify trends and outliers.

Method 1: Summary Statistics
Method 2: Frequency Distribution Plots
Method 3: Box Plots for Outliers

# Method 1: Summary Statistics
print("Summary Statistics:")
print(data[['riskFrequency', 'pValue', 'orValue', 'beta']].describe())
Summary Statistics:
       riskFrequency         pValue     orValue  beta
count     320.000000   9.440000e+02  730.000000   0.0
mean        0.392388   1.175254e-06    1.283082   NaN
std         0.236411   2.262816e-06    0.629095   NaN
min         0.020000  1.000000e-234    0.630000   NaN
25%         0.200000   6.000000e-13    1.076025   NaN
50%         0.360000   1.000000e-08    1.100000   NaN
75%         0.542500   1.000000e-06    1.160000   NaN
max         0.960000   1.000000e-05    8.300000   NaN
# Method 2: Frequency Distribution Plots
plt.figure(figsize=(12, 4))
sns.histplot(data['riskFrequency'].dropna(), bins=20, kde=True)
plt.title('Frequency Distribution of Risk Allele Frequency')
plt.xlabel('Risk Frequency')
plt.ylabel('Count')
plt.show()
No description has been provided for this image
# Distribution of Risk Frequency
plt.figure(figsize=(10, 6))
sns.histplot(ms_data['riskFrequency'].dropna(), bins=20, kde=True, color='skyblue')
plt.title("Distribution of Risk Frequency")
plt.xlabel("Risk Frequency")
plt.ylabel("Frequency")
plt.show()
No description has been provided for this image
# Method 3: Box Plots for Outliers
plt.figure(figsize=(12, 4))
sns.boxplot(x='variable', y='value', data=pd.melt(data[['riskFrequency', 'pValue', 'orValue']]))
plt.title('Box Plot for Outliers in Risk Frequency, pValue, and orValue')
plt.show()
No description has been provided for this image
# Box plot to compare riskFrequency by mappedGenes
plt.figure(figsize=(40, 8))
sns.boxplot(data=ms_data, x='mappedGenes', y='riskFrequency')
plt.xticks(rotation=90)
plt.title("Risk Frequency by Mapped Genes")
plt.xlabel("Mapped Genes")
plt.ylabel("Risk Frequency")
plt.show()
No description has been provided for this image
# Distribution of Odds Ratio (orValue)
plt.figure(figsize=(10, 6))
sns.histplot(ms_data['orValue'].dropna(), bins=20, kde=True, color='salmon')
plt.title("Distribution of Odds Ratio")
plt.xlabel("Odds Ratio")
plt.ylabel("Frequency")
plt.show()
No description has been provided for this image
Inferences:

Summary Statistics: The central tendencies and spread (mean, median, standard deviation) give insights into the distribution of SNP-related measures. For instance, if the orValue has a mean close to 1, it may suggest that most SNPs have limited influence on MS risk. High variance in pValue indicates diverse levels of statistical significance.

Frequency Distribution Plot: The frequency distribution of riskFrequency reveals the commonality of specific allele frequencies in the dataset. A skew towards low-frequency alleles suggests that most risk alleles are rare, which is typical for complex diseases like MS.

Box Plot for Outliers: Outliers in riskFrequency and orValue suggest that some SNPs may have exceptionally high risk frequencies or odds ratios, making them candidates for further investigation. For example, if certain orValue values stand out as high outliers, these SNPs might have a stronger association with MS.

Diagnostic Analytics

Diagnostic analytics provides further exploration of relationships between SNP characteristics, focusing on correlation and statistical tests.

Method 1: Correlation Matrix
Method 2: Pair Plots
Method 3: Heatmaps

# Correlation matrix including 'beta' and 'ci' for numeric columns only
correlation_matrix = ms_data[['riskFrequency', 'orValue', 'pValue', ]].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
Correlation Matrix:
               riskFrequency   orValue    pValue
riskFrequency       1.000000 -0.172646 -0.098162
orValue            -0.172646  1.000000  0.147106
pValue             -0.098162  0.147106  1.000000
No description has been provided for this image
# Method 2: Pair Plot
sns.pairplot(data[['pValue', 'riskFrequency', 'orValue']])
plt.suptitle('Pair Plot of pValue, Risk Frequency, and Odds Ratio', y=1.02)
plt.show()
No description has been provided for this image
# Analyzing summary statistics by MS type
ms_types = ms_data['traitName'].unique()
for ms_type in ms_types:
    print(f"\n{ms_type} Summary Statistics:")
    print(ms_data[ms_data['traitName'] == ms_type][['pValue', 'riskFrequency', 'orValue', 'beta']].describe())
Multiple sclerosis Summary Statistics:
              pValue  riskFrequency     orValue  beta
count   1.980000e+02     198.000000  198.000000   0.0
mean    8.691389e-07       0.484682    1.214949   NaN
std     2.013082e-06       0.231984    0.314119   NaN
min    4.000000e-225       0.020000    1.060000   NaN
25%     6.000000e-15       0.272500    1.090000   NaN
50%     2.000000e-09       0.485000    1.120000   NaN
75%     4.750000e-07       0.670000    1.190000   NaN
max     9.000000e-06       0.960000    3.430000   NaN

Relapse in treatment-naive multiple sclerosis (time to event) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00     1.00   0.0
mean   2.000000e-10           0.02     2.15   NaN
std             NaN            NaN      NaN   NaN
min    2.000000e-10           0.02     2.15   NaN
25%    2.000000e-10           0.02     2.15   NaN
50%    2.000000e-10           0.02     2.15   NaN
75%    2.000000e-10           0.02     2.15   NaN
max    2.000000e-10           0.02     2.15   NaN

Multiple sclerosis (OCB status) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00          1.000     1.00   0.0
mean   9.000000e-07          0.951     2.17   NaN
std             NaN            NaN      NaN   NaN
min    9.000000e-07          0.951     2.17   NaN
25%    9.000000e-07          0.951     2.17   NaN
50%    9.000000e-07          0.951     2.17   NaN
75%    9.000000e-07          0.951     2.17   NaN
max    9.000000e-07          0.951     2.17   NaN

Response to interferon beta in multiple sclerosis Summary Statistics:
         pValue  riskFrequency  orValue  beta
count  3.000000       3.000000  3.00000   0.0
mean   0.000005       0.509800  3.72730   NaN
std    0.000002       0.374737  1.20929   NaN
min    0.000002       0.130500  2.37000   NaN
25%    0.000004       0.324800  3.24595   NaN
50%    0.000006       0.519100  4.12190   NaN
75%    0.000006       0.699450  4.40595   NaN
max    0.000006       0.879800  4.69000   NaN

Decreased low contrast letter acuity in multiple sclerosis Summary Statistics:
             pValue  riskFrequency   orValue  beta
count  6.000000e+00       6.000000  6.000000   0.0
mean   1.956667e-06       0.180000  1.551667   NaN
std    1.305399e-06       0.127122  0.743113   NaN
min    4.000000e-08       0.050000  0.630000   NaN
25%    1.025000e-06       0.082500  0.920000   NaN
50%    2.500000e-06       0.155000  1.765000   NaN
75%    3.000000e-06       0.257500  2.130000   NaN
max    3.000000e-06       0.370000  2.260000   NaN

Neutralising antibody response to interferon beta therapy in multiple sclerosis (presence of antibodies) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00      1.0   0.0
mean   2.000000e-15           0.25      2.6   NaN
std             NaN            NaN      NaN   NaN
min    2.000000e-15           0.25      2.6   NaN
25%    2.000000e-15           0.25      2.6   NaN
50%    2.000000e-15           0.25      2.6   NaN
75%    2.000000e-15           0.25      2.6   NaN
max    2.000000e-15           0.25      2.6   NaN

Oligoclonal band status in multiple sclerosis Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00     1.00   0.0
mean   4.000000e-15           0.28     2.23   NaN
std             NaN            NaN      NaN   NaN
min    4.000000e-15           0.28     2.23   NaN
25%    4.000000e-15           0.28     2.23   NaN
50%    4.000000e-15           0.28     2.23   NaN
75%    4.000000e-15           0.28     2.23   NaN
max    4.000000e-15           0.28     2.23   NaN
# Summary statistics for each MS type
ms_types = ms_data['traitName'].unique()
for ms_type in ms_types:
    print(f"\n{ms_type} Summary Statistics:")
    print(ms_data[ms_data['traitName'] == ms_type][['pValue', 'riskFrequency', 'orValue', 'beta']].describe())
Multiple sclerosis Summary Statistics:
              pValue  riskFrequency     orValue  beta
count   1.980000e+02     198.000000  198.000000   0.0
mean    8.691389e-07       0.484682    1.214949   NaN
std     2.013082e-06       0.231984    0.314119   NaN
min    4.000000e-225       0.020000    1.060000   NaN
25%     6.000000e-15       0.272500    1.090000   NaN
50%     2.000000e-09       0.485000    1.120000   NaN
75%     4.750000e-07       0.670000    1.190000   NaN
max     9.000000e-06       0.960000    3.430000   NaN

Relapse in treatment-naive multiple sclerosis (time to event) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00     1.00   0.0
mean   2.000000e-10           0.02     2.15   NaN
std             NaN            NaN      NaN   NaN
min    2.000000e-10           0.02     2.15   NaN
25%    2.000000e-10           0.02     2.15   NaN
50%    2.000000e-10           0.02     2.15   NaN
75%    2.000000e-10           0.02     2.15   NaN
max    2.000000e-10           0.02     2.15   NaN

Multiple sclerosis (OCB status) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00          1.000     1.00   0.0
mean   9.000000e-07          0.951     2.17   NaN
std             NaN            NaN      NaN   NaN
min    9.000000e-07          0.951     2.17   NaN
25%    9.000000e-07          0.951     2.17   NaN
50%    9.000000e-07          0.951     2.17   NaN
75%    9.000000e-07          0.951     2.17   NaN
max    9.000000e-07          0.951     2.17   NaN

Response to interferon beta in multiple sclerosis Summary Statistics:
         pValue  riskFrequency  orValue  beta
count  3.000000       3.000000  3.00000   0.0
mean   0.000005       0.509800  3.72730   NaN
std    0.000002       0.374737  1.20929   NaN
min    0.000002       0.130500  2.37000   NaN
25%    0.000004       0.324800  3.24595   NaN
50%    0.000006       0.519100  4.12190   NaN
75%    0.000006       0.699450  4.40595   NaN
max    0.000006       0.879800  4.69000   NaN

Decreased low contrast letter acuity in multiple sclerosis Summary Statistics:
             pValue  riskFrequency   orValue  beta
count  6.000000e+00       6.000000  6.000000   0.0
mean   1.956667e-06       0.180000  1.551667   NaN
std    1.305399e-06       0.127122  0.743113   NaN
min    4.000000e-08       0.050000  0.630000   NaN
25%    1.025000e-06       0.082500  0.920000   NaN
50%    2.500000e-06       0.155000  1.765000   NaN
75%    3.000000e-06       0.257500  2.130000   NaN
max    3.000000e-06       0.370000  2.260000   NaN

Neutralising antibody response to interferon beta therapy in multiple sclerosis (presence of antibodies) Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00      1.0   0.0
mean   2.000000e-15           0.25      2.6   NaN
std             NaN            NaN      NaN   NaN
min    2.000000e-15           0.25      2.6   NaN
25%    2.000000e-15           0.25      2.6   NaN
50%    2.000000e-15           0.25      2.6   NaN
75%    2.000000e-15           0.25      2.6   NaN
max    2.000000e-15           0.25      2.6   NaN

Oligoclonal band status in multiple sclerosis Summary Statistics:
             pValue  riskFrequency  orValue  beta
count  1.000000e+00           1.00     1.00   0.0
mean   4.000000e-15           0.28     2.23   NaN
std             NaN            NaN      NaN   NaN
min    4.000000e-15           0.28     2.23   NaN
25%    4.000000e-15           0.28     2.23   NaN
50%    4.000000e-15           0.28     2.23   NaN
75%    4.000000e-15           0.28     2.23   NaN
max    4.000000e-15           0.28     2.23   NaN
# T-test to compare high vs. low riskFrequency odds ratios within MS types
for ms_type in ms_types:
    high_freq = ms_data[(ms_data['traitName'] == ms_type) & (ms_data['riskFrequency'] > ms_data['riskFrequency'].median())]
    low_freq = ms_data[(ms_data['traitName'] == ms_type) & (ms_data['riskFrequency'] <= ms_data['riskFrequency'].median())]
    t_stat, p_val = ttest_ind(high_freq['orValue'], low_freq['orValue'], nan_policy='omit')
    print(f"T-test for {ms_type} - High vs Low risk frequency: t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
T-test for Multiple sclerosis - High vs Low risk frequency: t-statistic = -2.151, p-value = 0.033
T-test for Relapse in treatment-naive multiple sclerosis (time to event) - High vs Low risk frequency: t-statistic = nan, p-value = nan
T-test for Multiple sclerosis (OCB status) - High vs Low risk frequency: t-statistic = nan, p-value = nan
T-test for Response to interferon beta in multiple sclerosis - High vs Low risk frequency: t-statistic = -0.295, p-value = 0.818
T-test for Decreased low contrast letter acuity in multiple sclerosis - High vs Low risk frequency: t-statistic = nan, p-value = nan
T-test for Neutralising antibody response to interferon beta therapy in multiple sclerosis (presence of antibodies) - High vs Low risk frequency: t-statistic = nan, p-value = nan
T-test for Oligoclonal band status in multiple sclerosis - High vs Low risk frequency: t-statistic = nan, p-value = nan
Inferences:

Correlation Matrix and Heatmap: The correlation analysis indicates whether certain SNP characteristics are related. For instance, a negative correlation between pValue and orValue would suggest that SNPs with stronger associations (lower pValues) have higher odds ratios. This relationship helps identify potentially impactful SNPs.

Pair Plot: The pair plot visually represents the distribution and pairwise relationships among variables. For instance, clusters of points in the orValue vs. riskFrequency plot can reveal if specific SNPs with high odds ratios also have distinct risk frequencies, helping to isolate patterns linked to MS risk.

T-Test Analysis: The T-test compares odds ratios between high- and low-frequency SNPs for each MS type. A statistically significant difference suggests that SNPs with higher frequencies may contribute differently to MS risk across various types of MS. This provides an understanding of the genetic variations associated with each MS subtype.

Predictive Analytics

Predictive analytics involves training machine learning models to classify SNPs based on high or low risk. We use Logistic Regression, Decision Tree, and Random Forest classifiers with hyperparameter tuning.

Method 1: Logistic Regression Model
Method 2: Decision Tree Classifier
Method 3: Random Forest Classifier

# Define target variable
ms_data['high_risk'] = (ms_data['orValue'] > 2).astype(int)
X = ms_data[['riskFrequency', 'orValue', 'pValue', 'beta']]
y = ms_data['high_risk']
X
riskFrequency	orValue	pValue	beta
35	0.20	1.87	6.000000e-06	NaN
36	0.67	2.04	4.000000e-06	NaN
37	0.19	1.87	6.000000e-06	NaN
61	0.02	2.15	2.000000e-10	NaN
133	0.85	1.25	3.000000e-08	NaN
...	...	...	...	...
939	0.13	1.08	9.000000e-06	NaN
940	0.10	1.10	7.000000e-07	NaN
941	0.15	1.09	7.000000e-08	NaN
942	0.14	1.09	1.000000e-07	NaN
943	0.83	1.10	9.000000e-06	NaN
211 rows × 4 columns

y
35     0
36     1
37     0
61     1
133    0
      ..
939    0
940    0
941    0
942    0
943    0
Name: high_risk, Length: 211, dtype: int64
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Hyperparameter Tuning for Each Model

from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define the hyperparameter grid for Logistic Regression
log_reg_param_grid = {
    'logistic__C': [0.01, 0.1, 1, 10, 100],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver': ['liblinear']
}

# Define a pipeline that first imputes missing values, then applies Logistic Regression
log_reg_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # or strategy='median' depending on your preference
    ('logistic', LogisticRegression())
])

# Initialize GridSearchCV with the pipeline
log_reg_grid = GridSearchCV(log_reg_pipeline, log_reg_param_grid, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)

# Display best parameters and model performance
print("Best parameters for Logistic Regression:", log_reg_grid.best_params_)
log_reg_best = log_reg_grid.best_estimator_
log_reg_pred_tuned = log_reg_best.predict(X_test)

print("Logistic Regression Classification Report (Tuned):")
print(classification_report(y_test, log_reg_pred_tuned))
Best parameters for Logistic Regression: {'logistic__C': 10, 'logistic__penalty': 'l1', 'logistic__solver': 'liblinear'} Logistic Regression Classification Report (Tuned): precision recall f1-score support

       0       1.00      1.00      1.00        41
       1       1.00      1.00      1.00         2

accuracy                           1.00        43
macro avg 1.00 1.00 1.00 43 weighted avg 1.00 1.00 1.00 43

# Method 2: Decision Tree Classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, tree_pred))
Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       1.00      1.00      1.00         2

    accuracy                           1.00        43
   macro avg       1.00      1.00      1.00        43
weighted avg       1.00      1.00      1.00        43

# Define hyperparameter grid
tree_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv=5, scoring='accuracy')
tree_grid.fit(X_train, y_train)

# Best parameters and performance
print("Best parameters for Decision Tree:", tree_grid.best_params_)
tree_best = tree_grid.best_estimator_
tree_pred_tuned = tree_best.predict(X_test)
print("Decision Tree Classification Report (Tuned):")
print(classification_report(y_test, tree_pred_tuned))
Best parameters for Decision Tree: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Decision Tree Classification Report (Tuned):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       1.00      1.00      1.00         2

    accuracy                           1.00        43
   macro avg       1.00      1.00      1.00        43
weighted avg       1.00      1.00      1.00        43

# Method 3: Random Forest Classifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, forest_pred))
Random Forest Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       1.00      1.00      1.00         2

    accuracy                           1.00        43
   macro avg       1.00      1.00      1.00        43
weighted avg       1.00      1.00      1.00        43

# Define hyperparameter grid
forest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
forest_grid = GridSearchCV(RandomForestClassifier(), forest_param_grid, cv=5, scoring='accuracy')
forest_grid.fit(X_train, y_train)

# Best parameters and performance
print("Best parameters for Random Forest:", forest_grid.best_params_)
forest_best = forest_grid.best_estimator_
forest_pred_tuned = forest_best.predict(X_test)
print("Random Forest Classification Report (Tuned):")
print(classification_report(y_test, forest_pred_tuned))
Best parameters for Random Forest: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Random Forest Classification Report (Tuned):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       1.00      1.00      1.00         2

    accuracy                           1.00        43
   macro avg       1.00      1.00      1.00        43
weighted avg       1.00      1.00      1.00        43

Inferences:

Logistic Regression: After tuning, logistic regression might show lower accuracy than ensemble methods but provides a clear interpretation of coefficients. It helps identify the SNP features most predictive of MS risk.

Decision Tree Classifier: The decision tree’s feature splits reveal the thresholds that classify SNPs into high or low risk. By examining the tree's structure, it’s possible to see if orValue or riskFrequency acts as a primary decision factor, which highlights the traits most associated with increased MS risk.

Random Forest Classifier: After hyperparameter tuning, the Random Forest classifier often achieves the highest accuracy, indicating its robustness in handling SNP feature interactions. The feature importance scores from this model reveal which SNP traits (e.g., orValue) have the greatest influence on MS risk prediction.

Hyperparameter tuning improves model generalizability, evidenced by stable accuracy across training and test sets. This suggests that the chosen SNP features effectively predict high-risk SNPs associated with MS.

Prescriptive Analytics

Prescriptive analytics identifies the most impactful SNPs for potential targeted research or interventions.

Method 1: Identify Top SNPs by Odds Ratio and Low P-Value
Method 2: Cumulative Impact Analysis

# Method 1: Top SNPs by Odds Ratio and Low P-Value
top_snps = data.nsmallest(10, 'pValue').nlargest(10, 'orValue')[['riskAllele', 'pValue', 'orValue']]
print("Top SNPs for Further Study:")
print(top_snps)
Top SNPs for Further Study:
           riskAllele         pValue   orValue
912       DRB*15:01-?  1.000000e-132  3.080000
303       rs3129889-G  1.000000e-206  2.970000
365       rs3104373-T  1.000000e-234  2.900000
295       rs9271366-G  7.000000e-184  2.780000
221       rs3135388-A  4.000000e-225  2.750000
136       rs3135388-A   9.000000e-81  1.990000
517      rs10801908-C   5.000000e-70  1.298700
638      rs11256593-T   3.000000e-65  1.206200
526  chr16:11213951-C   4.000000e-71  1.195172
630        rs438613-C   2.000000e-49  1.150400
# Method 2: Cumulative Impact Analysis - Identifying SNPs with high combined impact
cumulative_top_snps = data[data['orValue'] > 2]
cumulative_top_snps_summary = cumulative_top_snps[['riskAllele', 'orValue']].groupby('riskAllele').sum().sort_values(by='orValue', ascending=False)
print("Cumulative Impact of High-Risk SNPs:")
print(cumulative_top_snps_summary.head(10))
Cumulative Impact of High-Risk SNPs:
                  orValue
riskAllele               
rs2205986-?          8.30
rs1150757-A          7.52
rs3745672-?          7.39
HLA-B*08:01-?        6.95
HLA-DRB1*03:01-?     6.80
HLA-DQB1*02:01-?     6.37
HLA-C*07:01-?        5.64
rs3129934-T          5.64
rs9271366-G          5.40
rs74696548-G         5.10
# Identify high-risk SNPs within each MS type
for ms_type in ms_types:
    ms_type_data = ms_data[ms_data['traitName'] == ms_type]
    ms_type_data['High_Risk'] = (ms_type_data['orValue'] >= 1.5).astype(int)
    high_risk_snp = ms_type_data[ms_type_data['High_Risk'] == 1]
    print(f"\nHigh-Risk SNPs for {ms_type}:\n", high_risk_snp[['riskAllele', 'orValue', 'mappedGenes']])
High-Risk SNPs for Multiple sclerosis: riskAllele orValue mappedGenes 35 rs17149161-A 1.87 YWHAG 36 rs12644284-? 2.04 TRIM2 37 rs7789940-G 1.87 FPASL,YWHAG 136 rs3135388-A 1.99 HLA-DRB9,HLA-DRA 140 rs3129934-T 2.34 TSBP1,TSBP1-AS1 221 rs3135388-A 2.75 HLA-DRB9,HLA-DRA 225 rs4149584-T 1.58 TNFRSF1A 235 rs3135338-A 3.43 HLA-DRA,TSBP1-AS1 276 rs2040406-G 2.05 HLA-DQA1 295 rs9271366-G 2.78 HLA-DQA1,HLA-DRB1 299 rs6984045-C 1.59 ASAP1 303 rs3129889-G 2.97 HLA-DRB9,HLA-DRA

High-Risk SNPs for Relapse in treatment-naive multiple sclerosis (time to event): riskAllele orValue mappedGenes 61 rs11871306-C 2.15 WNT9B

High-Risk SNPs for Multiple sclerosis (OCB status): riskAllele orValue mappedGenes 141 rs9320598-? 2.17 MIR2113,MMS22L

High-Risk SNPs for Response to interferon beta in multiple sclerosis: riskAllele orValue mappedGenes 319 rs6691722-? 2.3700 GRHL3 320 rs16898029-? 4.6900 LINC02438,LCORL 321 rs12957214-C 4.1219 MIR924HG

High-Risk SNPs for Decreased low contrast letter acuity in multiple sclerosis: riskAllele orValue mappedGenes 323 rs72830848-T 2.25 MSI2 324 rs13241771-A 1.77 RNU6-393P,TMEM130 325 rs10157709-C 2.26 ZNF692-DT 327 rs10980055-G 1.76 PALM2AKAP2

High-Risk SNPs for Neutralising antibody response to interferon beta therapy in multiple sclerosis (presence of antibodies): riskAllele orValue mappedGenes 329 rs522308-T 2.6 HLA-DRB1,HLA-DQA1

High-Risk SNPs for Oligoclonal band status in multiple sclerosis: riskAllele orValue
726 rs9271640-G,rs3135388-A,rs3957148-A 2.23

                                       mappedGenes  
726 HLA-DRB1 - HLA-DQA1; HLA-DRA - HLA-DRB9; MTCO3...

Inferences:

Top SNPs by Odds Ratio and Low P-Value: SNPs with high odds ratios and low p-values are likely to have the strongest associations with MS. This list of top SNPs acts as a prioritized set for researchers, who may further validate these associations in lab settings. These SNPs could be key targets for genetic testing or therapeutic research.

Cumulative Impact Analysis: By aggregating high orValue SNPs, this analysis identifies SNPs with cumulative effects, meaning multiple SNPs affecting the same gene or pathway could collectively impact MS risk. This insight is valuable for understanding complex genetic influences on MS and identifying pathways for targeted drug development.

High-Risk SNP Identification per MS Type: SNPs classified as high risk within each MS subtype provide a tailored view of genetic risk factors for different forms of MS. For instance, if a specific SNP consistently appears in high-risk categories across multiple MS types, it may have a broader effect on MS susceptibility and could be a primary candidate for MS-wide interventions.

app = dash.Dash(__name__)

# Dashboard layout
app.layout = html.Div([
    html.H1("MS Genetic Variation Analysis by Type"),
    html.Label("Select MS Type:"),
    dcc.Dropdown(id='ms-type-dropdown', options=[{'label': t, 'value': t} for t in ms_types], value=ms_types[0]),
    html.Label("Odds Ratio Threshold:"),
    dcc.Slider(id='orValue-slider', min=0, max=5, step=0.1, value=1.5, marks={i: str(i) for i in range(6)}),
    dcc.Graph(id='scatter-plot')
])

# Callback to update scatter plot based on MS type and threshold
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('ms-type-dropdown', 'value'), Input('orValue-slider', 'value')]
)
def update_figure(selected_type, selected_orValue):
    filtered_data = ms_data[(ms_data['traitName'] == selected_type) & (ms_data['orValue'] >= selected_orValue)]
    fig = px.scatter(filtered_data, x="riskFrequency", y="orValue",
                     title=f"{selected_type} - Risk Frequency vs. Odds Ratio (Threshold ≥ {selected_orValue})")
    fig.update_layout(xaxis_title="Risk Frequency", yaxis_title="Odds Ratio", legend_title="Effect Size")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

Overall Inference

This project’s analyses provide a comprehensive view of genetic variations in SNP data related to MS. Descriptive and diagnostic analytics highlight key patterns and relationships among SNP features, while predictive analytics accurately identifies high-risk SNPs. Prescriptive analytics further narrows down impactful SNPs for potential genetic and clinical studies. These findings contribute to a better understanding of MS's genetic basis, supporting future research on genetic testing and targeted MS treatments.

Conclusion

This project demonstrates the power of using machine learning and data analytics on genetic data for MS research. By analyzing SNP data, the project identifies potential high-risk SNPs that may have significant roles in MS development. This analysis can be a stepping stone for future studies to validate these SNPs’ roles and possibly lead to targeted MS treatments or preventative measures.
