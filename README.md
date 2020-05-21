# Project-Customer-Churn-Python

#Data cleaning 

1.Checked missing/nulls

2.Found correlations of variables & importance of variables

3.Deal with outliers

4. Handle skewed numerical features with log +1 transformation 

#EDA

1.	Hist for numeric variables

2.	Boxplot for categorical variables

#Data Transformation (for model processing )

1) Encoded response variable (RENEW) into factor 1 and factor 0

2) Reduced the number of categorical predictors levels (4)

3) Created dummy variables to represent binary factors for categorical predictors

(replace: M2EXCFLG; HOMEFCTYCHANGE; RECENTMOVING );;

(one hot encoding:  F2HOMRGN ) ;;each categorical feature with n categories is transformed into n binary features

4) Data Normalization

#Feature selection: for numerical predictors vs categorical response (RENEW)

1. Univariate feature selection works by selecting the best features based on univariate statistical tests 

F_classif: ANOVA F-value between label/feature for classification tasks

2. Wrapper-based: consider the selection of a set of features as a search problem

Recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features

3. Embedded: use algorithms that have built-in feature selection methods

Tree-based feature selection (RF)

4. Principal Components Analysis (PCA)

#Build Machine Learning Models

Evaluate the model using various metrics (including precision and recall):

1.Logistic Regression: predictors should be independent; numerical variables only

After f-test feature selection +LogisticRegressionCV: accuracy=0.7;recall=0.7

After RFE feature selection +LogisticRegressionCV: accuracy=0.7;recall=0.7
      
Drawback: Logistic Regression learns a linear decision surface that separates your classes. It could be possible that our 2 classes may not be linearly separable. In such a case, I need to look at other classifiers such Support Vector Machines which are able to learn more complex decision boundaries. I also started looking at Tree-Based classifiers such as Decision Trees which can learn rules from the dataset

2. Random Forest: works for the mixed dataset(Numerical features+Categorical features); can deal with the missing values and outliers

After f-test feature selection: accuracy=0.75;recall=0.75

After tree-based feature selection+ Grid search: accuracy = 0.74;recall=0.75

3.KNN: numeric features only; require features to be scaled; sensitive to the feature selection

Larger K value leads to smoother decision boundary (less complex model). Smaller K leads to more complex model (may lead to overfitting)

After f-test feature selection: k=61, accuracy=0.70;recall=0.70

After RFE feature selection: k=62, accuracy=0.70;recall=0.69

After tree-based feature selection: k=70, accuracy=0.70; recall=0.70

4. SVM: require features to be scaled;  choose proper kernels functions; numerical variables only; It is not suitable for large dataset( longer training time); sensitive to kernels functions 

C (penalty number) : represents the misclassification and error. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane

Gamma: A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting

After f-test feature selection: accuracy=0.70;precision=0.70;recall=0.70

After RFE feature selection: accuracy=0.69;precision=0.68;recall=0.69

After tree-based feature selection: accuracy=0.69;precision=0.69;recall=0.69

Grid search: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}; accuracy=0.71; precision=0.70;recall=0.71

PCA (n=4) + SVM: accuracy=0.70;precision=0.70;recall=0.70

#Design and Build Neural Network using TensorFlow (Sequential Model)

1. Use the subset from f-test feature selection

2. Define and train a model using Keras

3. Improve performance with algorithm tuning (72% accuracy so far)


