# Data-Cleaning-EDA-and-Feature-Engineering-Skills

#### In One-Hot Encoding, if n features we only need n-1 columns; So use drop_first="True"

#### Important: Learn about Column Transformer below in Holiday Project (2nd Project) 

#### Always check for y.value_counts for Imbalanced Datasets (3968, 920), Krish said it still has good number of both categories; Ensemble Models such as Random Forest, XGBoost,etc.. perform well in Imbalanced Datasets

##### Always check for "df['Gender'].value_counts()" value_counts because if there is any difference in values. In Krish Project, there was "Female" and "Fe Male". We have to fix all those in Feature Engineering

#### Creating a New Feature to make more sense; and also remove one feature from the dataset for the mode

##### Getting Discrete Features: **Discrete Feature can have around 10**; Example Pincode; Whereas categorical will have 2 or 3

df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
   
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

**Algerian Forest Fires Project**

**Data Cleaning Learnings**
1. dataset.loc[:122,"Region"]=0
   dataset.loc[122:,"Region"]=1

2. df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)

**EDA Learnings**
3. plt.style.use('seaborn')
df_copy.hist(bins=50,figsize=(20,15))
plt.show()

4. percentage=df_copy['Classes'].value_counts(normalize=True)*100

5. Plotting piechart
classlabels=["Fire","Not Fire"]
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
plt.title("Pie Chart of Classes")
plt.show()

6. Monthly Fire Analysis
## Monthly Fire Analysis
dftemp=df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel('Number of Fires',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title("Fire Analysis of Sidi- Bel Regions",weight='bold')

7. Correlation for Featuring Selection:
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

## threshold--Domain expertise
corr_features=correlation(X_train,0.85)

## drop features when correlation is more than 0.85 
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)
X_train.shape,X_test.shape

8. Feature Scaling:
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


**Hyper Parameter Tuning:**
model=LogisticRegression()
penalty=['l1', 'l2', 'elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

params=dict(penalty=penalty,C=c_values,solver=solver)

from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold()

## GridSearchCV
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(estimator=model,param_grid=params,scoring='accuracy',cv=cv,n_jobs=-1)

grid.fit(X_train,y_train)

## RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV

model=LogisticRegression()
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,scoring='accuracy')

randomcv.fit(X_train,y_train)


****Student Performance Predictor (End-to-End-ML_Project)****

# Create Column Transformer with 3 types of transformers

num_features = X.select_dtypes(exclude="object").columns

cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()

oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
         ("StandardScaler", numeric_transformer, num_features),        
    ]
)

# Creating an Evaluate Function to give all metrics after model training

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

**Holiday Package Prediction Project**

1. Fixing "Female" and "Fe Male":

df['Gender'].value_counts()

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

2. Check for Missing Values:

##these are the features with nan value
features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
for feature in features_with_na:
    print(feature,np.round(df[feature].isnull().mean()*100,5), '% missing values')

3. statistics on numerical columns (Null cols)
   
df[features_with_na].select_dtypes(exclude='object').describe()

#### 4. Creating a New Feature to make more sense; and also remove one feature from the dataset for the mode

df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
   
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

5. get all the numeric features
num_features = [feature for feature in df.columns if df[feature].dtype != 'O'];
print('Num of Numerical Features :', len(num_features))

6. Getting Categorical Features

cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']; 
print('Num of Categorical Features :', len(cat_features))

7. Getting Discrete Features: **Discrete Feature can have around 10**

discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]; print('Num of Discrete Features :',len(discrete_features))

8. Getting Continuous Features:

continuous_features=[feature for feature in num_features if feature not in discrete_features]; print('Num of Continuous Features :',len(continuous_features))

9. Column Transformer

# Create Column Transformer with 3 types of transformers
cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
         ("OneHotEncoder", oh_transformer, cat_features),
          ("StandardScaler", numeric_transformer, num_features)
    ]
)
