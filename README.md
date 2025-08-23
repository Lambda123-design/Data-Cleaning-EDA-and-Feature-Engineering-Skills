# Data-Cleaning-EDA-and-Feature-Engineering-Skills

##### X_train=scaler.fit_transform(X_train); Fit will find mean and Std Dev; Transform will go and do Standardization (Z-Score); X_test=scaler.transform(X_test) For X_Test we don't need to use it because it will go and use the same mean and std dev found in X_Train; It also avoids Data Leakage

##### It shldn't know anything about test data; It shld only use Train Data mean and std dev

######### Very very very Important: In Used Car Price Prediction, "Car Model" has around 120 unique features and we did **Label Encoding** for it. Reason is **If we assign a Label, it will try to find out the relationship between Label and the target column Price. Say, If Label is very high, Price may also be high That kind of relationship it will find out**(In Krish Interview he has asked the question based on this - If Pincode has around 30 categories, how do you encode it, as it might be very important in House Price Prediction Project)

#### Or we can also assign labels to top 15, and assign "Others" label to rest; Out of those 120

#### In Column Transformer - remainder='passthrough' - Only categorical features will change; Don't change others

#### In One-Hot Encoding, if n features we only need n-1 columns; So use drop_first="True"

#### Important: Learn about Column Transformer below in Holiday Project (3rd Project) 

#### Always check for y.value_counts for Imbalanced Datasets (3968, 920), Krish said it still has good number of both categories; Ensemble Models such as Random Forest, XGBoost,etc.. perform well in Imbalanced Datasets

##### Always check for "df['Gender'].value_counts()" value_counts because if there is any difference in values. In Krish Project, there was "Female" and "Fe Male". We have to fix all those in Feature Engineering

#### Creating a New Feature to make more sense; and also remove one feature from the dataset for the mode

##### Getting Discrete Features: **Discrete Feature can have around 10**; Example Pincode; Whereas categorical will have 2 or 3 

##### Same thing we have used in Used Car Price Prediction Project too (4th Project); Check that code too

df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
   
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

###### Imp: For Standardization with Height and weight Krish got an error; Expecting 2D for X; So he did "X=df[['Weight']]"

##### X_train=scaler.fit_transform(X_train); Fit will find mean and Std Dev; Transform will go and do Standardization (Z-Score); X_test=scaler.transform(X_test) For X_Test we don't need to use it because it will go and use the same mean and std dev found in X_Train; It also avoids Data Leakage

##### It shldn't know anything about test data; It shld only use Train Data mean and std dev
 
**I) Algerian Forest Fires Project**

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


****II) Student Performance Predictor (End-to-End-ML_Project)****

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

**III) Holiday Package Prediction Project**

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

**IV) Used Car Price Prediction:**

1. Car model will be most important, so we are removing other two

df.drop('car_name', axis=1, inplace=True)

df.drop('brand', axis=1, inplace=True)

**2. Seeing the Feature Types**

num_features = [feature for feature in df.columns if df[feature].dtype != 'O']

cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']

discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]

continuous_features=[feature for feature in num_features if feature not in discrete_features]

**3. Label Encoding for "Model" - Read the top first point; Very Important**

le=LabelEncoder()

X['model']=le.fit_transform(X['model'])

**4. Create Column Transformer with 3 types of transformers:**

num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type','fuel_type','transmission_type']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features)
    ],remainder='passthrough'

)

#### remainder='passthrough' - Only categorical features will change; Don't change others

**V) Height Weight Prediction - Simple Linear Regression Project**

1. print("The slope or coefficient of weight is ",regressor.coef_)
print("Intercept:",regressor.intercept_)

2. Plotting the best fit line -plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),'r')

3. ## new data point weight is 80


### Krish got lot of errors here; Please check to video to learn
scaled_weight=scaler.transform([[80]])
scaled_weight

scaled_weight[0]

scaled_weight[0]

print("The height prediction for weight 80 kg is :",regressor.predict([scaled_weight[0]]))

#### 4. Assumptions to say that it is a good model:

#### (i) Assumption (i)

## plot a scatter plot for the prediction
plt.scatter(y_test,y_pred_test)

### If linearly distributed, then it is good model

**#### (ii) Assumption (ii)**

## Residuals
residuals=y_test-y_pred_test
residuals

## plot this residuals
import seaborn as sns
sns.distplot(residuals,kde=True)

#### If Residuals are normally distributed, then it is a Good Model

#### (iii) ## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(y_pred_test,residuals)

#### Residuals should be uniformly distributed; That is, it should be distributed here and there in the graph

###### Please refer to the Notebook for Evaluation and other visualization codes

**VI) California Housing Prediction:**

1. Loading the Dataset:

california=fetch_california_housing(); california.keys(); print(california.DESCR); california.target_names; california.feature_names

## Lets prepare the dataframe 
dataset=pd.DataFrame(california.data,columns=california.feature_names)
dataset.head()

dataset['Price']=california.target

dataset.head()

## Independent and Dependent features
X=dataset.iloc[:,:-1] #independent features
y=dataset.iloc[:,-1] #dependent features

X_test=scaler.transform(X_test)

#### Model should not have any idea about Test dataset; Mean and Std Dev will be taken from Train Dataset; It will avoid Data Leakage

2. ## slope or coefficients
regression.coef_

Co-efficients will be 8 (No. of features); But Intercept will be only one

## intercepts
regression.intercept_

3. ## Performance Metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

#### Score might be less because it might not have a linear property; It might be scattered here and there

#### 4. Assumptions:

1. Linear Property (In our case, it was having somewhat linear properties)

plt.scatter(y_test,y_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")

2. **Residual Check: If it is normal - then it is good; or not**

residuals=y_test-y_pred

sns.displot(residuals,kind="kde")

3. ## SCatter plot with predictions and residual
##uniform distribution
plt.scatter(y_pred,residuals)

**Here in our case, there was a Pattern and thus it might not be a good model; We we can try with other models as well**

**We will learn about Ridge, Lasso, Decision Tree, Random Forest. With those performance will get increased**

## Pickling - Python pickle module is used for serialising and de-serialising a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serialises” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character 
stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

import pickle; pickle.dump(regression,open('regressor.pkl','wb'))

model=pickle.load(open('regressor.pkl','rb'))

model.predict(X_test)
