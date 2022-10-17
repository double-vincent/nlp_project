import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

def spotlight_stats(df, feature, title, phase=None):
    ''' 
    Purpose:
        To create visuals and print statistics for the feature of the data set
    ---
    Parameters:
        df: dataframe containing features
        feature: the feature (column) to be used for testing and visualization
        phase: the phase of the pipeline for which the output is needed
    ---
    Output:
        prop_df: dataframe that contains population proportions and unemployment rate
    ---
    '''

    multi_col = pd.MultiIndex.from_tuples([('population_proportions', 'language'), 
                                    ('population_proportions', 'language'),
                                    ('population_proportions', 'change')])
    
    # dataframe, 3 columns, 
    prop_df = pd.DataFrame(columns=multi_col)
    prop_df['unemployment_rate'] = round(1 - df.groupby(by=feature).language.mean().sort_values(ascending=True), 2)

    # show the proportion of the population that each industry is
    language_pop_proportion = df[df.language == 1][feature].value_counts(normalize=True) 

    # show the proportion of the population that each industry is
    language_pop_proportion = df[df.language == 0][feature].value_counts(normalize=True) 
    
    #assign proper values to dframe
    prop_df[('population_proportions', 'language')] = language_pop_proportion
    prop_df[('population_proportions', 'language')] = language_pop_proportion
    prop_df[('population_proportions', 'change')] = language_pop_proportion - language_pop_proportion

    #chi2 testing and outcome printing
    alpha = .05
    crosstab = pd.crosstab(df[feature], df["language"])

    chi2, p, dof, expected = chi2_contingency(crosstab)

    if phase == 'explore':
        print('Crosstab\n')
        print(crosstab.values)
        print('---\nExpected\n')
        print(f'{expected.astype(int)}')
        print('---\n')

    print(f'chi^2: {chi2:.4f}')
    print(f'p: {p:.4f}')
    print(f'degrees of freedom: {dof}')

    if p < alpha :
        print('Reject null hypothesis')
    else: 
        print('Fail to reject null hypothesis')

    #plots the distributions of the feature in separate columns for language vs language
    plt.figure(figsize=(20,6))
    sns.catplot(data=df, x=feature, col='language', kind='count', sharey=False)
    plt.suptitle(title, y=1.02)
    plt.show()

    return round(prop_df, 3)

def split_scale(df):   
    ''' 
    Purpose:
        To split and scale the input dataframe
    ---
    Parameters:
        df: a tidy dataframe
    ---
    Output:
        train: unscaled subset of dataframe for exploration and model training
        validate: unscaled and unseen data for model testing
        test: unscaled and unseen data for final model test
        train_scaled: scaled subset of dataframe for exploration and model training
        validate_scaled: scaled and unseen data for model testing
        test_scaled: scaled and unseen data for model testing
    ---
    '''
    #train_test_split
    train_validate, test = train_test_split(df, test_size=.2, random_state=514, stratify=df['language'])
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514, stratify=train_validate['language'])
    
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled =  scaler.transform(test[num_cols.columns])

    return train, validate, test, train_scaled, validate_scaled, test_scaled

def split_X_y(train, validate, test):
    ''' 
    Purpose:
    ---
        To split the subsets further for modeling and testing
    ---
    Parameters:
        train: unscaled subset of dataframe for exploration and model training
        validate: unscaled and unseen data for model testing
        test: unscaled and unseen data for final model test
    ---
    Output:
        X_train: features to fit model and make predictions
        y_train: target variable outcome for model evaluation
        X_validate: features to make predictions
        y_validate: target variable outcome for model evaluation
        X_test: features to make predictions
        y_test: target variable outcome for model evaluation
    ---
    '''
    # split data into Big X, small y sets 
    X_train = train.drop(columns=['language'])
    y_train = train.language

    X_validate = validate.drop(columns=['language'])
    y_validate = validate.language

    X_test = test.drop(columns=['language'])
    y_test = test.language

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def create_comp_chart():
    """
    purpose: to create a dataframe with an index reflecting compuation metrics for future models

    returns: a pandas dataframe with appropriately set index
    """
    statistics = ['Accuracy/Score',
    'True Positives' , 'False Positives', 'True Negatives', 'False Negatives', \
    'TPR/Recall', 'False Positive Rate', 'True Negative Rate', 'False Negative Rate', \
    'Precision', 'F1-Score', 'Support Positive', 'Support Negative']


    return pd.DataFrame({}, index=statistics)

def create_description_chart(y_train):
    ''' 
    Purpose:
        To create a chart that will hold model descriptions
    ---
    Parameters:
        y_train: subset of data that contains target variable outcomes
    ---
    Output:
        descriptions: dataframe that holds initial baseline predictions for given subset
    ---
    '''
    # formulate baseline accuracy
    baseline_accuracy = (y_train == 1).mean()

    descriptions = pd.DataFrame({'Model': 'Baseline', \
                                'Accuracy(Score)': baseline_accuracy,
                                'Type': 'Basic Baseline',
                                'Features Used': 'Baseline Prediction',
                                'Parameters': 'n/a'
                                }, index=[0])
    
    return descriptions

def get_selectors(parameters):
    ''' 
    Purpose:
        To read information and return list of parameters for use in modeling
    ---
    Parameters:
        parameters: a column in a data that contains parameters for modeling
    ---
    Output:
        parameters: useable list of parameters for modeling
    ---
    '''
    removal_list = ['Depth: ','K-Neighbors: ','Leaves: ','C: ',' Solver: ', ' Class Weight: ']

    for word in removal_list:
        parameters = parameters.replace(word, "")

    return parameters.split(',')

def get_features():
    ''' 
    Purpose:
    ---
        To create list of lists of features for modeling
    Parameters:
        None
    ---
    Output:
        feature_sets: group of features for use in modeling
    ---
    '''
    feat_set1 = ['industry', 'occupation', 'country_region', 'metro_area_size' , 'professional_certification', 'own_bus_or_farm','education']
    feat_set2 = ['household_num', 'children_in_household', 'education', 'enrolled_in_school', 'family_income', 'marital_status']
    feat_set3 = ['age', 'is_male', 'veteran', 'hispanic_non', 'race', 'birth_country', 'mother_birth_country', 'father_birth_country', 'citizenship', 'education']
    feat_set4 = ['age', 'industry', 'occupation','professional_certification','education','marital_status','is_male','citizenship']

    feature_sets = [feat_set1, feat_set2, feat_set3, feat_set4]

    return feature_sets

def compute_metrics(model, X_df, y_df):
    """
    purpose: function executes performs computations to produce evaulation metrics for a given model

    inputs: 
        model: a model that has been previous fit to spec
        X_df: a dataframe featuring the X subset of data for evaluation
        y_df: a dataframe featuring the model target variable

    Returns: a rounded pandas Series that can be adding to an evaulation metric comparison chart
    """
    # Make Predictions
    y_pred = model.predict(X_df)

    # Estimate Probability 
    y_pred_proba = model.predict_proba(X_df)

    #create confusion matrix
    confusion = confusion_matrix(y_df, y_pred)

    #assign results of confusion matrix to variables
    true_negative = confusion[0,0]
    false_positive = confusion[0,1]
    false_negative = confusion[1,0]
    true_positive = confusion[1,1]

    #accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    #true positive rate / recall
    recall = true_positive / (true_positive +false_negative)

    #false positive rate
    false_positive_rate = false_positive / (true_negative + false_positive)

    #true negative rate
    true_negative_rate = true_negative / (true_negative + false_positive)

    #false negative rate
    false_negative_rate = false_negative / (false_negative + true_positive)

    #precision
    precision = true_positive / (true_positive + false_positive)

    #f1-score
    f1_score = 2 * (precision * recall) / (precision + recall)

    #support
    support_positive = true_positive + false_negative
    support_negative = false_positive + true_negative

    metrics = pd.Series([accuracy, true_positive, false_positive, true_negative, false_negative,\
                        recall, false_positive_rate, true_negative_rate, false_negative_rate, \
                        precision, f1_score, support_positive, support_negative])
                        
    return metrics.round(4)

def model_dtc(feat_set,\
        model_descriptions,
        comparison_chart,
        subsets):
    ''' 
    Purpose:
        To fit and score models using a Decision Tree model
    ---
    Parameters:
        feat_set: set of features to be used for model fitting and scoring
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
    ---
    '''      
    train=subsets[0]
    X_train=subsets[1]
    y_train=subsets[2]

    features = []
    for feature in feat_set:
        features += [col for col in train.columns if feature in col]

    selectors = list(product(np.arange(20,25,2)))

    for idx, item in enumerate(selectors):
        model_id = 'DTC_'+f'{idx}{len(feat_set)}'
        dtc = DecisionTreeClassifier(max_depth=item[0],\
                                            random_state=514)
        
        dtc.fit(X_train[features], y_train)

        comparison_chart[model_id] = compute_metrics(dtc, X_train[features], y_train).values

        score = dtc.score(X_train[features], y_train).round(4)

        description = pd.DataFrame({'Model': model_id,
                                    'Accuracy(Score)': score,
                                    'Type': 'Decision Tree Classifier',
                                    'Features Used': f'{feat_set}',
                                    'Parameters': f'Depth: {item[0]}'},
                                    index=[0])

        model_descriptions = pd.concat([model_descriptions, description], ignore_index=True)

    return model_descriptions, comparison_chart

def model_rf(feat_set,\
        model_descriptions,
        comparison_chart,
        subsets, ):
    ''' 
    Purpose:
        To fit and score models using a Random Forest model
    ---
    Parameters:
        feat_set: set of features to be used for model fitting and scoring
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
    ---
    '''   
    train=subsets[0]
    X_train=subsets[1]
    y_train=subsets[2]

    features = []
    for feature in feat_set:
        features += [col for col in train.columns if feature in col]

    selectors = list(product([20,25], [3,2,1]))

    for idx, item in enumerate(selectors):
        model_id = 'RF_'+f'{idx}{len(feat_set)}'
        rf = RandomForestClassifier(max_depth=item[0],\
                                            min_samples_leaf=item[1],
                                            random_state=514)
        
        rf.fit(X_train[features], y_train)

        comparison_chart[model_id] = compute_metrics(rf, X_train[features], y_train).values

        score = rf.score(X_train[features], y_train).round(4)

        description = pd.DataFrame({'Model': model_id,
                                    'Accuracy(Score)': score,
                                    'Type': 'Random Forest',
                                    'Features Used': f'{feat_set}',
                                    'Parameters': f'Depth: {item[0]}, Leaves: {item[1]}'},
                                    index=[0])
       
    model_descriptions = pd.concat([model_descriptions, description], ignore_index=True)

    return model_descriptions, comparison_chart

def model_knn(feat_set,\
        model_descriptions,
        comparison_chart,
        subsets, ):
    ''' 
    Purpose:
        To fit and score models using a K-Nearest Neighbor model
    ---
    Parameters:
        feat_set: set of features to be used for model fitting and scoring
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
    ---
    '''   
    train=subsets[0]
    X_train=subsets[1]
    y_train=subsets[2]

    features = []
    for feature in feat_set:
        features += [col for col in train.columns if feature in col]

    k_range = range(4,5)
    scores = []
    
    for k in k_range:
        
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train[features], y_train)
        scores.append(knn.score(X_train[features], y_train))

        model_id = 'Knn_'+f'{k,len(feat_set)}'

        comparison_chart[model_id] = compute_metrics(knn, X_train[features], y_train).values

        score = knn.score(X_train[features], y_train).round(5)

        description = pd.DataFrame({'Model': model_id,
            'Accuracy(Score)': score,
            'Type': 'Knn',
            'Features Used': f'{feat_set}',
            'Parameters': f'K-Neighbors: {k}'},
            index=[0])

        model_descriptions = pd.concat([model_descriptions, description], ignore_index=True)
   
    # plt.figure()
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    # plt.scatter(k_range, scores)
    # plt.xticks([0,5,10,15,20])
    # plt.show()
    # np.mean(scores)

    
    return model_descriptions, comparison_chart

def model_lr(feat_set,\
        model_descriptions,
        comparison_chart,
        subsets, ):

    ''' 
    Purpose:
        To fit and score models using a Logistic Regression model
    ---
    Parameters:
        feat_set: set of features to be used for model fitting and scoring
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        model_descriptions: model descriptions for models used
        comparison: evaluation metrics for the models
    ---
    '''    
    train=subsets[0]
    X_train=subsets[1]
    y_train=subsets[2]

    features = []
    for feature in feat_set:
        features += [col for col in train.columns if feature in col]

    cees = [.1,.5,1]
    solver = ['newton-cg']
    weights = [None, 'balanced']

    selectors = list(product(cees, solver, weights))

    for idx, item in enumerate(selectors):
        model_id = 'LR_'+f'{idx}{len(feat_set)}'
        lr = LogisticRegression(C=item[0],\
                                solver=item[1],
                                class_weight=item[2],
                                max_iter=1000,
                                random_state=514)
        
        lr.fit(X_train[features], y_train)

        comparison_chart[model_id] = compute_metrics(lr, X_train[features], y_train).values

        score = lr.score(X_train[features], y_train).round(4)

        description = pd.DataFrame({'Model': model_id,
            'Accuracy(Score)': score,
            'Type': 'Logistic Regression',
            'Features Used': f'{feat_set}',
            'Parameters': f'C: {item[0]}, Solver: {item[1]}, Class Weight: {item[2]}'}, index=[0])

        model_descriptions = pd.concat([model_descriptions, description], ignore_index=True)

    return model_descriptions, comparison_chart

def train_models(y, feature_groups, subsets):
    ''' 
    Purpose:
        To fit and evaluate various models on the traning data subset
    ---
    Parameters:
        y: the y_train data subset
        feature_groups: groups of features to be used for modeling
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        train_descriptions: model descriptions for models used on the validation subset
        train_metrics: evaluation metrics for validation predictions
    ---
    '''    
    #take in features sets and run them through each of the different types
    #of models and their variations
    train_descriptions = create_description_chart(y)
    train_metrics = create_comp_chart()

    for features in feature_groups:
        
        train_descriptions, train_metrics = model_dtc(features, train_descriptions, train_metrics, subsets)
        train_descriptions, train_metrics = model_rf(features, train_descriptions, train_metrics, subsets)
        #train_descriptions, train_metrics = model_knn(features, train_descriptions, train_metrics, subsets)
        train_descriptions, train_metrics = model_lr(features, train_descriptions, train_metrics, subsets)

    train_descriptions.insert(loc=2, column='Sensitivity', value=0)

    for idx in train_descriptions.index:
        model_id = train_descriptions.iloc[idx]['Model']
        if model_id != 'Baseline':
            train_descriptions.loc[idx, 'Sensitivity'] = train_metrics.T.loc[model_id]['True Negative Rate']
            
    return train_descriptions, train_metrics


def score_on_validation(descriptions, subsets):
    ''' 
    Purpose:
        To score models on the validation subset
    ---
    Parameters:
        descriptions: descriptions of the best performing model from training phase
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        val_descriptions: model descriptions for models used on the validation subset
        validate_metrics: evaluation metrics for validation predictions
    ---
    '''    
    X_train=subsets[1]
    y_train=subsets[2]
    X_validate= subsets[3]
    y_validate = subsets[4]
    
    validate_metrics = create_comp_chart()
    val_descriptions = create_description_chart(y_validate)
    
    #feat_set = descriptions.iloc[1]['Features Used'].strip('\[]\'').split('\', \'')
    for idx in descriptions.index:

        model_id = descriptions.loc[idx]['Model']
        feat_set = descriptions.loc[idx]['Features Used'].strip('\[]\'').split('\', \'')
        selectors = get_selectors(descriptions.loc[idx]['Parameters'])

        features = []

        for feature in feat_set:
            features += [col for col in X_validate.columns if feature in col]

        if model_id.startswith('DTC'):
            val_model = DecisionTreeClassifier(max_depth=int(selectors[0]),\
                                                random_state=514)
        elif model_id.startswith('RF'):
            val_model = RandomForestClassifier(max_depth=int(selectors[0]),\
                                            min_samples_leaf=int(selectors[1]),
                                            random_state=514)
        elif model_id.startswith('Knn'):
            val_model = KNeighborsClassifier(n_neighbors = int(selectors[0]))
        elif model_id.startswith('LR'):
            val_model = LogisticRegression(C=float(selectors[0]),\
                                            solver=selectors[1],
                                            class_weight=selectors[2],
                                            max_iter=200,
                                            random_state=514)  
                                    
        val_model.fit(X_train[features], y_train)
        validate_metrics[model_id] = compute_metrics(val_model, X_validate[features], y_validate).values

        score = val_model.score(X_validate[features], y_validate).round(4)

        val_descriptions.loc[idx+1] = {'Model': model_id,
            'Accuracy(Score)': score,
            'Type': descriptions.loc[idx]['Type'],
            'Features Used': f'{feat_set}',
            'Parameters': descriptions.loc[idx]['Parameters']
        }
        
    val_descriptions.insert(loc=2, column='Sensitivity', value=0)

    for idx in val_descriptions.index:
        model_id = val_descriptions.loc[idx]['Model']
        if model_id != 'Baseline':
            val_descriptions.loc[idx, 'Sensitivity'] = validate_metrics.T.loc[model_id]['True Negative Rate']        


    return val_descriptions, validate_metrics

def score_on_test(descriptions, subsets):
    ''' 
    Purpose:
        To score a model on the test subset
    ---
    Parameters:
        descriptions: descriptions of the best performing model from validation phase
        subsets: necessary data subsets for use with modeling
    ---
    Output:
        test_descriptions: model description for model used on test set
        test_metrics: evaluation metrics for test predictions
    ---
    '''    
    X_train=subsets[1]
    y_train=subsets[2]
    X_test = subsets[5]
    y_test = subsets[6]

    test_metrics = create_comp_chart()
    test_descriptions = create_description_chart(y_test)
        
    for idx in descriptions.index:
        
        model_id = descriptions.loc[idx]['Model']
        feat_set = descriptions.loc[idx]['Features Used'].strip('\[]\'').split('\', \'')
        selectors = get_selectors(descriptions.loc[idx]['Parameters'])

        features = []

        for feature in feat_set:
            features += [col for col in X_test.columns if feature in col]

        if model_id.startswith('DTC'):
            test_model = DecisionTreeClassifier(max_depth=int(selectors[0]),\
                                                random_state=514)
        elif model_id.startswith('RF'):
            test_model = RandomForestClassifier(max_depth=int(selectors[0]),\
                                            min_samples_leaf=int(selectors[1]),
                                            random_state=514)
        elif model_id.startswith('Knn'):
            test_model = KNeighborsClassifier(n_neighbors = int(selectors[0]))
        elif model_id.startswith('LR'):
            test_model = LogisticRegression(C=float(selectors[0]),\
                                            solver=selectors[1],
                                            class_weight=selectors[2],
                                            max_iter=200,
                                            random_state=514)  
                                    
        test_model.fit(X_train[features], y_train)

        test_metrics[model_id] = compute_metrics(test_model, X_test[features], y_test).values

        score = test_model.score(X_test[features], y_test).round(4)

        test_descriptions.loc[idx+1] = {'Model': model_id,
            'Accuracy(Score)': score,
            'Type': descriptions.loc[idx]['Type'],
            'Features Used': f'{feat_set}',
            'Parameters': descriptions.loc[idx]['Parameters']
        }

    return test_descriptions, test_metrics

def for_final_report(train, validate, test, feature_bank):
    ''' 
    Purpose:
        To create and train models for scoring on the train, validate, and test subsets
    ---
    Parameters:
        train: subset of data for model training purposes
        validate: subset of data for model validation purposes
        test: subset of data for model testing purposes
        feature_bank: list contain the combinations of features for use with modeling
    ---
    Output:
        test_metrics: evaluation metrics for test predictions
    ---
    '''    

    X_train, y_train, X_validate, y_validate, X_test, y_test = split_X_y(train, validate, test)

    subsets=[train, X_train, y_train, X_validate, y_validate, X_test, y_test]

    train_descriptions, train_metrics = train_models(y_train, feature_bank, subsets)

    top_10 = train_descriptions[train_descriptions['Accuracy(Score)'] > .68].\
        sort_values('Sensitivity', ascending=False).\
        head(10)

    # top_4 = train_descriptions[train_descriptions['Accuracy(Score)'] > .60].\
    #     sort_values('Sensitivity', ascending=False).\
    #     head(4)

    val_descriptions, validate_metrics = score_on_validation(top_10, subsets)

    top_4 = val_descriptions[(val_descriptions.Sensitivity > .30) & (val_descriptions['Accuracy(Score)'] > .60)].\
        sort_values('Sensitivity', ascending=False).\
        head(4)

    top_1 = val_descriptions[(val_descriptions.Sensitivity > .30) & (val_descriptions['Accuracy(Score)'] > .60)].\
        sort_values('Sensitivity', ascending=False).\
        head(1)

    test_descriptions, test_metrics = score_on_test(top_1, subsets)

    return test_metrics, top_4