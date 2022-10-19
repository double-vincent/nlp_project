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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
import nltk

def vectorize_split(df):

    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
        X_train, y_train, X_validate, y_validate, X_test, y_test: data subsets
    """

    tfidf = TfidfVectorizer()
    df['lemmatized'] = tfidf.fit_transform(df.lemmatized).todense()
    df['language_bigrams']= tfidf.fit_transform(df.language_bigrams).todense()

    scaler = preprocessing.MinMaxScaler()
    scaler.fit_transform(df[['word_count']])

    train_validate, test = train_test_split(df, test_size=.3, random_state=514, stratify=df['language'])
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514, stratify=train_validate['language'])

     # split data into Big X, small y sets 
    X_train = train.drop(columns=['language'])
    y_train = train.language

    X_validate = validate.drop(columns=['language'])
    y_validate = validate.language

    X_test = test.drop(columns=['language'])
    y_test = test.language

    return train, X_train, y_train, X_validate, y_validate, X_test, y_test

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
    baseline_accuracy = (y_train ==  'Python').mean()

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
    feat_set1 = ['word_count', 'lemmatized']

    feature_sets = [feat_set1]

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
    #confusionb = classification_report(y_df, y_pred)
    #pprint(confusionb)

    #assign results of confusion matrix to variables
    true_negative = confusion[0,0]
    false_positive = confusion[0,1]
    false_negative = confusion[1,0]
    true_positive = confusion[1,1]

    #accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + .0000001)

    #true positive rate / recall
    recall = true_positive / (true_positive +false_negative +.00000000001)

        # false positive rate
    try: 
        false_positive_rate = false_positive / (true_negative + false_positive + .0000001 )
    except RuntimeWarning:
        print(true_negative, false_positive)

    #true negative rate
    try:
        true_negative_rate = true_negative / (true_negative + false_positive + .0000001)
    except RuntimeWarning:
        print(true_negative, false_positive)

    #false negative rate
    false_negative_rate = false_negative / (false_negative + true_positive + .000000001) 

    #precision
    precision = true_positive / (true_positive + false_positive + .00000001)

    #f1-score
    f1_score = 2 * (precision * recall) / (precision + recall + .00000001)

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



def test_dtc(feat_set,\
        model_descriptions,
        comparison_chart,
        subsets):
    
    train=subsets[0]
    X_train=subsets[1]
    y_train=subsets[2]
    X_test=subsets[5]
    y_test=subsets[6]

    features = []
    for feature in feat_set:
        features += [col for col in train.columns if feature in col]

    selectors = list(product(np.arange(5,6,1)))

    for idx, item in enumerate(selectors):
        model_id = 'DTC_'+f'{idx}'
        dtc = DecisionTreeClassifier(max_depth=item[0],\
                                            random_state=514)
        
        dtc.fit(X_train[features], y_train)

        comparison_chart[model_id] = compute_metrics(dtc, X_test[features], y_test).values

        score = dtc.score(X_test[features], y_test).round(4)

        description = pd.DataFrame({'Model': model_id,
                                    'Accuracy(Score)': score,
                                    'Type': 'Decision Tree Classifier',
                                    'Features Used': f'{feat_set}',
                                    'Parameters': f'Depth: {item[0]}'},
                                    index=[0])

        model_descriptions = pd.concat([model_descriptions, description], ignore_index=True)

    return model_descriptions, comparison_chart

#created class in order to facilitate bigram and trigram creation
class code_language:
  def __init__(self, words, label:str):
    self.words = words
    self.label = label
    self.unique_to_language = set()

  def whole_words(self): 
    return pd.Series(self.words.split())

  def word_counts(self):
    return pd.Series(self.words.split()).value_counts()

  def unique_words(self):
    return set(pd.Series(self.whole_words().unique()))

  def bigrams(self):
    return pd.Series(list(nltk.bigrams(self.words.split())))

  def trigrams(self):
    return pd.Series(list(nltk.ngrams(self.words.split(), 3)))

  def readme_count(self):
    return df[df.language == self.label].word_count.count()



def get_test_score(df, subsets):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """

    test_description = pd.DataFrame({'Model': 'Baseline', \
    'Accuracy(Score)': 0.402299,
    'Type': 'Basic Baseline',
    'Features Used': 'Baseline Prediction',
    'Parameters': 'n/a'
    }, index=[0])
    feat_set = ['word_count', 'lemmatized', 'language_bigrams']

    test_comparisons = create_comp_chart()

    test_description, test_comparisons = test_dtc(feat_set, test_description, test_comparisons, subsets)

    return test_description