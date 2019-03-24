import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    Load message data from SQL database to a dataframe.
    The independent varlables and dependent variables will then be separated into X and Y.
    The category lists will be saved as "category_names".

    INPUT: database_filepath
    OUTPUT: X, Y, category_names
    '''    
    database_path = 'sqlite:///' + database_filepath
    engine = create_engine(database_path)
    df = pd.read_sql_table('main_data', con=engine)
    Y = df.drop(['genre', 'id', 'message', 'original'], axis=1)
    X = df['message']
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Transfer a text into a vetorized frequency array (clean_tokens).

    INPUT: text
    OUTPUT: clean_tokens
    ''' 
    clean_tokens = []
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    '''
    Build a model with pipeline that includes 
    1. CountVectorizer transformer, 
    2. TfidfTransformer 
    3. MultiOutputClassifier (random forest algorithm).

    Besides, GridSearchCV is also used.
    In order to fasten the process, only one parameter is used.
    (multi_clf__estimator__n_jobs)

    INPUT: None
    OUTPUT: GridSearchCV Model
    '''     
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters for GridSearch
    parameters = {
        'multi_clf__estimator__n_jobs': [2]
        #'estimator__features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'estimator__features__text_pipeline__vect__max_df': (0.5, 1.0),
        #'estimator__features__text_pipeline__vect__max_features': (5000, 10000)
        #'estimator__multi_clf__estimator__n_estimators': [50, 200] 
        #'estimator__multi_clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate a give model.

    INPUT: model, X_test, Y_test, category_names
    OUTPUT: None (will print out classification_report for each category)
    '''         
    Y_pred = model.predict(X_test)

    # Evaluate the performance in each categories
    i = 0
    for column in category_names:
        print('\n')
        print('-----','Evaluation of ' , column, ': -----')
        print(classification_report(Y_test[column], Y_pred[:,i]), '\n')
        print('\n')
        i += 1


def save_model(model, model_filepath):
    '''
    Save a model to a pickle file.

    INPUT: model, model_filepath
    OUTPUT: None (a picle file will be created with the given model_filepath)
    ''' 
    with open(model_filepath, mode='wb') as ml:
        pickle.dump(model, ml)


def main():
    '''
    Load data and then train a model, print out the evaluation results,
    and then save the model to a pickle file.

    INPUT: None
    OUTPUT: None
    ''' 
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        

        # debug
        print('\n', 'pipeline.get_params().keys', model.get_params().keys(), '\n',)


        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()