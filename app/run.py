'''
Use the following results to categorize a give mesage with flask application.

1. The ETL process of message data.
2. THe trained model based on the results from (1).

The application is running on http://0.0.0.0:3001/ 
'''
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('main_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Plot the counts for all the categories 
    df_categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories_name = df_categories.sum().index
    categories_counts = df_categories.sum().values

    # plot the counts for all the categories (direct genre) 
    df_genre = df.drop(['id', 'message', 'original'], axis=1)
    df_genre = df_genre[df_genre['genre'] == 'direct'].drop('genre', axis=1)
    categories_name_direct = df_genre.sum().index
    categories_counts_direct = df_genre.sum().values

    # plot the counts for all the categories (news genre) 
    df_news = df.drop(['id', 'message', 'original'], axis=1)
    df_news = df_news[df_news['genre'] == 'news'].drop('genre', axis=1)
    categories_name_news = df_news.sum().index
    categories_counts_news = df_news.sum().values

    # plot the counts for all the categories (social genre) 
    df_social = df.drop(['id', 'message', 'original'], axis=1)
    df_social = df_social[df_social['genre'] == 'social'].drop('genre', axis=1)
    categories_name_social = df_social.sum().index
    categories_counts_social = df_social.sum().values

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_name,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name_direct,
                    y=categories_counts_direct
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories (direct genre)',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Category (direct genre)"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name_news,
                    y=categories_counts_news
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories (news genre)',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Category (news genre)"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name_social,
                    y=categories_counts_social
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories (social genre)',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Category (social genre)"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()