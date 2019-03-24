# Disaster Response Pipeline Project

### Project Introductions
This project is originally based on "Disaster Response Pipeline" from Udacity.
The goal is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.    

There are mainly three parts in the project:
1. Build the ETL pipeline (in the `data` folder)
2. Build the Machine Learning Pipeline (in the `models` folder)
3. Build the Flask Web Application (in the `app` folder)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### File Descriptions
- app    
| - template    
| |- master.html  # main page of web app    
| |- go.html  # classification result page of web app    
|- run.py  # Flask file that runs app    

- data    
|- disaster_categories.csv  # data to process    
|- disaster_messages.csv  # data to process    
|- process_data.py # The main ETL process is here    
|- DisasterResponse.db   # database to save clean data to    

- models    
|- train_classifier.py # The main ML process is here    
|- classifier.pkl  # saved model    

- README.md    

