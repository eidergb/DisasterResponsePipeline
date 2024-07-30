# Disaster Response Pipeline Project

## Project Description
This project aims to build a classification model to categorize real-time messages during disasters. The model categorizes messages into 36 predefined categories, such as medical assistance, search and rescue, food, among others. The web application allows users to input a message and automatically receive the corresponding classification.

## Project Structure
- `app`: Contains the web application files based on Flask.
    - `template`: Contains the HTML templates.
        - `master.html`: Main page of the web application.
        - `go.html`: Classification result page.
    - `run.py`: Flask script that runs the web application.

- `data`: Contains the input data and data processing scripts.
    - `disaster_categories.csv`: Category data.
    - `disaster_messages.csv`: Message data.
    - `process_data.py`: Script for processing data and saving it to an SQLite database.
    - `DisasterResponse.db`: Database to store the processed data.

- `models`: Contains the trained classification model.
    - `train_classifier.py`: Script for training the classification model.
    - `classifier.pkl`: Trained model saved as a pickle file.

- `README.md`: Project documentation file.

## Instructions
1. Setting up the Database and Model.

    - To prepare the database and train the model, run the following commands from the project's root directory.

        - To run the ETL (Extract, Transform, Load) pipeline that cleans the data and stores it in the database:
                `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

        - To train the classification model and save it:
                `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Running the Web Application.

    - To start the web application, run the following command in the app directory:
        `python run.py`

    - Then, open http://0.0.0.0:3001/ in your browser to interact with the web app.

## Application Usage
The web application allows users to input disaster-related messages and receive a classification based on 36 categories. The interface presents graphs showing the distribution of message genres, classifications, and more.

## Dependencies
Ensure you have the following libraries installed:

    - Python 3
    - pandas
    - numpy
    - sqlalchemy
    - scikit-learn
    - nltk
    - flask
    - plotly

Install all dependencies using the requirements.txt file or by manually installing them.