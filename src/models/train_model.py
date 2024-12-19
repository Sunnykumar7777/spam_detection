import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
from pathlib import Path
import joblib
import click
from dotenv import find_dotenv, load_dotenv


def load_features(features_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Loading features')
    return pd.read_csv(features_filepath)


def train_model(X_train, y_train):
    logger = logging.getLogger(__name__)
    logger.info('Training Random Forest model')
    
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def evaluate_model(model, X_test, y_test):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating model')
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    logger.info(f'Model accuracy: {accuracy:.2f}')
    
    return accuracy


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(features_filepath, model_filepath):
    logger = logging.getLogger(__name__)
    
    final_df = load_features(features_filepath)
    
    X = final_df.drop(['v1'], axis=1)
    y = final_df['v1']
    
    #Split the data
    logger.info('Splitting data into train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    #Train model
    model = train_model(X_train, y_train)
    
    #Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    #Save the model
    logger.info(f'Saving model to {model_filepath}')
    Path(model_filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filepath)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    features_filepath = r'D:\spam_detection\spam_detection\data\interim\spam_features.csv'
    model_filepath = r'D:\spam_detection\spam_detection\models\random_forest_spam.joblib'

    # input_filepath = r'..\..\data\interim\spam_features.csv'
    # output_filepath = r'..\..\models\random_forest_spam.joblib'

    main(features_filepath, model_filepath)