import click
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from pathlib import Path
import logging
from dotenv import find_dotenv, load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# nltk.download('punkt')

def count_total_words(text):
    return len(word_tokenize(text))

def create_tfidf_features(df, text_column='v2', max_features=4000):
    logger = logging.getLogger(__name__)
    logger.info('Creating word count feature')

    df[text_column] = df[text_column].astype(str)
    df['num_words'] = df[text_column].apply(count_total_words)
    
    # Create TF-IDF features
    logger.info('Creating TF-IDF features')
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        lowercase=True,
    )
    bow_matrix = vectorizer.fit_transform(df[text_column])
    
    logger.info('Converting TF-IDF matrix to dataframe')
    bow_matrix_df = pd.DataFrame(bow_matrix.toarray())
    num_w_df = pd.DataFrame(df['num_words'])
    y_df = pd.DataFrame(df['v1'])
    
    bow_matrix_df = bow_matrix_df.reset_index(drop=True)
    num_w_df = num_w_df.reset_index(drop=True)
    final_df = pd.concat([bow_matrix_df, num_w_df, y_df], axis=1)
    
    final_df = final_df.rename(str, axis="columns")
    
    return final_df, vectorizer


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def build_features(input_filepath, output_filepath):
    """Build features from processed data."""
    logger = logging.getLogger(__name__)
    
    logger.info('Loading processed data')
    df = pd.read_csv(input_filepath)
    
    # Create features
    final_df, vectorizer = create_tfidf_features(df)

    
    final_df.to_csv(output_filepath, index=False)
    logger.info(f'Features built and saved to {output_filepath}')

    output_path = Path(output_filepath)
    vectorizer_path = output_path.parent / 'tfidf_vectorizer.joblib'
    dump(vectorizer, vectorizer_path)
    logger.info(f'Vectorizer saved to {vectorizer_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    input_filepath = r'D:\spam_detection\spam_detection\data\processed\cleaned_spam_data.csv'
    output_filepath = r'D:\spam_detection\spam_detection\data\interim\spam_features.csv'

    # input_filepath = r'..\..\data\processed\cleaned_spam_data.csv'
    # output_filepath = r'..\..\data\interim\spam_features.csv'

    build_features(input_filepath, output_filepath)