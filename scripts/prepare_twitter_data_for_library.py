from text_clsf_lib.utils.files_io import write_json
from typing import List

INPUT_CSV_FILE_PATH = 'data/Data_tweets.csv'
OUTPUT_JSON_DATA_PATH = 'data/data_tweets.json'

POSITIVE_TWEET = '4'

def prepare_twitter_data_for_library(csv_file_path: str = INPUT_CSV_FILE_PATH) -> List[dict]:
    twitter_data = read_twitter_input_data(csv_file_path)
    return extract_important_features(twitter_data)

def read_twitter_input_data(csv_file_path: str) -> List[dict]:
    twitter_input_data = []
    with open(INPUT_CSV_FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            twitter_input_data.append({
                'polarity': line_split[1],
                'id': line_split[2],
                'date': line_split[3],
                'query': line_split[4],
                'user': line_split[5],
                'text': ','.join(line_split[6:])
            })
    return twitter_input_data



def extract_important_features(twitter_data: List[dict]) -> List[dict]:
    # There are no neutral tweets in the dataset, that's why this task becomes a binary classification problem.
    data = [{'polarity': 1 if sample['polarity'] == POSITIVE_TWEET else 0,
             'text': sample['text']} for sample in twitter_data]
    return data


if __name__ == '__main__':
    twitter_input_data = prepare_twitter_data_for_library()
    write_json(out_path=OUTPUT_JSON_DATA_PATH, data=twitter_input_data)
