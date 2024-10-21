import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
from thefuzz import fuzz

def load_and_prepare_test_data():
    def get_full_text(df):
        def covnert_row_to_text(row):
            values = [', '.join(x) for x in row.values()]
            keys = row.keys()
            text = ''
            for j, k in zip(keys, values):
                text += str(j) + ': ' + str(k) + '\n'
            return text

        df['full_text'] = df['characteristic_attributes_mapping'].apply(covnert_row_to_text)
        cats_joined = df['categories'].apply(lambda x: ', '.join(x.values()))
        df['full_text'] = cats_joined + '\n' + df['full_text']
        return df[['variantid', 'full_text']]
    
    descriptions_path = './data/test/text_and_bert_test.parquet'
    attributes_path = './data/test/attributes_test.parquet'
    test_path = './data/test/test.parquet'
    resnet_path = './data/test/resnet_test.parquet'

    attributes = pd.read_parquet(attributes_path)
    attributes['characteristic_attributes_mapping'] = attributes['characteristic_attributes_mapping'].apply(json.loads)
    attributes['categories'] = attributes['categories'].apply(json.loads)
    attributes_text = get_full_text(attributes)
    descriptions = pd.read_parquet(descriptions_path)
    test = pd.read_parquet(test_path)

    test = test.merge(attributes_text, left_on='variantid1', right_on='variantid', how='left').drop('variantid', axis=1)
    test = test.rename(columns={'full_text': 'full_text_var1'})
    test = test.merge(attributes_text, left_on='variantid2', right_on='variantid', how='left').drop('variantid', axis=1)
    test = test.rename(columns={'full_text': 'full_text_var2'})

    test = test.merge(descriptions, left_on='variantid1', right_on='variantid', how='left').drop('variantid', axis=1)
    test.drop('description', axis=1, inplace=True)
    test = test.rename(columns={'name': 'name_var1'})
    test = test.merge(descriptions, left_on='variantid2', right_on='variantid', how='left').drop('variantid', axis=1)
    test.drop('description', axis=1, inplace=True)
    test = test.rename(columns={'name': 'name_var2'})

    test['text'] = test['name_var1'] + '\n' + test['full_text_var1'] + '=&=\n' + test['name_var2'] + '\n' + test['full_text_var2']
    test = test.drop(['full_text_var1', 'full_text_var2', 'name_var1', 'name_var2'], axis=1)

    resnet = pd.read_parquet(resnet_path)
    test = test.merge(resnet, left_on = 'variantid1', right_on = 'variantid', suffixes=('_var1', '_var2')).drop('variantid', axis=1)
    test = test.merge(resnet, left_on = 'variantid2', right_on = 'variantid', suffixes=('_var1', '_var2')).drop('variantid', axis=1)


    test = test.merge(attributes, left_on = 'variantid1', right_on = 'variantid', suffixes=('_var1', '_var2')).drop('variantid', axis=1)
    test = test.merge(attributes, left_on = 'variantid2', right_on = 'variantid', suffixes=('_var1', '_var2')).drop('variantid', axis=1)

    return test


def get_test_embeddings(test_data):
    def pad(data):
        tensors = [torch.tensor(x) if len(x) <= 512 else torch.tensor(x)[:512] for x in data]
        return pad_sequence(tensors, batch_first=True)
    
    preds = []
    model = AutoModelForSequenceClassification.from_pretrained("models/e5_v2/", num_labels=2)
    model = model.bert
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('models/e5_v2/')
    
    batch_size = 1
    end = len(test_data)
    idx = [(range(i, i + batch_size)) for i in range(0, end - batch_size + 1, batch_size)]
    if np.mod(end, batch_size) != 0:
        idx.append(range(idx[-1][-1] + 1, end))

    with torch.no_grad():
        for i in idx:
            input_ = tokenizer(list(test_data['text'].iloc[i]))
            preds.append((model(pad(input_['input_ids']), pad(input_['attention_mask'])).pooler_output))
    
    return np.vstack(preds)


def load_model_and_predict(path, test_data):
    model = CatBoostClassifier()
    model.load_model(path, format = 'cbm')
    return model.predict_proba(test_data)[:, 1]


def cosine_similarity(X, Y):
    normed_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    normed_Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return np.sum(normed_X * normed_Y, axis=1)

def get_fuzz_score(row):
    splits = row.split('=&=\n')
    return fuzz.token_sort_ratio(splits[0], splits[1])

def is_valid_dimension_format(dim_str):
    """Helper for checking if string is in the dimension format"""
    pattern = r'^\s*\d+(?:\.\d+)?\s*([xх×*]\s*\d+(?:\.\d+)?\s*)+$'
    standardized_str = re.sub(r'[х×*]', 'x', dim_str.lower())
    return bool(re.match(pattern, standardized_str))

def compare_dimensions(dim_str1, dim_str2):
    """Helper for checking if the dimensions in such strings don't match"""
    standardized_str1 = re.sub(r'[х×*]', 'x', dim_str1.lower())
    numbers1 = re.findall(r'\d+(?:\.\d+)?', standardized_str1)
    numbers1 = sorted(list(map(float, numbers1)))

    standardized_str2 = re.sub(r'[х×*]', 'x', dim_str2.lower())
    numbers2 = re.findall(r'\d+(?:\.\d+)?', standardized_str2)
    numbers2 = sorted(list(map(float, numbers2)))
    return numbers1 != numbers2

def is_number(s):
    """Helper function to check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def compare_select_keys_in_dataframe(df, col1, col2, selected_categories):
    results = []
    categories = selected_categories['category'].values

    for index, row in df.iterrows():
        dict1 = row[col1]
        dict2 = row[col2]
        
        for key, value in dict1.items():
            if key not in categories:
                continue
            if key in dict2:
                value_in_col2 = dict2[key]

                if key == 'ISBN':
                    value = [value[0].replace('-', '')]
                    value_in_col2 = [value_in_col2[0].replace('-', '')]
                    
                if key == 'Образец цвета':
                    not_match = value != value_in_col2
                    results.append({
                        'key': key,
                        'value_in_col1': value,
                        'value_in_col2': value_in_col2,
                        'not_match': not_match,
                        'row_index': index
                    })
                    continue

                if key == 'Размеры, мм':
                    if is_valid_dimension_format(value[0]) and is_valid_dimension_format(value_in_col2[0]):
                        not_match = compare_dimensions(value[0], value_in_col2[0])
                        results.append({
                        'key': key,
                        'value_in_col1': value,
                        'value_in_col2': value_in_col2,
                        'not_match': not_match,
                        'row_index': index
                        })
                        continue
                
                if (value[0].isdigit() or is_number(value[0])) and \
                   (value_in_col2[0].isdigit() or is_number(value_in_col2[0])):
                    not_match = value != value_in_col2
                    results.append({
                        'key': key,
                        'value_in_col1': value,
                        'value_in_col2': value_in_col2,
                        'not_match': not_match,
                        'row_index': index
                    })
    
    results_df = pd.DataFrame(results)
    return results_df

def main():
    test_data = load_and_prepare_test_data()
    attributes_embd = get_test_embeddings(test_data)

    fuzz_score = test_data['text'].apply(get_fuzz_score).values.reshape(-1,1)

    X = test_data.apply(
    lambda row: np.concatenate([
        [row['variantid1']],
        [row['variantid2']],
        [abs(row['variantid1'] - row['variantid2'])],
        row['name_bert_64_x'],
        row['name_bert_64_y'],
        row['main_pic_embeddings_resnet_v1_var1'][0],
        row['main_pic_embeddings_resnet_v1_var2'][0],
    #         row['pic_embeddings_resnet_v1_var1'],
    #         row['pic_embeddings_resnet_v1_var2']
        ]),
        axis=1
    )
    X = np.vstack(X)
    X = np.concatenate([attributes_embd, X], axis=1)

    X_embed1 = np.vstack([x[0] if x is not None else [np.nan for i in range(128)] for x in test_data['pic_embeddings_resnet_v1_var1'].values])
    X_embed2 = np.vstack([x[0] if x is not None else [np.nan for i in range(128)] for x in test_data['pic_embeddings_resnet_v1_var2'].values])

    bert_64_similarity = cosine_similarity(np.vstack(test_data['name_bert_64_x'].values), np.vstack(test_data['name_bert_64_y'])).reshape(-1,1)
    main_pic_similarity = cosine_similarity(np.vstack([x[0] for x in test_data['main_pic_embeddings_resnet_v1_var1'].values]),\
                                            np.vstack([x[0] for x in test_data['main_pic_embeddings_resnet_v1_var2'].values])).reshape(-1,1)
    
    selected_categories = pd.read_csv('selected_categories_train.csv')
    summary = compare_select_keys_in_dataframe(test_data, 'characteristic_attributes_mapping_var1', 'characteristic_attributes_mapping_var2', selected_categories)
    name_to_ind = {k: v for v, k in zip(range(len(selected_categories)), selected_categories['category'].values)}
    A = np.zeros(shape=(len(test_data), len(name_to_ind)))
    res = summary[summary['not_match'] != False]
    A[res['row_index'].values, res['key'].apply(lambda x: name_to_ind.get(x))] = -1
    res = summary[summary['not_match'] != True]
    A[res['row_index'].values, res['key'].apply(lambda x: name_to_ind.get(x))] = 1
    
    X = np.concatenate([X, X_embed1, X_embed2, fuzz_score, bert_64_similarity, main_pic_similarity, A], axis=1)

    predictions = load_model_and_predict('models/catboost_small.cbm', X)

    submission = pd.DataFrame({
        'variantid1': test_data['variantid1'],
        'variantid2': test_data['variantid2'],
        'target': predictions
    })

    submission.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    main()
