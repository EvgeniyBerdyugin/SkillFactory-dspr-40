import os

import numpy as np
import requests
import pandas as pd
import warnings
warnings.simplefilter('ignore')

df = pd.read_pickle(os.path.join('../src', 'data', 'test_data10000.pkl'))
city = pd.read_pickle(os.path.join('../src', 'data', 'city.pkl'))
state = pd.read_pickle(os.path.join('../src', 'data', 'state.pkl'))
zipcode = pd.read_pickle(os.path.join('../src', 'data', 'zipcode.pkl'))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))


def get_school_lists(schoolFacts):
    r_list = str(list((schoolFacts).values())[0])
    g_list = str(list((schoolFacts).values())[1]['Grades'])

    return [r_list, g_list]


def get_schools_stats(row):
    r_list = row.ratings
    nr_list = []
    zero_list = ['', 'NA', 'NR', 'None']
    for r in r_list:
        if '/' in r:
            r = r.split('/')[0]
        if r in zero_list:
            r = '0'
        nr_list.append(int(r))

    max_r, mean_r = None, None
    number = len(nr_list)
    if number > 0:
        max_r = max(nr_list)
        mean_r = round(sum(nr_list) / number, 2)

    return max_r, mean_r


def prepare_dataset(df, city, state, zipcode):
    df.status = df.status.str.lower()
    df.propertyType = df.propertyType.str.lower()
    df.city = df.city.str.lower()
    df.state = df.state.str.lower()
    df = df[~df.target.isna()]
    df = df[~(df.propertyType == 'lot/land')]
    df = df[~((df.propertyType == 'land') & (df.sqft == '0'))]
    df.status.fillna('unknown', inplace=True)
    df = df[~df.status.str.contains('rent')]
    df['target'] = df['target'].str.replace(r'[^0-9]', '', regex=True)
    df.target = df.target.astype('int64')
    df = df[(df.target > 18000) & (df.target < 6250000)]
    df['sqft'] = df['sqft'].str.replace(r'[^0-9]', '', regex=True)
    df['sqft'][df['sqft'] == ''] = np.NaN
    df['sqft'] = df['sqft'].astype('float')

    df = df[~df['sqft'].isna()]

    df.reset_index(drop=True, inplace=True)

    df = df.loc[:20]

    target = df[['target']]
    df.drop(
        ['status', 'baths', 'private pool', 'propertyType', 'street', 'fireplace', 'stories', 'mls-id', 'PrivatePool',
         'MlsId', 'target'], axis=1, inplace=True)

    df['atAGlanceFacts'] = df.homeFacts.apply(lambda x: eval(x)['atAGlanceFacts'])
    new_cols = list(df.from_dict(eval(df.homeFacts[0])['atAGlanceFacts'])['factLabel'])
    df[new_cols] = None
    df[new_cols] = df.atAGlanceFacts.apply(lambda x: pd.DataFrame(x)['factValue'])
    df.drop(['homeFacts', 'atAGlanceFacts', 'Remodeled year', 'Heating', 'Price/sqft', 'Parking'], axis=1, inplace=True)

    df['schoolFacts'] = df.schools.apply(lambda x: eval(x)[0])

    new_cols = ['ratings', 'grades']

    for i in range(len(df)):
        df.loc[i, new_cols] = get_school_lists(df.schoolFacts[i])

    df.ratings = df.ratings.apply(lambda x: eval(x))
    df.grades = df.grades.apply(lambda x: eval(x))

    new_cols = 'max_school_rating', 'mean_school_rating'

    for i, row in df.iterrows():
        df.loc[i, new_cols] = get_schools_stats(row)

    df.drop(['schools', 'schoolFacts', 'ratings', 'grades'], axis=1, inplace=True)

    nan_list = ['bath', 'sqft', 'acres']
    df['beds'] = df['beds'].fillna('unknown')
    df.beds = df.beds.str.lower()
    for i in nan_list:
        df['beds'][df['beds'].str.contains(i)] = 'unknown'

    df['beds'] = df['beds'].str.replace(r'[^0-9]', '', regex=True)
    df.beds[df['beds'] == ''] = 'unknown'
    df.beds[df['beds'] == 'unknown'] = 2
    df.beds = df.beds.astype('int')

    df['Year built'] = df['Year built'].astype('int64')

    df.Cooling = df.Cooling.str.lower()
    df.Cooling[df.Cooling.isna()] = 'no data'
    df.Cooling[df.Cooling == ''] = 'no data'
    df.Cooling[df.Cooling == 'none'] = 'no data'
    df.Cooling[df.Cooling == 'no data'] = 0
    df.Cooling[df.Cooling != 0] = 1

    df.lotsize = df.lotsize.str.lower()
    df.lotsize[df.lotsize == ''] = '0'
    df.lotsize[df.lotsize == 'no data'] = '0'
    df.lotsize = df.lotsize.fillna('0')
    df['lotsize_mes'] = None
    df['lotsize_mes'][df.lotsize.str.contains('acre')] = 1
    df.lotsize[df['lotsize_mes'] == 1]
    df['lotsize'] = df.lotsize.str.replace(r'[^.0-9]', '', regex=True)
    df.lotsize[df.lotsize.str.endswith('..')] = df.lotsize[df.lotsize.str.endswith('..')].apply(lambda x: x[:-2])
    df['lotsize'][df['lotsize'] == ''] = 0
    df['lotsize'][df['lotsize'] == '.'] = 0
    df['lotsize'] = df['lotsize'].astype('float')
    df['lotsize'][df['lotsize_mes'] == 1] = (df['lotsize'][df['lotsize_mes'] == 1] * 43560).round()
    df.drop('lotsize_mes', axis=1, inplace=True)

    df.max_school_rating = df.max_school_rating.fillna(0)
    df.mean_school_rating = df.mean_school_rating.fillna(0)

    df.columns = ['city', 'sqft', 'zipcode', 'beds', 'state', 'year_built', 'cooling',
                  'lotsize', 'max_school_rating', 'mean_school_rating']

    df['city_mean_price'] = df.city.apply(lambda x: city.loc[x, 'price_per_sqft'].round())
    df['city_mean_sqft'] = df.city.apply(lambda x: city.loc[x, 'sqft'].round())
    df['city_prop_count'] = df.city.apply(lambda x: city.loc[x, 'city'].round())

    df['state_mean_price'] = df.state.apply(lambda x: state.loc[x, 'price_per_sqft'].round())
    df['state_prop_count'] = df.state.apply(lambda x: state.loc[x, 'state'].round())

    df['zip_mean_sqft'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'sqft'].round())
    df['zip_prop_count'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'zipcode'].round())
    df['zip_baths_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'baths'].round(1))
    df['zip_beads_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'beds'].round())
    df['zip_year_bilt_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'year_built'].round())
    df['zip_heating_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'heating'].round(2))
    df['zip_cooling_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'cooling'].round(2))
    df['zip_number_of_school_mean'] = df.zipcode.apply(lambda x: zipcode.loc[x, 'number_of_school'].round(2))

    df.drop(['city', 'state', 'zipcode'], axis=1, inplace=True)

    df = df[['beds', 'year_built', 'cooling', 'lotsize', 'sqft',
             'max_school_rating', 'mean_school_rating', 'city_mean_price',
             'city_mean_sqft', 'city_prop_count', 'state_mean_price',
             'state_prop_count', 'zip_mean_sqft', 'zip_prop_count',
             'zip_baths_mean', 'zip_beads_mean', 'zip_year_bilt_mean',
             'zip_heating_mean', 'zip_cooling_mean',
             'zip_number_of_school_mean']]
    return df, target


if __name__ == '__main__':
    data, target = prepare_dataset(df, city, state, zipcode)
    # выполняем POST-запрос на сервер по эндпоинту add с параметром json
    r = requests.post('http://localhost:5000/predict', json=data.values.tolist())
    # выводим статус запроса
    print('Status code: {}'.format(r.status_code))
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        # выводим результат на экран
        pred = eval(r.json()['prediction'])

        print(f'MAPE: {mape(target.target, pred)}')
    else:
        # если запрос завершён с кодом, отличным от 200,
        # выводим содержимое ответа
        print(r.text)

# добавить расчет точности