import argparse
import pandas as pd
import os
import requests as requests


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Input properety paremeters')
    parser.add_argument('-b', type=int, default=2, help='number of bedrooms.')
    parser.add_argument('-yb', type=int, help='year of built.')
    parser.add_argument('-c', type=int, help='cooling, 0 - no, 1 - yes.')
    parser.add_argument('-ls', type=float, default=0, help='lotsize, sqft.')
    parser.add_argument('-sf', type=float, help='sqft.')
    parser.add_argument('-sr', type=str, help='near schools ratings in [].')
    parser.add_argument('-city', type=str, help='city.')
    parser.add_argument('-st', type=str, help='state in short form.')
    parser.add_argument('-zc', type=str, help='zipcode.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    city = pd.read_pickle(os.path.join('../src', 'data', 'city.pkl'))
    state = pd.read_pickle(os.path.join('../src', 'data', 'state.pkl'))
    zipcode = pd.read_pickle(os.path.join('../src', 'data', 'zipcode.pkl'))

    enter_dict = {'beds': args.b,
                  'year_built': args.yb,
                  'cooling': args.c,
                  'lotsize': args.ls,
                  'sqft': args.sf
                  }

    try:
        ratings = eval(args.sr)
        enter_dict['max_school_rating'] = max(ratings)
        enter_dict['mean_school_rating'] = sum(ratings) / len(ratings)

        enter_dict['city_mean_price'] = round(city.loc[args.city, 'price_per_sqft'])
        enter_dict['city_mean_sqft'] = round(city.loc[args.city, 'sqft'])
        enter_dict['city_prop_count'] = round(city.loc[args.city, 'city'])

        enter_dict['state_mean_price'] = round(state.loc[args.st, 'price_per_sqft'])
        enter_dict['state_prop_count'] = round(state.loc[args.st, 'state'])

        enter_dict['zip_mean_sqft'] = round(zipcode.loc[args.zc, 'sqft'])
        enter_dict['zip_prop_count'] = round(zipcode.loc[args.zc, 'zipcode'])
        enter_dict['zip_baths_mean'] = round(zipcode.loc[args.zc, 'baths'], 1)
        enter_dict['zip_beads_mean'] = round(zipcode.loc[args.zc, 'beds'])
        enter_dict['zip_year_bilt_mean'] = round(zipcode.loc[args.zc, 'year_built'])
        enter_dict['zip_heating_mean'] = round(zipcode.loc[args.zc, 'heating'], 2)
        enter_dict['zip_cooling_mean'] = round(zipcode.loc[args.zc, 'cooling'], 2)
        enter_dict['zip_number_of_school_mean'] = round(zipcode.loc[args.zc, 'number_of_school'], 2)
    except:
        print('check entered values')

    r = requests.post('http://localhost:5000/predict', json=[list(enter_dict.values())])

    print('Status code: {}'.format(r.status_code))

    if r.status_code == 200:

        print(f"Estimate property cost: {eval(r.json()['prediction'])[0]}")
    else:
        print(r.text)

# check
# python custom.py -b 2 -yb 2010 -c 1 -ls 10000 -sf 2500 -sr [3,4,5] -city houston -st tx -zc 77068