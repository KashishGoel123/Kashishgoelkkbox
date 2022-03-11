import streamlit as st
import pandas as pd
import numpy as np
import pickle



st.title('KKBOX Churn Prediction')

sentence = st.text_input('Input your mnso:')

members_data = pd.read_csv('members_v3.csv')
transactions_data = pd.read_csv('transactions_v2.csv')
user_logs_data = pd.read_csv('user_logs_v2.csv')


a = members_data[members_data.isin([sentence]).any(axis=1)]

b = transactions_data[transactions_data.isin([sentence]).any(axis=1)]

c = user_logs_data[user_logs_data.isin([sentence]).any(axis=1)]

result = pd.concat([a, b, c], ignore_index=True, sort=False)
def preprocess(data):
    # imputing 0 in place of nan values in the city column
    data['city'] = data['city'].fillna(0)
    
    # removing outliers
    data['bd'] = data['bd'].apply(lambda x: x if (x < 69.0) and (x > 0.0) else np.nan)
    # imputing 28 as age instead of nan
    data['bd'] = data['bd'].fillna(28.0)
    
    # replacing male with 1 in gender
    data['gender'] = data['gender'].replace(to_replace='male', value=1)
    # replacing male with 2 in gender
    data['gender'] = data['gender'].replace(to_replace='female', value=2)
    # replacing nan with 0 in gender
    data['gender'] = data['gender'].fillna(0)
    
    # replace 0 instead of nan in registered_via
    data['registered_via'] = data['registered_via'].fillna(0)
    
    # filling median date in place of nan in the df
    data['registration_init_time'] = data['registration_init_time'].fillna(20151010.0)
    # converting float date to datetime
    # data['registration_init_time'] = pd.to_datetime(data['registration_init_time'], format='%Y%m%d')
    
    # replace 0 instead of nan in registered_via
    data['registered_via'] = data['registered_via'].fillna(0)
    
    # removing outliers
    data['payment_plan_days'] = data['payment_plan_days'].apply(lambda x: x if (x <= 30.0) else np.nan)
    # imputing 30 in place of nan in payment_plan_days
    data['payment_plan_days'] = data['payment_plan_days'].fillna(30.0)
    
    # removing outliers
    data['plan_list_price'] = data['plan_list_price'].apply(lambda x: x if (x <= 180.0) else np.nan)
    # imputing 149 in place of nan in plan_list_price
    data['plan_list_price'] = data['plan_list_price'].fillna(149.0)
    
    # imputing 0 in place of nan value in payment_method_id
    data['payment_method_id'] = data['payment_method_id'].fillna(0)
    
    # removing outliers
    data['actual_amount_paid'] = data['actual_amount_paid'].apply(lambda x: x if (x <= 180.0) else np.nan)
    # imputing 149 in place of nan in actual_amount_paid
    data['actual_amount_paid'] = data['actual_amount_paid'].fillna(149.0)
    
    # imputing 2 in place of nan values in is_auto_renew
    data['is_auto_renew'] = data['is_auto_renew'].fillna(2)
    
    # filling median date in place of nan in the df
    data['transaction_date'] = data['transaction_date'].fillna(20170311.0)
    # converting float date to datetime
    # data['transaction_date'] = pd.to_datetime(data['transaction_date'], format='%Y%m%d')
    
    # filling median date in place of nan in the df
    data['membership_expire_date'] = data['membership_expire_date'].fillna(20170421.0)
    # converting float date to datetime
    # data['membership_expire_date'] = pd.to_datetime(data['membership_expire_date'], format='%Y%m%d')
    
    # imputing 2 in place of nan values in is_cancel
    data['is_cancel'] = data['is_cancel'].fillna(2)
    
    # filling median date in place of nan in the df
    data['date'] = data['date'].fillna(20170316.0)
    # converting float date to datetime
    # data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    
    # removing outliers
    data['num_25'] = data['num_25'].apply(lambda x: x if (x <= 15.0) else np.nan)
    # now I can impute 2 instead of nan in num_25
    data['num_25'] = data['num_25'].fillna(2.0)
    
    # removing outliers
    data['num_50'] = data['num_50'].apply(lambda x: x if (x <= 4.0) else np.nan)
    # now I can impute 0 instead of nan in num_50
    data['num_50'] = data['num_50'].fillna(1.0)

    # removing outliers
    data['num_75'] = data['num_75'].apply(lambda x: x if (x <= 3.0) else np.nan)
    # now I can impute 0 instead of nan in num_75
    data['num_75'] = data['num_75'].fillna(0)

    # removing outliers
    data['num_985'] = data['num_985'].apply(lambda x: x if (x <= 3.0) else np.nan)
    # now I can impute 0 instead of nan in num_985
    data['num_985'] = data['num_985'].fillna(0)

    # removing outliers
    data['num_100'] = data['num_100'].apply(lambda x: x if (x <= 74.0) else np.nan)
    # now I can impute 14 instead of nan in num_100
    data['num_100'] = data['num_100'].fillna(17.0)
    
    # removing outliers
    data['num_unq'] = data['num_unq'].apply(lambda x: x if (x <= 68.0) else np.nan)
    # now I can impute 16 instead of nan in num_unq
    data['num_unq'] = data['num_unq'].fillna(18.0)

    # removing outliers
    data['total_secs'] = data['total_secs'].apply(lambda x: x if (x <= 19167.549700000025) else np.nan)
    # now I can impute 3880.765 instead of nan in total_secs
    data['total_secs'] = data['total_secs'].fillna(4588.99)
    
preprocess(result)

result["city"].replace({0.0: 0.083907, 1.0: 0.069631 , 3.0: 0.131210 , 4.0: 0.133819 , 5.0: 0.132687 , 6.0: 0.127799 , 7.0: 0.106309 , 8.0: 0.142899 , 9.0: 0.125337 , 10.0: 0.132318 , 11.0: 0.107256 , 12.0: 0.139945 , 13.0: 0.127451 , 14.0: 0.126532 , 15.0: 0.122181 , 16.0: 0.117568 , 17.0: 0.089775 , 18.0: 0.112245 , 19.0: 0.140959 , 20.0: 0.079720 , 21.0: 0.143717 , 22.0 : 0.119596}, inplace=True)

result["gender"].replace({0.0: 0.074407, 1.0: 0.127283 , 2.0: 0.127856}, inplace=True)

result["payment_method_id"].replace({0.0   :  0.109574 , 3.0   :  1.000000 , 6.0   :  1.000000 , 8.0   :  0.852273 , 10.0  :  0.196382 , 11.0  :  0.015400 , 12.0  :  0.947433 , 13.0  :  0.994592 , 14.0  :  0.052452 , 15.0  :  0.910628 , 16.0  :  0.165079 , 17.0  :  0.895794 , 18.0  :  0.009904 , 19.0  :  0.021050 , 20.0  :  0.991945 , 21.0  :  0.060914 , 22.0  :  0.993191 , 23.0  :  0.057582 , 26.0  :  0.609262 , 27.0  :  0.033693 , 28.0  :  0.238390 , 29.0  :  0.092717 , 30.0  :  0.075253 , 31.0  :  0.023816 , 32.0  :  0.960869 , 33.0  :  0.035524 , 34.0  :  0.032791 , 35.0  :  0.863726 , 36.0  :  0.093306 , 37.0  :  0.026523 , 38.0  :  0.276168 , 39.0  :  0.055477 , 40.0  :  0.065581 , 41.0  :  0.054938}, inplace=True)

result["is_cancel"].replace({0.0  :  0.085300 , 1.0  :  0.371507 , 2.0  :  0.639022}, inplace=True)

result["registered_via"].replace({0.0   :  0.072946 , 3.0   :  0.146431 , 4.0   :  0.194327 , 7.0   :  0.050228 , 9.0   :  0.120079 , 13.0  :  0.045631}, inplace=True)

result["is_auto_renew"].replace({0.0  :  0.085300 , 1.0  :  0.371507 , 2.0  :  0.639022}, inplace=True)


del result["msno"]

result.rename(columns = {'city':'city_mean_enc', 'gender':'gender_mean_enc',
                              'payment_method_id':'payment_method_id_mean_enc','is_cancel':'is_cancel_mean_enc' , 'registered_via':'registered_via_mean_enc' , 'is_auto_renew':'is_auto_renew_mean_enc'}, inplace = True)

result= result.astype(float)


pickle_in = open('LGBM.pkl', 'rb')
classifier = pickle.load(pickle_in)


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(result)

st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='result.csv',
     mime='text/csv',
 )

prediction = classifier.predict(result)

submit = st.button('Predict')
if submit:
        prediction = classifier.predict(result)
        if prediction[0] == 0:
            st.write('Not Churned')
        else:
            st.write("Churned")


