import requests
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import cryptocompare
import datetime
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
import requests
from math import sqrt
from datetime import timedelta
import random
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def generate_dataframe(cryptos): 
    # setup dataframe
    cols = ['Ticker','Position', 'Market', 'WAP', 'UPL', 'RPL', 'Cash', "P/L", "% Shares", "% Dollars", "RNN", "LSTM"]
    blotter = pd.DataFrame(columns = cols)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # loop through function get_price
    for crypto in cryptos.keys():
        Ticker = crypto
        Position = 0
        Market = get_price(crypto)
        WAP = 0
        UPL = 0
        RPL = 0
        Cash = 0
        PL = 0
        Shares = 0
        Dollars = 0 
        RNN = main_RNN(crypto)
        LSTM = run_LSTM(crypto)
        temp_blotter = pd.Series([Ticker, Position, as_currency(Market), WAP,  UPL, RPL, Cash, PL, Shares, Dollars, RNN, LSTM])
        df_blotter = pd.DataFrame([list(temp_blotter)], columns = cols)
        blotter = blotter.append(df_blotter)

    cash = pd.Series(["Cash", as_currency(10000000), as_currency(10000000), " ",  " ", " ", " ", " ", " ", " ", " ", " "])
    cash_df = pd.DataFrame([list(cash)], columns = cols)
    blotter = blotter.append(cash_df)   
    blotter = blotter.set_index('Ticker')
    return blotter, RNN

def get_price(crypto):
    dict_price = cryptocompare.get_price(crypto)
    string_price = str(dict_price)
    regex_price = re.findall('\d+[\,\.]{1}\d{1,2}', string_price)
    if (regex_price == []):
        new_regex_price = re.findall('\d+', str(string_price))
        price = float(new_regex_price[0])
    else:
        price = float(regex_price[0])
    return(price)

def as_float(number):
    number = str(number)
    if number != 0 and ',' in number:
        number = number.replace(",","")
        number = float(number.replace("$",""))
    elif number != 0:
        number = float(number.replace("$",""))
    return number

def calculate_new_variables(position, WAP, UPL, RPL, blotter):
    cash_position = blotter.loc["Cash", "Position"]
    total_cash = as_float(cash_position)
    stocks_purchased = blotter['Position'].head(2).sum()
    if WAP == []:
        Cash = " "
    else:
        Cash = position * WAP
    if UPL == []:
        PL = " "
    else:
        PL = UPL + RPL
    if stocks_purchased == 0:
        Shares = 0
    else:
        Shares =  round(position/stocks_purchased, 2)
    if cash_position == 0 or Cash == 0:
        Percent_Dollars = 0
    else:
        Percent_Dollars = Cash/total_cash
    Cash = as_currency(Cash)
    PL = as_currency(PL)
    Shares = str(Shares) + "%"
    Percent_Dollars = str(round(Percent_Dollars, 2)) + "%"
    return {"Cash": Cash, "PL": PL, "Shares": Shares, "Percent_Dollars": Percent_Dollars}

def update_dataframe (cryptos, blotter):
    cols = ['Ticker','Position', 'Market', 'WAP', 'UPL', 'RPL', 'Cash', "P/L", "% Shares", "% Dollars", "RNN", "LSTM"]
    crypto_data = pd.DataFrame(columns = cols)

    #update crypto data
    for crypto in cryptos.keys():
        Ticker = crypto
        Position = blotter.loc[crypto, "Position"]
        Market = get_price(crypto)
        WAP = as_float(clean_data(blotter, crypto, "WAP"))
        # update UPL
        current_UPL = clean_data(blotter, crypto, "UPL")
        current_market = clean_data(blotter, crypto, "Market")
        if Market != current_market:
            UPL = as_float((Market - WAP) * Position)
        else:
            UPL = as_float(current_UPL)
        # obtain RPL
        RPL = as_float(blotter.loc[crypto, "RPL"])
        # new variables added
        variables = calculate_new_variables(Position, WAP, UPL, RPL, blotter)
        RNN = main_RNN(crypto)
        LSTM = run_LSTM(crypto)
        temp_blotter = pd.Series([Ticker, Position, as_currency(Market), as_currency(WAP),  as_currency(UPL), as_currency(RPL), variables['Cash'], variables['PL'], variables['Shares'], variables['Percent_Dollars'], RNN, LSTM])
        df_blotter = pd.DataFrame([list(temp_blotter)], columns = cols)
        crypto_data = crypto_data.append(df_blotter)

    #update cash data
    cash_position = blotter.loc["Cash", "Position"]
    cash_market = blotter.loc["Cash", "Market"]
    cash = pd.Series(["Cash", cash_position, cash_market, " ",  " ", " ", " ", " ", " ", " ", " ", " "])
    cash_df = pd.DataFrame([list(cash)], columns = cols)
    crypto_data = crypto_data.append(cash_df)   
    crypto_data = crypto_data.set_index('Ticker')
    #update blotter
    blotter.update(crypto_data)
    print(blotter)
    return (blotter)

def clean_data (blotter, index, column):
    value = str(blotter.loc[index, column])
    if value != 0:
        value = value.replace(",","")
        value = float(value.replace("$",""))
    return(value)

def as_currency(amount):
    if type(amount) == str:
        return '${:,.2f}'.format(amount)
    elif amount >= -10000000:
        return '${:,.2f}'.format(amount)

def transactions(transaction_question, cryptos, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list):
    transaction = input(transaction_question).upper()
    if transaction == "A":
        blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = get_transactions(cryptos, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
    elif transaction == "B":
        blotter = update_dataframe(cryptos, blotter)
    elif transaction == "C":
        print("Goodbye")
    else:
        print("Error choose an appropriate action")
    return {"transaction": transaction, "blotter": blotter}

def get_future_data(crypto_data):
    crypto_data = crypto_data.reset_index()
    columns = ['timestamp', 'average']
    crypto_data_avg = crypto_data[columns]
    price = crypto_data['average']
    last = crypto_data_avg.tail(1)['timestamp'].dt.date
    future = []

    for i in range(35):
        time = last + timedelta(days=1)
        last = time
        future.append(time)

    usage = random.sample(range(int(min(price)), int(max(price)+100)), 35)    
    future_array = np.concatenate(future, axis=0)
    d = {'timestamp': future_array, 'average': usage}
    df = pd.DataFrame(data=d)
    crypto_data_avg_random = crypto_data_avg.append(df)
    prices = crypto_data_avg_random['average']
    
    crypto_data_avg.append(crypto_data_avg_random)
    
    return future_array, prices

def daily_price_historical(symbol, comparison_symbol, all_data=True, limit=100, aggregate=1, exchange=''):
    # https://medium.com/@galea/cryptocompare-api-quick-start-guide-ca4430a484d4
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    if all_data:
        url += '&allData=true'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    df = df.set_index("timestamp")
    df['average'] = (df['high'] + df['low'])/2
    historic_data = df.tail(120)
    return df, historic_data

def historic_price_chart(df, crypto):
    short_rolling_df = df.rolling(window=20).mean()

    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)
    plt.title(crypto)
    historic_data = df.tail(100)
    short_rolling_data = short_rolling_df.tail(100)
    Price = historic_data[['close']]
    Date = historic_data.index.values
    ax.plot(Date, Price, label = "Closing Price")
    short_rolling = short_rolling_data[['close']]
    ax.plot(Date, short_rolling, label = '20 Day Simple Moving Average')

    ax.legend(loc='best')
    ax.set_ylabel('Price in $ (' + crypto + ")")
    plt.show()

def avg_price(symbol, comparison_symbol, UTCHourDiff=-24, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/dayAvg?fsym={}&tsym={}&UTCHourDiff={}'\
            .format(symbol.upper(), comparison_symbol.upper(), UTCHourDiff)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()
    regex_price = re.findall('\d+[\,\.]{1}\d{1,2}', str(data))
    if (regex_price == []):
        new_regex_price = re.findall('\d+', str(val))
        price = float(new_regex_price[0])
    else:
        price = float(regex_price[0])
    return price

def price_24_hours(symbol, comparison_symbol, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit=10000&aggregate=3&e=CCCAGG'\
            .format(symbol.upper(), comparison_symbol.upper())
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    df = df.set_index("timestamp")
    return df

def crypto_analytics(df, crypto):
    price = df['close'].values
    min_val = min(price)
    max_val = max(price)
    num_items = len(price)
    mean = sum(price) / num_items
    differences = [x - mean for x in price]
    sq_differences = [d ** 2 for d in differences]
    ssd = sum(sq_differences)
    variance = ssd / num_items
    sd = round(sqrt(variance), 2)
    avg_price_crypto = avg_price(crypto, "USD")
    d = {"cryptocurrency": [crypto], "min": [min_val], "max": [max_val], "sd": [sd], "avg price": [avg_price_crypto]}
    stats = pd.DataFrame(data = d).set_index('cryptocurrency')
    return stats

def show_plots_and_stats(crypto, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list):
    df, historic_data = daily_price_historical(crypto, "USD")
    historic_price_chart(historic_data, crypto)
    cash_position_chart(cash_position_list, time_list)
    data_24_hours = price_24_hours(crypto, "USD")
    stats = crypto_analytics(data_24_hours, crypto)
    total_p_l_chart(P_L_list, time_list)
    vwap_chart(ETH_WAP_list, BTC_WAP_list,  time_list)
    executed_price_chart(price_list,  time_list)
    return (stats)

def purchase (crypto, price, shares_for_purchase, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list):
    # generating values for the dataframe
    cols = ['Ticker','Position', 'Market', 'WAP', 'UPL', 'RPL', 'Cash', "P/L", "% Shares", "% Dollars", "RNN", "LSTM"]    
    Ticker = crypto
    current_shares = blotter.loc[crypto, "Position"]
    Position = current_shares + shares_for_purchase
    purchase_price = float(price)

    # calcauting WAP values
    current_WAP = clean_data(blotter, crypto, "WAP")
    if current_shares == 0:
        WAP = purchase_price
    else:
        WAP = (((current_WAP * current_shares)/current_shares) + ((shares_for_purchase * purchase_price)/shares_for_purchase))/2

    # determining if value of crypto has changed since getting price for UPL
    Values = get_price(crypto)
    Market = float(Values)
    UPL = (Market - WAP) * Position

    # get cash position, cash market values, and current RPL values
    current_cash = clean_data(blotter, "Cash", "Position")
    current_market = clean_data(blotter, "Cash", "Market")
    RPL = blotter.loc[crypto, "RPL"]
    Cash_Position = current_cash - (shares_for_purchase * purchase_price)
    Cash_Market = Cash_Position

    # update blotter crypto info
    # new variables added
    WAP = as_float(WAP)
    UPL = as_float(UPL)
    RPL = as_float(RPL)
    RNN = blotter.loc[crypto, "RNN"]
    LSTM = blotter.loc[crypto, "LSTM"]
    variables = calculate_new_variables(Position, WAP, UPL, RPL, blotter)
    updated_info = pd.Series([Ticker, Position, as_currency(Market), as_currency(WAP),  as_currency(UPL), as_currency(RPL), variables['Cash'], variables['PL'], variables['Shares'], variables['Percent_Dollars'], RNN, LSTM])
    crypto_df = pd.DataFrame([list(updated_info)], columns = cols)
    crypto_df = crypto_df.set_index('Ticker')
    # update blotter cash info
    updated_cash_info = pd.Series(["Cash", as_currency(Cash_Position), as_currency(Cash_Market), "",  "", "", " ", " ", " ", " ", " ", " "])
    cash_df = pd.DataFrame([list(updated_cash_info)], columns = cols)
    cash_df = cash_df.set_index("Ticker")
    # blotter.update()
    blotter.update(crypto_df)
    blotter.update(cash_df)
    #list update
    cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = update_blotter_info(blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, price, time_list)
    return (blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)

def sell (crypto, price, shares_for_selling, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list):
    # generating values for the dataframe
    cols = ['Ticker','Position', 'Market', 'WAP', 'UPL', 'RPL', 'Cash', "P/L", "% Shares", "% Dollars", "RNN", "LSTM"]
    Ticker = crypto
    current_shares = blotter.loc[crypto, "Position"]
    Position = current_shares - shares_for_selling
    selling_price = float(price)
    Market = clean_data(blotter, crypto, "Market")

    # calcauting WAP values
    current_WAP = clean_data(blotter, crypto, "WAP")
    if Position == 0:
        WAP = 0
    else:
        WAP = (((current_WAP * Position)/Position) + ((shares_for_selling * selling_price)/shares_for_selling))/2

    # UPL changes to cryptos currently have
    UPL = (Position * (Market - WAP))
    # RPL gets recoginized
    RPL = ((price - current_WAP) * shares_for_selling)
    # new Market becomes last transaction price
    Market = price

    # get cash position, cash market values, and current RPL values
    current_cash = clean_data(blotter, "Cash", "Position")
    Cash_Position = current_cash + RPL + (price * shares_for_selling)
    Cash_Market = Cash_Position

    #update blotter crypto info
    WAP = as_float(WAP)
    UPL = as_float(UPL)
    RPL = as_float(RPL)
    RNN = blotter.loc[crypto, "RNN"]
    LSTM = blotter.loc[crypto, "LSTM"]
    variables = calculate_new_variables(Position, WAP, UPL, RPL, blotter)
    updated_info = pd.Series([Ticker, Position, as_currency(Market), as_currency(WAP),  UPL, as_currency(RPL), variables['Cash'], variables['PL'], variables['Shares'], variables['Percent_Dollars'], RNN, LSTM])
    crypto_df = pd.DataFrame([list(updated_info)], columns = cols)
    crypto_df = crypto_df.set_index('Ticker')
    # update blotter cash info
    updated_cash_info = pd.Series(["Cash", as_currency(Cash_Position), as_currency(Cash_Market), "",  "", "", " ", " ", " ", " ", " ", " "])
    cash_df = pd.DataFrame([list(updated_cash_info)], columns = cols)
    cash_df = cash_df.set_index("Ticker")
    #blotter.update()
    blotter.update(crypto_df)
    blotter.update(cash_df)
    cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = update_blotter_info(blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, price, time_list)
    return (blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)

def get_crypto(cryptos):
    loop = 1
    print("What crypto would you like to trade? As a reminder, below are the cryptos available.")
    print(cryptos.keys())
    crypto = input("Enter crypto:").upper()
    if crypto in cryptos.keys():
        loop = 2
    while loop == 1:
        print("Error. Choose one of the above five cryptos")
        crypto = input("What crypto would you like to trade?").upper()
        if crypto in cryptos.keys():
            loop = 2
    return (crypto)

def get_transactions(cryptos, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list):
    crypto = get_crypto(cryptos)
    buy_sell = input("Would you like to buy or sell?")
    if buy_sell == "buy":
        Values = get_price(crypto)
        df = show_plots_and_stats(crypto, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
        print(df)
        price_question = "The price is " + str(Values) + ". Historical data and relevant stats are shown here. Continue? Write yes or no."
        price_ok = input(price_question).lower()
        if price_ok == "yes":
            price_list.append(float(Values))
            shares = float(input("How many shares would you like to purchase?"))
            if shares <0:
                print("Error. Cannot purchase a negative number")
            else:
                current_cash = clean_data(blotter, "Cash", "Position")
                total_purchase = float(Values) * float(shares)
                if total_purchase > current_cash:
                    print("Error. Do not have adequate funds to complete this transaction")
                else:
                    blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = purchase(crypto, float(Values), float(shares), blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
                    print("Thank you, your blotter is updating")
                    print(blotter)
    if buy_sell == "sell":
        Values = get_price(crypto)
        df = show_plots_and_stats(crypto, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
        print(df)
        price_question = "The selling price is " + str(Values) + ". Historical data and relevant stats are shown here. Continue? Write yes or no."
        price_ok = input(price_question).lower()
        if price_ok == "yes":
            price_list.append(float(Values))
            current_cryptos = clean_data(blotter, crypto, "Position")
            selling_question = "You currently have " + str(current_cryptos) + " shares of " + str(crypto) + ". How many shares would you like to sell?"
            shares = float(input(selling_question))
            if shares <0:
                print("Error. Cannot purchase a negative number")
            else:
                if current_cryptos < shares:
                    print("Error. Selling more cryptos than you own.")
                else:
                    blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = sell(crypto, float(Values), float(shares), blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
                    print("Thank you, your blotter is updating")
                    print(blotter)
    return(blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)

def create_blotter_lists():
    cash_position_list = []
    ETH_WAP_list = []
    BTC_WAP_list = []
    P_L_list = []  
    price_list = []
    time_list = []
    return cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list

def save_blotter_info(blotter):
    cash_position = blotter.loc['Cash', "Position"]
    ETH_WAP = blotter.loc['ETH', 'WAP']
    BTC_WAP = blotter.loc['BTC', 'WAP']
    ETH_P_L = blotter.loc['ETH','P/L']
    BTC_P_L = blotter.loc['BTC','P/L']
    P_L = as_float(ETH_P_L) + as_float(BTC_P_L)
    return cash_position, ETH_WAP, BTC_WAP, P_L

def update_blotter_info(blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, price, time_list):
    cash_position, ETH_WAP, BTC_WAP, P_L = save_blotter_info(blotter)

    cash_position_list.append(cash_position)
    ETH_WAP_list.append(ETH_WAP)
    BTC_WAP_list.append(BTC_WAP)
    P_L_list.append(P_L)
    price_list.append(price)
    time_list.append(str(datetime.datetime.now()))
    
    return cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list

def cash_position_chart(cash_position_list, time_list):
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)

    plt.title("Total Cash Position")

    cash_position = cash_position_list
    Date = time_list
    
    ax.plot(Date, cash_position)

    ax.legend(loc='best')
    plt.show()
    
def total_p_l_chart(P_L_list, time_list):
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)

    plt.title("Profit and Loss in the Blotter")

    profit_loss = P_L_list
    Date = time_list
    
    ax.plot(Date, profit_loss)

    ax.legend(loc='best')
    plt.show()
    
def vwap_chart(ETH_WAP_list, BTC_WAP_list,  time_list):
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)

    plt.title("VWAP")

    vwap_eth = ETH_WAP_list
    vwap_btc = BTC_WAP_list
    Date = time_list
    
    ax.plot(Date, vwap_eth, label = 'Ethereum')
    ax.plot(Date, vwap_btc, label = 'Bitcoin')

    ax.legend(loc='best')
    plt.show()
    
def executed_price_chart(price_list,  time_list):
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)

    plt.title("Executed Prices")

    prices = price_list
    Date = time_list
    
    ax.plot(Date, prices)

    ax.legend(loc='best')
    plt.show()

def batch_data(prices, num_periods, forecast):    
    time_series = np.array(prices)
    x_data = time_series[:(len(time_series)-(len(time_series)%num_periods))]
    x_batches = x_data.reshape(-1, num_periods, 1)
    y_data = time_series[:(len(time_series)-(len(time_series)%num_periods))+forecast]
    y_batches = x_data.reshape(-1, num_periods, 1)
    return time_series, x_data, x_batches, y_data, y_batches

def test_data(time_series, forecast, num_periods):
    test_x_setup = time_series[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = time_series[-(num_periods):].reshape(-1, num_periods, 1)
    tf.reset_default_graph()
    return testX, testY

def set_requirements(num_periods, forecast, inputs, nodes, output, learning_rate, epochs):
    return num_periods, forecast, inputs, nodes, output, learning_rate, epochs

def create_RNN(nodes, inputs, output, num_periods, learning_rate, x_batches, y_batches, testX):
    X = tf.placeholder(tf.float32, [None, num_periods, inputs])   
    y = tf.placeholder(tf.float32, [None, num_periods, output])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=nodes, activation=tf.nn.relu)   
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)  
    stacked_rnn_output = tf.reshape(rnn_output, [-1, nodes])          
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) 
    
    loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
    training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

    init = tf.global_variables_initializer()
    epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
                save_path = saver.save(sess, "/tmp/model.ckpt")

        y_pred_RNN = sess.run(outputs, feed_dict={X: testX})
    return X, y_pred_RNN

def main_RNN(crypto):
    num_periods, forecast, inputs, nodes, output, learning_rate, epochs = set_requirements(num_periods=35, forecast=40, inputs=1, nodes=500, output=1, learning_rate=.0001, epochs=1000)

    crypto_data, hist = daily_price_historical(crypto, "USD")
    
    future_array, prices = get_future_data(crypto_data)
    time_series, x_data, x_batches, y_data, y_batches = batch_data(prices, num_periods, forecast)
    testX, testY = test_data(time_series, forecast, num_periods)
    X, y_pred_RNN = create_RNN(nodes, inputs, output, num_periods, learning_rate, x_batches, y_batches, testX)
    one_month_RNN  = y_pred_RNN[0][34]

    return one_month_RNN

def get_train_data(prices):
    values = prices.values.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size
    train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
    return train, test, scaler

def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_LSTM(crypto):
    tf.reset_default_graph() 
    crypto_data, hist = daily_price_historical(crypto, "USD")
    future_array, prices = get_future_data(crypto_data)
    
    train, test, scaler = get_train_data(prices)
    trainX, trainY = create_dataset(train, 50)
    testX, testY = create_dataset(test, 50)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    history = model.fit(trainX, trainY, epochs=500, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)
    
    yhat = model.predict(testX)
    
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
    
    return yhat_inverse

def start_up ():
    cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list = create_blotter_lists()
    loop = 1
    print("Hello, here is your blotter for the day!")
    cryptos = {"ETH": "Ethereum",
              "BTC": 'Bitcoin'}
    blotter = generate_dataframe(cryptos)
    print(blotter)
    transaction_question = "What would you like to do? A) Trade B) Update Blotter C) Exit Program"
    while loop == 1:
        transaction_values = transactions(transaction_question, cryptos, blotter, cash_position_list, ETH_WAP_list, BTC_WAP_list, P_L_list, price_list, time_list)
        if transaction_values['transaction'] == "C":
            loop = 2
            
start_up()