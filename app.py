import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas_datareader as pdr
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from flask import Flask,render_template,request
import numpy as np
import base64
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


app = Flask(__name__)

# @app.route('/')
# def main():
#     company_name = test()
#     df = pdr.get_data_tiingo(company_name, api_key = '6893d5ea1530c62591c55ef4b19a16fd4420ba4e')
    

@app.route('/')
def index():
    return render_template(
        'index.html',
data=[{'name':'Microsoft (MSFT)'}, {'name':'Apple (AAPL)'}, {'name':'Alphabet Inc (GOOG)'}, {'name':'Amazon (AMZN)'}, {'name':'Meta Platforms (META)'}, {'name':'Oracle (ORCL)'}, {'name':'Infosys (INFY)'}, {'name':'Netflix (NFLX)'}, {'name':'Starbuks (SBUX)'}, {'name':'Tesla (TSLA)'}, {'name':'Adobe (ADBE)'}, {'name':' Airbnb(ABNB)'}, {'name':'Intel group (INTC)'}])

@app.route("/" , methods=['GET', 'POST'])
def test():
    img = io.BytesIO()
    
    select = request.form.get('option_select')
    company_code = str(select)
    company_code = company_code[-5:-1]
    df = pdr.get_data_tiingo(company_code, api_key = '6893d5ea1530c62591c55ef4b19a16fd4420ba4e')
    df1 = df.reset_index()['close']
    # fig, ax = plt.subplots(1, 2, figsize=(8,8))
    # plt.subplot(1,1,1)
    plt.plot(df1)
    plt.xlabel("Time (5 years)")
    plt.ylabel("Price of the stock")
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size = int(len(df1)*0.65)
    test_size = len(df1)-training_size
    train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]
    
    x_train, y_train = create_dataset(train_data,time_step=100)
    x_test, y_test = create_dataset(test_data,time_step=100)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

    model = Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64, verbose=1)
    x_input = test_data[len(test_data)-100:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    n_steps = 100
    i = 0
    while i<30:
        if(len(temp_input)>100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1
    new_days = np.arange(1,101)
    pred_days = np.arange(101,131)
    plt.plot(new_days,scaler.inverse_transform(df1[len(df1)-100:]))
    plt.plot(pred_days,scaler.inverse_transform(lst_output))
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode('utf8')
    

    return render_template('index.html',data=[{'name':'Microsoft (MSFT)'}, {'name':'Apple (AAPL)'}, {'name':'Alphabet Inc (GOOG)'}, {'name':'Amazon (AMZN)'}, {'name':'Meta Platforms (META)'}, {'name':'Oracle (ORCL)'}, {'name':'Infosys (INFY)'}, {'name':'Netflix (NFLX)'}, {'name':'Starbuks (SBUX)'}, {'name':'Tesla (TSLA)'}, {'name':'Adobe (ADBE)'}, {'name':' Airbnb(ABNB)'}, {'name':'Intel group (INTC)'}], plot_url=plot_url,plot_url1=plot_url1,name_of_Stock = str(select) )

    # return(str(select)) # just to see what select is

def create_dataset(dataset, time_step=1):
    dataX = []
    dataY = []
    for i in range(len(dataset)-time_step -1):
        a = dataset[i: i+time_step ,0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)

@app.route('/print-plot')
def plot_png():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   xs = np.random.rand(100)
   ys = np.random.rand(100)
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run()