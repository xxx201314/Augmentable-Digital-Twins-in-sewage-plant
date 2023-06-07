import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_data(test_y,predict_y,title):
    plt.figure(figsize=(10,8),dpi=100)
    x = [i for i in range(test_y.shape[0])]
    plt.plot(x, test_y, color='navy',lw = 2, label='Actual data')
    plt.plot(x, predict_y, color='c',lw = 2, label='Predict data')
    x_label = [i for i in x[::5]]
    
    y_label = [j for j in range(0,60,5)]
    plt.yticks(y_label)
    plt.xticks(x_label)
    plt.xlabel('Time/d')
    plt.ylabel('COD')
    plt.title(title)
    #plt.title('Support Vector Machine')
    plt.legend()
    plt.show()

fnn_fd = pd.read_csv(".\show\FNN_data.csv")
fnn_data = fnn_fd["y_predict_FNN"].values.tolist()

gru_fd = pd.read_csv(".\show\GRU_data.csv")
gru_data = gru_fd["predicted_cod"].values.tolist()

lstm_fd = pd.read_csv(".\show\LSTM_data.csv")
lstm_data = lstm_fd["predicted_cod"].values.tolist()

bp_fd = pd.read_csv(".\show\BP_data.csv")
bp_data = bp_fd["predicted_cod"].values.tolist()

Transformer_fd = pd.read_csv(".\show\Transformer_data.csv")
Transformer_data = Transformer_fd["y_predict_Transformer"].values.tolist()

real_data = bp_fd["real_cod"]


plt.figure(figsize=(10,8),dpi=100)
x = [i for i in range(real_data.shape[0])]
plt.plot(x, fnn_data, color='#DA70D6', lw = 2, label='FNN Predict data')
plt.plot(x, gru_data, color='#87CEEB', lw = 2, label='GRU Predict data')
plt.plot(x, lstm_data, color='#00F000', lw = 2, label='LSTM Predict data')
plt.plot(x, bp_data, color='#FFA500', lw = 2, label='BP Predict data')
plt.plot(x, Transformer_data, color='k', lw = 2, label='Transformer Predict data')
plt.plot(x, real_data, color='#FF0000', lw = 2, label='Actual data')
x_label = [i for i in x[::5]]
y_label = [j for j in range(0,60,5)]
plt.yticks(y_label)
plt.xticks(x_label)
plt.xlabel('Time/d')
plt.ylabel('COD')
plt.title('Comparison of predicted results')
#plt.title('Support Vector Machine')
plt.legend()
plt.show()


# show_data(real_data,fnn_data,"FNN")
# show_data(real_data,gru_data,"GRU")
# show_data(real_data,lstm_data,"LSTM")
# show_data(real_data,bp_data,"BP")
show_data(real_data,Transformer_data,"Transformer")