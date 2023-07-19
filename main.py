import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from data_processing import Data
from learning_functions import ML

style.use('fivethirtyeight')

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)

instrument_1 = "SBER"
instrument_2 = "VTBR"

data_engine = Data(instrument_1, instrument_2)

diff_df = data_engine.get_ml_df()
ml = ML(diff_df)
ml.learn_model()


def animate(i):
    minute_diff = data_engine.minute_extend()
    displayed_graph = minute_diff
    predictions = ml.predict(minute_diff)
    svm_pred = predictions['SVM']
    lr_pred = predictions['Logistic Regression']
    ensemble_pred = predictions['Ensemble']
    buy = displayed_graph['diff'].copy()
    sell = displayed_graph['diff'].copy()
    buy[svm_pred != 1] = 0
    sell[svm_pred != -1] = 0
    axis.clear()
    axis.plot(displayed_graph['DATETIME'], displayed_graph['diff'])
    axis.plot(displayed_graph['DATETIME'], buy, color='g', linestyle='None', marker='^')
    axis.plot(displayed_graph['DATETIME'], sell, color='r', linestyle='None', marker='v')
    axis.set_ylim(bottom=displayed_graph['diff'].min(), top=displayed_graph['diff'].max())


ani = animation.FuncAnimation(fig, animate, interval=60000)
plt.show()
