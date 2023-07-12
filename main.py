import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from data_colleting import Finam
import pandas as pd

style.use('fivethirtyeight')

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)

current_datetime = pd.to_datetime('today').normalize()
instrument_1 = "SBER"
instrument_2 = "VTBR"

finam_collector = Finam(instrument_1, instrument_2)
pair_data = finam_collector.get_pair_data((current_datetime - pd.DateOffset(months=1)), current_datetime)
month_data1, month_data2 = pd.DataFrame(data=list(map(lambda x: str(x).split(';'), pair_data[0][1:])),
                                        columns=str(pair_data[0][0]).split(';')), \
    pd.DataFrame(data=list(map(lambda x: str(x).split(';'), pair_data[1][1:])), columns=str(pair_data[1][0]).split(';'))

diff_df = pd.DataFrame().assign(DATE=month_data1['<DATE>'], DIFF=month_data1['<CLOSE>'].astype(float) -
                                                                 month_data2['<CLOSE>'].astype(float) * 5000)

diff_df['Z-score'] = (diff_df['DIFF'] - diff_df['DIFF'].mean()) / diff_df['DIFF'].std()

spread_avg60 = diff_df['Z-score'].rolling(window=60, center=False).mean()
spread_avg3h = diff_df['Z-score'].rolling(window=180, center=False).mean().mean()
spread_std3h = diff_df['Z-score'].rolling(window=180, center=False).std().mean()


def animate(i):
    minute_diff = finam_collector.get_minute_diff()
    displayed_graph = minute_diff
    zscore_2h = (displayed_graph['DIFF'] - displayed_graph['DIFF'].mean()) / displayed_graph['DIFF'].std()
    spread_avg10 = zscore_2h.rolling(window=10, center=False).mean()
    spread_avg20 = zscore_2h.rolling(window=20, center=False).mean()
    spread_std20 = zscore_2h.rolling(window=20, center=False).std()
    zscore_2h = (spread_avg10 - spread_avg20) / spread_std20
    buy = displayed_graph['DIFF'].copy()
    sell = displayed_graph['DIFF'].copy()
    buy[zscore_2h > -1] = 0
    sell[zscore_2h < 1] = 0
    axis.clear()
    axis.plot(displayed_graph['TIME'], displayed_graph['DIFF'])
    axis.plot(displayed_graph['TIME'], buy, color='g', linestyle='None', marker='^')
    axis.plot(displayed_graph['TIME'], sell, color='r', linestyle='None', marker='v')
    axis.set_ylim(bottom=displayed_graph['DIFF'].min(), top=displayed_graph['DIFF'].max())


ani = animation.FuncAnimation(fig, animate, interval=60000)
plt.show()
print(finam_collector.get_minute_diff())
