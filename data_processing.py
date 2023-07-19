from data_colleting import Finam
import pandas as pd
import numpy as np


class EmptyDataError(Exception):
    """Returns exception when data_prepare called, but data fields are empty"""

    def __init__(self):
        super().__init__("No data object")


class Data:
    def __init__(self, instrument1, instrument2):
        self.instr1 = instrument1
        self.instr2 = instrument2
        self.finam_collector = Finam(self.instr1, self.instr2)
        self.data1, self.data2 = (None, None)
        self.avg_data = None
        self.minute_date = None

    def _collect_data(self, datetime_offset=pd.DateOffset(months=1)):
        current_datetime = pd.to_datetime('today').normalize()
        if datetime_offset:
            pair_data = self.finam_collector.get_pair_data((current_datetime - datetime_offset), current_datetime)
        else:
            pair_data = self.finam_collector.get_pair_data(current_datetime, current_datetime)
        self.data1, self.data2 = pd.DataFrame(
            data=list(map(lambda x: x.decode('utf-8').rstrip('\r\n').split(';'), pair_data[0][1:])),
            columns=pair_data[0][0].decode('utf-8').rstrip('\r\n').split(';')), \
            pd.DataFrame(data=list(map(lambda x: x.decode('utf-8').rstrip('\r\n').split(';'), pair_data[1][1:])),
                         columns=pair_data[1][0].decode('utf-8').rstrip('\r\n').split(';'))

    def _data_prepare(self, type='month'):
        if self.data1 is None or self.data2 is None:
            assert EmptyDataError
        self.data1.rename(columns={"<DATE>": "DATE",
                                   "<TIME>": "TIME",
                                   "<OPEN>": "OPEN",
                                   "<HIGH>": "HIGH",
                                   "<LOW>": "LOW",
                                   "<CLOSE>": "CLOSE",
                                   "<VOL>": "VOL"}, inplace=True)
        self.data2.rename(columns={"<DATE>": "DATE",
                                   "<TIME>": "TIME",
                                   "<OPEN>": "OPEN",
                                   "<HIGH>": "HIGH",
                                   "<LOW>": "LOW",
                                   "<CLOSE>": "CLOSE",
                                   "<VOL>": "VOL"}, inplace=True)

        self.data1, self.data2 = self.data1.drop(['<TICKER>', '<PER>'], axis=1), \
            self.data2.drop(['<TICKER>', '<PER>'], axis=1)

        self.data1['DATETIME'], self.data2['DATETIME'] = pd.to_datetime(
            self.data1["DATE"] + self.data1['TIME']), pd.to_datetime(
            self.data2['DATE'] + self.data2['TIME'])

        self.data1 = self.data1.drop(['DATE', 'TIME'], axis=1)
        self.data2 = self.data2.drop(['DATE', 'TIME'], axis=1)

        self.data1.iloc[:, :-1] = self.data1.iloc[:, :-1].astype(float)
        self.data2.iloc[:, :-1] = self.data2.iloc[:, :-1].astype(float)

        self.data1['AVG'] = self.data1[['OPEN', 'CLOSE']].mean(axis=1)
        self.data2['AVG'] = self.data2[['OPEN', 'CLOSE']].mean(axis=1)

        data1_avg = self.data1[['AVG', 'VOL', 'DATETIME']]
        data2_avg = self.data2[['AVG', 'VOL', 'DATETIME']]

        if type == 'month':
            self.avg_data = data1_avg.merge(data2_avg, on='DATETIME', suffixes=['_1', '_2'])
            self.avg_data['diff'] = self.avg_data['AVG_1'] - self.avg_data['AVG_2']

            self.avg_data['z-score'] = (self.avg_data['diff'] - self.avg_data['diff'].mean()) / self.avg_data[
                'diff'].std()

            spread_avg60 = self.avg_data['z-score'].rolling(window=60, center=False).mean()
            spread_avg200 = self.avg_data['z-score'].rolling(window=200, center=False).mean()
            spread_std200 = self.avg_data['z-score'].rolling(window=200, center=False).std()
            zscore_60_120 = (spread_avg60 - spread_avg200) / spread_std200

            conditions = [(zscore_60_120 < 1.20) & (zscore_60_120 > -1.20),
                          (zscore_60_120 >= 1.20),
                          (zscore_60_120 <= -1.20)]
            values = [0, 1, -1]

            self.avg_data['indicator'] = np.select(conditions, values)
        else:
            self.minute_date = data1_avg.merge(data2_avg, on='DATETIME', suffixes=['_1', '_2'])
            self.minute_date['diff'] = self.minute_date['AVG_1'] - self.minute_date['AVG_2']

    def get_ml_df(self):
        self._collect_data()
        self._data_prepare()
        return self.avg_data

    def minute_extend(self):
        self._collect_data(datetime_offset=0)
        self._data_prepare('minute')
        return self.minute_date
