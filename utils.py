import string
import numpy as np
import pandas as pd

# sklearn imports to create custom transformers
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NLTK imports
#from nltk.corpus import stopwords
from nltk.util import ngrams

# SpaCy imports
from spacy.lang.en import English
import en_core_web_sm

# Variable definitions
nlp = en_core_web_sm.load()
parser = English()
#STOPLIST = set(stopwords.words('english'))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]


class FeaturesEngineering(BaseEstimator, TransformerMixin):
    
    # Implement the __init__ method.
    def __init__(self,
                 keep_orig=False,
                 text_stats=False,
                 binning=False,
                 nlp=False,
                 process_feats=False,
                 verbose=None):
        self.keep_orig = keep_orig
        self.text_stats = text_stats
        self.binning = binning
        self.nlp = nlp
        self.process_feats = process_feats
        self.verbose = verbose

    # All SciKit-Learn compatible transformers and classifiers have the
    # same interface. `fit` always returns the same object.
    def fit(self, X, *_):
        # Fit the transformer and store it.
        return self


    def transform(self, X, y=None):
        X = X.copy()
        
        if self.verbose is not None:
            print('Creating new features...')

        list_of_series = []
        cols_names=X.columns.tolist()
        
        X = X.replace(to_replace=['N/A or Unknown', 'unknown', 'nan', 'none', None], value=np.nan)
        
        if self.text_stats is True:
            df1 = pd.concat([self._text_stats(X.person_attributes),
                            self._text_stats(X.seat),
                            self._text_stats(X.other_person_location),
                            self._text_stats(X.other_factor_1),
                            self._text_stats(X.other_factor_2),
                            self._text_stats(X.other_factor_2)], axis=1)
            list_of_series.append(df1)
            cols_names += df1.columns.tolist()
        
        if self.binning is True:
            s2 = self._age_binning(X.age_in_years)
            X = X.drop(columns='age_in_years')
            list_of_series.append(s2)
            cols_names += s2.name
        
        if self.nlp is True:
            df5 = self._nlp(X)
            list_of_series.append(df5)
            if isinstance(df5, pd.core.series.Series):
                cols_names += df5.name
            else:
                cols_names += df5.columns.tolist()
        
        if self.process_feats is True:
            df2 = self._process_attributes()
            df3 = self._process_seat()
            df4 = self._process_other_person()
            list_of_series.append(df2)
            list_of_series.append(df3)
            list_of_series.append(df4)
            cols_names += df2.columns.tolist()
            cols_names += df3.columns.tolist()
            cols_names += df4.columns.tolist()
        
        if self.keep_orig is True:
            list_of_series.insert(0, X)
            
        df = pd.concat(list_of_series, axis=1, ignore_index=False)

        return df

    def _generic(self, df_):
        df = df_.copy()
        
        df['n_factors'] = -df.isnull()[['other_factor_1', 'other_factor_2', 'other_factor_3']].sum(axis=1) + 3

        return df
        
    def _nlp(self, df_):
        df = df_.copy()
        df_out = pd.DataFrame()
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        ### Creating column with all available text (cleaned)
        s = pd.Series('', index=df.index)
        
        for col in text_columns:
            s += (' ' + df[col].fillna('').astype(str))
        
        df_out['text'] = s.apply(self._cleanText)
        
        ### Creating column with number of occurrences of the text
        occurrences = s.value_counts()
        df_out['occurrences'] = s.apply(lambda x: occurrences[x])
        
        ###
        #df_out['ratio'] = self._10ratio(df_out)
        
        return df_out

    def _text_stats(self, series_):
            series = series_.copy()

            chars_count = series.apply(self._count_chars)
            words_count = series.apply(self._count_words)
            ratio = (chars_count / words_count).fillna(0)

            return pd.DataFrame({series.name + '_char_count': chars_count,
                                 series.name + '_word_count': words_count,
                                 series.name + '_char_word_ratio':ratio})
        
    def _process_attributes(self):
        series = X.person_attributes
        series = series.str.lower().fillna('unknown')

        feature_list = ['vehicle', 'on_bike', 'on_foot', 'driving', 'passenger', 'stopped']
        df_attr = pd.DataFrame(0, index=np.arange(len(series)), columns=feature_list)

        for col in feature_list:
            df_attr[col] = series.str.contains(col, regex=False).astype(int)

        df_attr.vehicle = df_attr.vehicle + df_attr.driving + df_attr.passenger
    
        return df_attr

    def _process_seat(self):
        series = X.seat#series_.copy()
        series = (series.str.lower()
                        .fillna('unknown')
                        .str.split('_')
                        #.apply(cleanText)
                 )

        pos1 = ['not', 'front', 'second', 'third', 'fourth', 'fifth', 'back',
                'cargo', 'trailer', 'outside', 'truck']
        pos2 = ['left', 'middle', 'right', 'open', 'sleeper']
        
        df_seat = pd.DataFrame('unknown', index=np.arange(len(series)), columns=['seat_pos1', 'seat_pos2'])
        df_seat.seat_pos1 = series.apply(self._crosscheck, args=(pos1,)).fillna('unknown').replace('not', 'not_in_vehicle')
        df_seat.seat_pos2 = series.apply(self._crosscheck, args=(pos2,)).fillna('unknown')
    
        return df_seat
    
    def _process_other_person(self):
        series = X.other_person_location#series_.copy()

        series = series.str.lower().replace(['no_', 'non_', 'not_'], 'NO').fillna('unknown')

        feature_list = ['intersection', 'unmarked', 'legal', 'NOintersection', 'xwalk', 'NOxwalk', 'middle_of_road', 'NOroad', 'NOhighway']
        df_op = pd.DataFrame(0, index=np.arange(len(series)), columns=feature_list)

        for col in feature_list:
            df_op[col] = series.str.contains(col, regex=False).astype(int)

        df_op.intersection = df_op.intersection - df_op.NOintersection
        df_op.xwalk = df_op.xwalk - df_op.NOxwalk
        #df_op.not_road = df_op.NOroad + df_op.NOhighway
        #df_op = df_op.drop(columns = ['NOintersection', 'NOxwalk', 'NOhighway'])

        return df_op
    
    def _age_binning(self, series_):
        series = series_.copy()
        bins = [0, 15, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 150]
        labels = ['<16', '16-19', '20-24', '25-29', '30-34', '35-39', '40-44','45-49',
                  '50-54', '55-59','60-64', '65-69','70-74', '>74']
        return pd.cut(series, bins, labels=labels).astype('object')
    
    def _cleanText(self, text):
        if not isinstance(text, str):
            pass
        else:
            text = (text.strip()
                        .replace("\n", " ")
                        .replace("\r", " ")
                        .replace("_", " ")
                        .replace("-", " ")
                        .lower())
        return text
    
    def _count_words(self, text):
        if not isinstance(text, str):
            return 0
        else:
            return len(self._cleanText(text).split())
        
    def _count_chars(self, text):
        if not isinstance(text, str):
            return 0
        else:
            return len(self._cleanText(text))
    
    def _crosscheck(self, list1, list2):
        for item in list1:
            if item in list2:
                return item
    
    def _uniques(self, df_, text_label, target_label, target_value):
        df_ = df_.copy()
        l =[]
        for val in df_[text_label].unique():
            a = df_[(df_[text_label] == val) & (df_[target_label] == true_class)]
            l.append(a.shape[0])

        return pd.Series(l, index=df_[text_label].unique())
    
    def _10ratio(self, df_):    
        df_ = df_.copy()

        L1 = self._uniques(df_, 'text', 'target', 1)
        L0 = self._uniques(df_, 'text', 'target', 0)

        return (df_.apply(lambda x: L0[x[0]], axis=1) / df.apply(lambda x: L1[x[0]], axis=1)).replace(0, 1)

class NanImputer(BaseEstimator, TransformerMixin):
        def __init__(self, columns=None, dtype='number', categorical_feat=None, strategy='median', verbose=None):
            self.columns = columns
            self.dtype = dtype
            self.categorical_feat = categorical_feat
            self.strategy = strategy
            self.verbose = verbose

        def fit(self, X, y=None):
            if self.columns is None:
                self.columns = X.select_dtypes(include=[self.dtype]).columns
                
            if self.dtype == 'object':
                self.fills = 'unknown'
            else:
                if self.categorical_feat is not None:
                    self.fills = getattr(X.groupby(self.categorical_feat), self.strategy)()
                else:
                    self.fills = getattr(X, self.strategy)()
            return self

        def transform(self, X):
            if self.verbose is not None:
                print('Filling in numerical missing values...')
            X = X.copy()
            if self.categorical_feat is not None:
                for cat in X[self.categorical_feat].unique():
                    row_indexer = X[self.categorical_feat] == cat
                    temp = X.loc[row_indexer, self.columns]
                    X.loc[row_indexer, self.columns] = temp.fillna(self.fills.loc[cat, self.columns])
            else:
                X = X.fillna(self.fills)
            
            return X

class Scaler(BaseEstimator, TransformerMixin):
        # Implement the __init__ method.
        # Our ColumnSelector has a parameter columns.
        # The default value for columns is 'all'
        def __init__(self, method='minmax', columns=None):
            self.method = method.lower()
            if self.method == 'minmax':
                self.scaler = MinMaxScaler()
            if self.method == 'standard':
                self.scaler = StandardScaler()
            self.columns = columns
            
        # All SciKit-Learn compatible transformers and classifiers have the
        # same interface. `fit` always returns the same object.
        def fit(self, X, y=None):
            # Fit the transformer and store it.
            if self.scaler is None:
                return self
            else:
                if self.columns is None:
                    self.columns = X.select_dtypes(exclude=['object']).columns
                self.scaler.fit(X[self.columns])
                
                return self

        # Transform  return all columns is equal to 'all'.
        # If a column or a list of columns are passed, only those should be returned.
        def transform(self, X, *_):
            if self.scaler is None:
                return X
            else:
                return pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    # Implement the __init__ method.
    # Our ColumnSelector has a parameter columns.
    # The default value for columns is 'all'
    def __init__(self, columns=None, verbose=None):
        if columns:
            self.columns = columns
        else:
            self.columns = 'all'
        self.verbose = verbose
        
    # All SciKit-Learn compatible transformers and classifiers have the
    # same interface. `fit` always returns the same object.
    def fit(self, *_):
        # Fit the transformer and store it.
        return self

    # Transform  return all columns is equal to 'all'.
    # If a column or a list of columns are passed, only those should be returned.
    def transform(self, X, *_):
        X = X.copy()
        if self.columns == 'all':
            if self.verbose is not None:
                print('\nReturning original DataFrame.\n')
            return X
        else:
            if self.verbose is not None:
                print('\nReturning only selected columns.\n')
            return X[self.columns]

class ColumnDropper(BaseEstimator, TransformerMixin):
    # Implement the __init__ method.
    # Our ColumnSelector has a parameter columns.
    # The default value for columns is 'all'
    def __init__(self, columns=None, verbose=None):
        self.columns = columns
        self.verbose = verbose

        
    # All SciKit-Learn compatible transformers and classifiers have the
    # same interface. `fit` always returns the same object.
    def fit(self, *_):
        # Fit the transformer and store it.
        return self

    # Transform  return all columns is equal to 'all'.
    # If a column or a list of columns are passed, only those should be returned.
    def transform(self, X, *_):
        df = pd.DataFrame(X, columns=X.columns)
        if self.columns is None:
            if self.verbose is not None:
                print('No column dropped. Returning original DataFrame.\n')
            return df
        else:
            if self.verbose is not None:
                print('Selected columns dropped. Returning smaller DataFrame.\n')
            return df.drop(columns=self.columns)

def tokenizeText(sample):#, stopwords=True):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    #if stopwords:
    #    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    
    return tokens