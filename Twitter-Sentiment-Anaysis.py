#!/usr/bin/env python
# coding: utf-8


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import datetime
import json
import time,re
import pandas as pd
import numpy as np
from google.cloud import pubsub_v1
import unicodedata
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
## python -m textblob.download_corpora
from textblob import TextBlob
import string
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


PROJECT='evident-zone-334821'
BUCKET='evident-zone-334821'
ROOT='sentiment_analysis'
SERVICE_ACCOUNT='twitterstream@evident-zone-334821.iam.gserviceaccount.com'
MODEL_DIR="sentiment_analysis/models"
PACKAGES_DIR="sentiment_analysis/packages"


#config
get_ipython().system('gcloud config set project {PROJECT}')
#df.logging.set_verbosity(df.logging.INFO)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:\\Swansea\\teaching\\project\\code\\config\\evident-zone-334821-208a175a9393.json"


# # Data Preprocessing

# pre-load sentiments

sentiment_mapping={
    0:"negative",
    2:"neutral",
    4:"positive"
}

df_twitter = pd.read_csv("./training.csv",encoding="latin1", header=None)             .rename(columns={
                 0:"sentiment",
                 1:"id",
                 2:"time",
                 3:"query",
                 4:"username",
                 5:"text"
             })[["sentiment","text"]]


df_twitter["sentiment_label"] = df_twitter["sentiment"].map(sentiment_mapping)


df_twitter.head()


get_ipython().run_cell_magic('writefile', 'preprocess.py', '\nfrom tensorflow.python.keras.preprocessing import sequence\nfrom tensorflow.keras.preprocessing import text\nimport tensorflow as tf\nimport re\n\nclass TextPreprocessor(object):\n    def __init__(self, vocab_size, max_sequence_length):\n        self._vocab_size = vocab_size\n        self._max_sequence_length = max_sequence_length\n        self._tokenizer=None\n    \n    def fit(self, text_list):        \n        # Create vocabulary from input corpus.\n        #text_list_cleaned = [self._clean_line(txt) for txt in text_list]\n        tokenizer = text.Tokenizer(num_words=self._vocab_size)\n        tokenizer.fit_on_texts(text_list)\n        self._tokenizer = tokenizer\n\n    def transform(self, text_list):        \n        # Transform text to sequence of integers\n        #text_list = [self._clean_line(txt) for txt in text_list]\n        text_sequence = self._tokenizer.texts_to_sequences(text_list)\n\n        # Fix sequence length to max value. Sequences shorter than the length are\n        # padded in the beginning and sequences longer are truncated\n        # at the beginning.\n        padded_text_sequence = sequence.pad_sequences(\n          text_sequence, maxlen=self._max_sequence_length)\n        return padded_text_sequence')


requests = (["God I hate the north","god I love this"])


from preprocess import TextPreprocessor

processor = TextPreprocessor(5, 5)
processor.fit(['hello machine learning','test'])
processor.transform(['hello machine learning',"lol"])


CLASSES = {'negative':0, 'positive': 1}  # label-to-int mapping
VOCAB_SIZE = 25000  # Limit on the number vocabulary size used for tokenization
MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length

#from preprocess import TextPreprocessor
from sklearn.model_selection import train_test_split


sents = df_twitter.text
labels = np.array(df_twitter.sentiment_label.map(CLASSES))

# Train and test split
X, _, y, _ = train_test_split(sents, labels, test_size=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create vocabulary from training corpus.
processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
processor.fit(X_train)

# Preprocess the data
train_texts_vectorized = processor.transform(X_train)
eval_texts_vectorized = processor.transform(X_test)

import pickle

with open('./processor_state.pkl', 'wb') as f:
    pickle.dump(processor, f)


# # Model Preparation


LEARNING_RATE=.001
EMBEDDING_DIM=50
FILTERS=64
DROPOUT_RATE=0.5
POOL_SIZE=3
NUM_EPOCH=15
BATCH_SIZE=128
KERNEL_SIZES=[2,5,8]


def create_model(vocab_size, embedding_dim, filters, kernel_sizes, dropout_rate, pool_size, embedding_matrix):
    
    # Input layer
    model_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Embedding layer
    z = tf.keras.layers.Embedding(
        input_dim=vocab_size+1,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix]
    )(model_input)

    z = tf.keras.layers.Dropout(dropout_rate)(z)

    # Convolutional block
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = tf.keras.layers.Convolution1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation="relu",
            bias_initializer='random_uniform',
            strides=1)(z)
        conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
        conv = tf.keras.layers.Flatten()(conv)
        conv_blocks.append(conv)
        
    z = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(100, activation="relu")(z)
    model_output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(model_input, model_output)
    
    return model


# ### load the embeddings

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open("./glove.twitter.27B.50d.txt","r",encoding="utf8"))


word_index = processor._tokenizer.word_index
nb_words = min(VOCAB_SIZE, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= VOCAB_SIZE: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


model = create_model(VOCAB_SIZE, EMBEDDING_DIM, FILTERS, KERNEL_SIZES, DROPOUT_RATE,POOL_SIZE, embedding_matrix)


# Compile model with learning parameters.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])


physical_devies=tf.config.list_physical_devices('CPU')
print(physical_devies)
#with tf.device('/gpu:0'):
tf.config.experimental.set_memory_growth(physical_devies[0],True)


# ## Model Training


history = model.fit(
    train_texts_vectorized, 
    y_train, 
    epochs=NUM_EPOCH, 
    batch_size=BATCH_SIZE,
    validation_data=(eval_texts_vectorized, y_test),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc',
            min_delta=0.005,
            patience=3,
            factor=0.5),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.005, 
            patience=3, 
            verbose=0, 
            mode='auto'
        ),
        tf.keras.callbacks.History()
    ]
)


history.history


from matplotlib import pyplot as plt
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('acc')
plt.ylabel('val_acc')
plt.legend(['Training Accuracy',
           'Validation Accuracy'])
plt.title('Accuracy Curves')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Training Loss',
           'Validation Loss'])
plt.title('Loss Curves')
plt.show()


#results_cnn = model_cnn.evaluate(X_test, y_test)
#print(f'Test set loss: {results_cnn[0]:0.2f}, test set accuracy: {results_cnn[1]*100:0.2f}%')


with open("history.pkl",'wb') as file:
    pickle.dump(history.history,file)


model.save('./model')


get_ipython().run_cell_magic('writefile', 'model_prediction.py', '\nimport os\nimport pickle\nimport numpy as np\nimport tensorflow as tf\n\n\nclass CustomModelPrediction(object):\n\n    def __init__(self, model, processor):\n        self._model = model\n        self._processor = processor\n\n    def _postprocess(self, predictions):\n        labels = [ \'positive\',\'negative\']\n        return [\n            {\n                "label":labels[int(np.round(prediction))],\n                "score":float(np.round(prediction,4))\n            } for prediction in predictions]\n\n\n    def predict(self, instances, **kwargs):\n        preprocessed_data = self._processor.transform(instances)\n        predictions =  self._model.predict(tf.convert_to_tensor(preprocessed_data, dtype=tf.float32))\n        labels = self._postprocess(predictions)\n        return labels\n\n\n    @classmethod\n    def from_path(cls, model_dir):\n        import tensorflow.keras as keras\n        model = keras.models.load_model(\n          os.path.join(model_dir,\'keras_saved_model.h5\'))\n        with open(os.path.join(model_dir, \'processor_state.pkl\'), \'rb\') as f:\n            processor = pickle.load(f)\n    \n        return cls(model, processor)')


from model_prediction import CustomModelPrediction
import time
classifier = CustomModelPrediction.from_path('.')
t0 = time.time()
requests = (["okay love crypto hate maxis"])
results = classifier.predict(requests)
t1 = time.time()
print(results)
print(t1-t0)



get_ipython().run_cell_magic('writefile', 'setup.py', '\nfrom setuptools import setup\n\nsetup(\n  name="tweet_sentiment_classifier",\n  version="0.1",\n  include_package_data=True,\n  scripts=["preprocess.py", "model_prediction.py"]\n)')



get_ipython().system('python setup.py sdist')
get_ipython().system('gsutil cp ./dist/tweet_sentiment_classifier-0.1.tar.gz "gs://{PACKAGES_DIR}/tweet_sentiment_classifier-0.1.tar.gz"')


# # Database connection


from sqlalchemy import create_engine
from datetime import datetime
class connect_db():
    
    def __init__(self):
        self.conn=self.connect()
        print('connection estb')
    
    def connect(self):
        # read the config from the config file and create the connection string
        df=pd.read_csv('db_config.csv')
        config = {
          'user': df['user'][0],
          'password': df['password'][0],
          'host': df['host'][0],
          'database': df['database'][0],
          'raise_on_warnings': df['raise_on_warnings'][0] 
        }
        # create the connection URL
        url='mysql+pymysql://'+df['user'][0]+':'+df['password'][0]+'@'+df['host'][0]+'/'+df['database'][0]+'?charset=utf8mb4'
        # get an instance of SQL Alchemy engine for the DB connection
        engine = create_engine(url)
        autocommit_engine = engine.execution_options(autocommit=True,isolation_level="AUTOCOMMIT")
        print('connected')
        return engine
        
            
    def write_to_db(self,records):
        try:
            with self.conn.connect() as connection:
                #df=pd.DataFrame([[record.values()]],columns=[record.keys()])
                records.drop('clean_text', axis=1, inplace=True)
                records.to_sql(name='tweet_sentiments', con=connection, if_exists='append', index=False)
                connection.close()
        except Exception as e:
            print(e)
            connection.close()
            self.conn.dispose()


class TwitterAuthenticator():
    """
    This class will authenicate using the keys and return an Authenticated variable for the TwitterAPI
    """ 
    def __init__(self):
        from model_prediction import CustomModelPrediction
        df=pd.read_csv('twitter_config.csv')
        self.consumer_key=df['consumer_key'][0]
        self.consumer_secret=df['consumer_secret'][0]
        self.access_token=df['access_token'][0]
        self.access_token_secret=df['access_token_secret'][0]
    
    def Twitter_Authenticate(self):
        auth=OAuthHandler(self.consumer_key,self.consumer_secret)
        auth.set_access_token(self.access_token,self.access_token_secret)
        return auth;


class TextCleanup():
    """
    This class contains methods used to clean up the data for the sentiment analysis
    """
    def removeEmoticons(self, txt):
        
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00002500-\U00002BEF"
        u"\U00010000-\U0010ffff"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\U0001f98d"
        u"\U0001f914" 
        u"\U0001f9f1"
        u"\U0001f911"
        u"\U0001f923"
        u"\U0001f9d9"
        u"\u200d"
        u"\u23f1"
        u"\u23f3"
        u"\U0001f91f"
        u"\U0001f7e5"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
        
        txt=re.sub(r'@[A-Za-z0-9_]+','',txt)
        txt=re.sub(r'RT : ','',txt)
        txt = txt.lower()
        txt = txt.strip()
        #txt=re.sub(r'[\w\s]','',txt)
        txt=re.sub(r'\n','',txt)
        txt=re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
        re.sub('\s+', ' ',txt)
        txt=emoji_pattern.sub(r'', txt) # no emoji
        return txt.translate(str.maketrans('','',string.punctuation))
    
    def removeAccentChar(self,data):
        return unicodedata.normalize('NFKD',data).encode('ascii','ignore').decode('utf-8','ignore')
    
    def removeStopWords(self,text):
        tweet=""
        for word in text.split(' '):
            word=word.lower()
            if word not in stop_words and word not in other_stop_words:
                tweet=tweet+str(' ')+word
        return tweet
    
    def correctSpell(self,text):
        text=TextBlob(text)
        return text.correct()
    
    def lemmatizeText(self,text):
        tweet=""
        for word in text.split(' '):
            word = WordNetLemmatizer().lemmatize(word)
            tweet=tweet+str(' ')+word
        return tweet



import traceback
import sys
import tensorflow as tf

# Listener class
class TweetListener(StreamListener):
    """
    This is the streaming class that will stream the twitter data
    """
    def on_status(self, data):
        global tweet_lst
        global text_lst
        
        # When receiveing a tweet: extract the json from the dictionary
        tweet=data._json
        
        # we are only intertested in the English tweets and only the text part of the tweet
        if  tweet['lang']=="en" and not tweet['text'].startswith('RT') :
            #if not any(item in tweet['text'] for item in filter_list):
            #    return True
            try:
                # clean up the tweet
                crypto=''
                new_tweet=text_cleanup.removeEmoticons(tweet['text'])
                #new_tweet=text_cleanup.removeAccentChar(new_tweet)
                clean_text=text_cleanup.removeStopWords(new_tweet)
                
                # if the text is too short skip the tweet
                if len(clean_text)<15:
                    return True;
                
                tweet_txt=''
                # get the crypto curreny related to the tweet
                if 'extended_tweet' in tweet:
                    for item in filter_list:
                        if item in tweet['extended_tweet']['full_text'].lower():
                            crypto=item
                            tweet_txt=tweet['extended_tweet']['full_text'].lower()
                else:
                    for item in filter_list:
                        if item in tweet['text'].lower():
                            crypto=item
                            tweet_txt=tweet['text'].lower()
                            
                
                tweet_lst.append({
                    "tweet": tweet['text'],
                    "user_id": tweet['user']['id'],
                    "tweet_id": tweet["id"],
                    "crypto":crypto,
                    #"created_at": datetime.strptime(tweet["created_at"],'%a %b %d %H:%M:%S +0000 %Y').strftime('%Y-%m-%d %H:%M:%S'),
                    "created_at":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "clean_text":clean_text.replace('...','').strip()
                })
                #text_lst.append(clean_text.replace('...','').strip())
                
                print('[Tweet ',len(tweet_lst),'] : ','[',crypto,'] ',clean_text)
                
                # when we have a certain number of tweets, flush them to the database together
                if len(tweet_lst)>=20:
                    df=pd.DataFrame(tweet_lst)
                    
                    #get the sentiments
                    #sentiments=pd.DataFrame(self.getSentiments(text_lst))
                    sentiments=pd.DataFrame(self.getSentiments(df['clean_text'].values.tolist()))
                    #sentiment=return_sentiments)
                    df['polarity']=sentiments['score']
                    df['sentiment']=sentiments['label']
    
                    #tweet_lst_copy=tweet_lst
                    tweet_lst=[]
                    #store the sentiments and data to db
                    stream2db=connect_db()
                    stream2db.write_to_db(df)
                
                #stream2db.write_to_db(tweet,clean_text,'crypto',sentiment[0]['label'],round(sentiment[0]['score'],4))
            
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                return True;
            return True

    def on_error(self, status):
        # stop the streaming when the streaming limit is reached
        if status == 420:
            print("rate limit active")
            return False
        
    def getSentiments(self,tweet):
        # get the sentiment from the trained model
        from model_prediction import CustomModelPrediction
        classifier = CustomModelPrediction.from_path('.')
        return classifier.predict(tweet)
    
    
    
class TweetExtratcor():
    """
    This class is used to make a connection and retrieve the data stream from twitter's API(tweepy)
    """
    def extractTweet(self,filterList):
    
        # obtain an authenticated auth variable
        twitterAuthenicator=TwitterAuthenticator()
        auth=twitterAuthenicator.Twitter_Authenticate()
        
        # Configure to wait on rate limit if necessary
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=False)
        
        # Make an instance of the class
        listener = TweetListener()
        
        
        # Start streaming
        stream = tweepy.Stream(auth, listener)
        stream.filter(track=filterList, languages=['en'])

        

        
if __name__=="__main__":        
    filter_list=['dodge coin','shibu','shiba','xrp','ethereum','bitcoin','btc','lite coin','litecoin']
    tweet_lst=[]
    text_lst=[]
    # filter  out retweets
    #load the model
    from model_prediction import CustomModelPrediction
    text_cleanup=TextCleanup()
    classifier = CustomModelPrediction.from_path('.')
    other_stop_words=['i','whats','join','group','can','this','that','what','for','will','here','so','yes', 'it', 'until', 'you','to','or','get','told','would','week','us','test','right','left','one','even','also','go','ask' ]
    tweet_file_name="tweets.json"
    twitter_stream= TweetExtratcor()
    twitter_stream.extractTweet(filter_list);

