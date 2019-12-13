import psycopg2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense, SpatialDropout1D
from keras.layers import Flatten, Conv1D, LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import re

try:
    connection = psycopg2.connect(user="admin",
                                  password="FosterInsight.2020",
                                  host="127.0.0.1",
                                  port="3308",
                                  database="dbappalign")
    cursor = connection.cursor()
    postgreSQL_select_Query = "select * from data_source_comment ORDER BY id"
    cursor.execute(postgreSQL_select_Query)
    print("Selecting rows from mobile table using cursor.fetchall")
    mobile_records = cursor.fetchall()
    dfObj = pd.DataFrame(data=mobile_records,
                         columns=['id', 'writenby', 'date', 'rate', 'city', 'text', 'bank_id']).sort_values(by='id')

    dfObj.to_csv(r'C:\Users\Samsung\Desktop\comments.csv')


except (Exception, psycopg2.Error) as error:
    print("Error while fetching data from PostgreSQL", error)

pd.set_option('display.max_columns', 500)

#df = dfObj[['text','sentiment']]
#df_new = pd.DataFrame(df)
#print(df[0])
# dfObj.insert(7, "sentiment", "null")

rate = dfObj['rate']
sentiment = []
for i in range(1229):
    if rate[i] == 1 or rate[i]== 2 or rate[i]==3:
        sentiment.append("negative")
    else:
        sentiment.append("positive")

dfObj.insert(7, "sentiment", sentiment)
dfObj=dfObj.drop(["id","writenby","date","rate","city","bank_id"],axis=1)
df2=pd.read_csv('./pozitif yorum.csv', encoding='ISO-8859-9')
dfObj=pd.concat([dfObj,df2],ignore_index=True)

#!!!!!!!!!!!!!! STOPWORDS !!!!!!!!!!!!!!

dfObj = dfObj.reset_index(drop=True)
STOPWORDS = set(stopwords.words('turkish'))


def clean_text(text):

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


dfObj['text'] = dfObj['text'].apply(clean_text)
dfObj['text'] = dfObj['text'].str.replace('\d+', '')
#!!!!!!!!!!!!!! STOPWORDS END !!!!!!!!!!!!!!1

sns.countplot(x='sentiment', data=dfObj)
plt.show()

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(dfObj['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(dfObj['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(dfObj['sentiment'].values)
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 32

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))



plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


def new_comment(text):
    new_complaint = [text]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['negative', 'positive']
    return print(pred, labels[np.argmax(pred)])

#<---KONSOLDA DENEME İÇİN ---> new_comment('DENEME CÜMLESİ')



