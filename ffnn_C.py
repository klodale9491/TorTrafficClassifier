from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from util import load_merged_nibble_datasets
from deep_models import create_deep_model_4


# Caricamento datasets
datasets = [
    'AUDIO-STREAMING',
    'BROWSING',
    'EMAIL',
    'CHAT',
    'FILE-TRANSFER',
    'P2P',
    'VIDEO-STREAMING',
    'VOIP'
]
x, y = load_merged_nibble_datasets(datasets, samples=50000, classification='label')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# creazione ed allenamento del modello
model = create_deep_model_4(n_cols=x_train.shape[1], n_rows=x_train.shape[2], classes=16)
# early_stopping_monitor = EarlyStopping(patience=3)
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50,
          callbacks=[], verbose=1)
accuracy = model.evaluate(x_test, y_test, verbose=0)
print('accuracy = ' + str(accuracy))
