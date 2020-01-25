from keras.layers import Activation, Dense, Dropout
from keras.layers import Input, Embedding, merge, LSTM
from keras.models import Sequential, Model
from keras.optimizers import Adam
import settings

def LSTM_model():

	INPUT_LENGTH = settings.max_length
	dictionary_length = settings.dictionary_length
	encoder_inputs = Input(shape=(INPUT_LENGTH,))
	encoder = Embedding(dictionary_length, 128, input_length=INPUT_LENGTH, mask_zero=True)(encoder_inputs)
	encoder = LSTM(512, return_sequences= False)(encoder)
	encoder = Dropout(0.2)(encoder)
	output = Dense(dictionary_length, activation="softmax")(encoder)
	model = Model(inputs=encoder_inputs, outputs=output)

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

#plot_model(model, show_shapes=True, show_layer_names=True)