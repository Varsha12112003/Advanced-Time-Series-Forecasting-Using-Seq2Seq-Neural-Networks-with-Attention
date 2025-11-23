from tensorflow.keras import layers, Model

def build_seq2seq_attention(input_dim, timesteps, output_dim):
    encoder_inputs = layers.Input(shape=(timesteps, input_dim))
    encoder_lstm = layers.LSTM(128, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    attention = layers.Attention()([encoder_outputs, encoder_outputs])

    decoder_inputs = layers.Input(shape=(timesteps, input_dim))
    decoder_lstm = layers.LSTM(128, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h,state_c])

    outputs = layers.TimeDistributed(layers.Dense(output_dim))(decoder_outputs + attention)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
