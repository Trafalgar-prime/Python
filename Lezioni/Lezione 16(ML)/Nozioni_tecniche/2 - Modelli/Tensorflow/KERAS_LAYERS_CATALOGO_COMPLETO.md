# Firme costruttori layer (keras)

## Activation
`keras.layers.Activation(activation, **kwargs)`

## ActivityRegularization
`keras.layers.ActivityRegularization(l1=0.0, l2=0.0, **kwargs)`

## AdaptiveAveragePooling1D
`keras.layers.AdaptiveAveragePooling1D(output_size, data_format=None, **kwargs)`

## AdaptiveAveragePooling2D
`keras.layers.AdaptiveAveragePooling2D(output_size, data_format=None, **kwargs)`

## AdaptiveAveragePooling3D
`keras.layers.AdaptiveAveragePooling3D(output_size, data_format=None, **kwargs)`

## AdaptiveMaxPooling1D
`keras.layers.AdaptiveMaxPooling1D(output_size, data_format=None, **kwargs)`

## AdaptiveMaxPooling2D
`keras.layers.AdaptiveMaxPooling2D(output_size, data_format=None, **kwargs)`

## AdaptiveMaxPooling3D
`keras.layers.AdaptiveMaxPooling3D(output_size, data_format=None, **kwargs)`

## Add
`keras.layers.Add(**kwargs)`

## AdditiveAttention
`keras.layers.AdditiveAttention(use_scale=True, dropout=0.0, **kwargs)`

## AlphaDropout
`keras.layers.AlphaDropout(rate, noise_shape=None, seed=None, **kwargs)`

## Attention
`keras.layers.Attention(use_scale=False, score_mode='dot', dropout=0.0, seed=None, **kwargs)`

## AugMix
`keras.layers.AugMix(value_range=(0, 255), num_chains=3, chain_depth=3, factor=0.3, alpha=1.0, all_ops=True, interpolation='bilinear', seed=None, data_format=None, **kwargs)`

## AutoContrast
`keras.layers.AutoContrast(value_range=(0, 255), **kwargs)`

## Average
`keras.layers.Average(**kwargs)`

## AveragePooling1D
`keras.layers.AveragePooling1D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## AveragePooling2D
`keras.layers.AveragePooling2D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## AveragePooling3D
`keras.layers.AveragePooling3D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## AvgPool1D
`keras.layers.AvgPool1D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## AvgPool2D
`keras.layers.AvgPool2D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## AvgPool3D
`keras.layers.AvgPool3D(pool_size, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## BatchNormalization
`keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, synchronized=False, **kwargs)`

## Bidirectional
`keras.layers.Bidirectional(layer, merge_mode='concat', weights=None, backward_layer=None, **kwargs)`

## CategoryEncoding
`keras.layers.CategoryEncoding(num_tokens=None, output_mode='multi_hot', sparse=False, **kwargs)`

## CenterCrop
`keras.layers.CenterCrop(height, width, data_format=None, **kwargs)`

## Concatenate
`keras.layers.Concatenate(axis=-1, **kwargs)`

## Conv1D
`keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Conv1DTranspose
`keras.layers.Conv1DTranspose(filters, kernel_size, strides=1, padding='valid', output_padding=None, data_format=None, dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Conv2D
`keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Conv2DTranspose
`keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Conv3D
`keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Conv3DTranspose
`keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, output_padding=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## ConvLSTM1D
`keras.layers.ConvLSTM1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, **kwargs)`

## ConvLSTM2D
`keras.layers.ConvLSTM2D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, **kwargs)`

## ConvLSTM3D
`keras.layers.ConvLSTM3D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, **kwargs)`

## Convolution1D
`keras.layers.Convolution1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Convolution1DTranspose
`keras.layers.Convolution1DTranspose(filters, kernel_size, strides=1, padding='valid', output_padding=None, data_format=None, dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Convolution2D
`keras.layers.Convolution2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Convolution2DTranspose
`keras.layers.Convolution2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Convolution3D
`keras.layers.Convolution3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Convolution3DTranspose
`keras.layers.Convolution3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, output_padding=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)`

## Cropping1D
`keras.layers.Cropping1D(cropping=(1, 1), **kwargs)`

## Cropping2D
`keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None, **kwargs)`

## Cropping3D
`keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs)`

## CutMix
`keras.layers.CutMix(factor=1.0, seed=None, data_format=None, **kwargs)`

## Dense
`keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, lora_rank=None, lora_alpha=None, quantization_config=None, **kwargs)`

## DepthwiseConv1D
`keras.layers.DepthwiseConv1D(kernel_size, strides=1, padding='valid', depth_multiplier=1, data_format=None, dilation_rate=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs)`

## DepthwiseConv2D
`keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs)`

## Discretization
`keras.layers.Discretization(bin_boundaries=None, num_bins=None, epsilon=0.01, output_mode='int', sparse=False, dtype=None, name=None)`

## Dot
`keras.layers.Dot(axes, normalize=False, **kwargs)`

## Dropout
`keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)`

## EinsumDense
`keras.layers.EinsumDense(equation, output_shape, activation=None, bias_axes=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None, lora_rank=None, lora_alpha=None, gptq_unpacked_column_size=None, quantization_config=None, **kwargs)`

## ELU
`keras.layers.ELU(alpha=1.0, **kwargs)`

## Embedding
`keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, embeddings_constraint=None, mask_zero=False, weights=None, lora_rank=None, lora_alpha=None, quantization_config=None, **kwargs)`

## Equalization
`keras.layers.Equalization(value_range=(0, 255), bins=256, data_format=None, **kwargs)`

## Flatten
`keras.layers.Flatten(data_format=None, **kwargs)`

## FlaxLayer
`keras.layers.FlaxLayer(module, method=None, variables=None, **kwargs)`

## GaussianDropout
`keras.layers.GaussianDropout(rate, seed=None, **kwargs)`

## GaussianNoise
`keras.layers.GaussianNoise(stddev, seed=None, **kwargs)`

## GlobalAveragePooling1D
`keras.layers.GlobalAveragePooling1D(data_format=None, keepdims=False, **kwargs)`

## GlobalAveragePooling2D
`keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False, **kwargs)`

## GlobalAveragePooling3D
`keras.layers.GlobalAveragePooling3D(data_format=None, keepdims=False, **kwargs)`

## GlobalAvgPool1D
`keras.layers.GlobalAvgPool1D(data_format=None, keepdims=False, **kwargs)`

## GlobalAvgPool2D
`keras.layers.GlobalAvgPool2D(data_format=None, keepdims=False, **kwargs)`

## GlobalAvgPool3D
`keras.layers.GlobalAvgPool3D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPool1D
`keras.layers.GlobalMaxPool1D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPool2D
`keras.layers.GlobalMaxPool2D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPool3D
`keras.layers.GlobalMaxPool3D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPooling1D
`keras.layers.GlobalMaxPooling1D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPooling2D
`keras.layers.GlobalMaxPooling2D(data_format=None, keepdims=False, **kwargs)`

## GlobalMaxPooling3D
`keras.layers.GlobalMaxPooling3D(data_format=None, keepdims=False, **kwargs)`

## GroupNormalization
`keras.layers.GroupNormalization(groups=32, axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)`

## GroupQueryAttention
`keras.layers.GroupQueryAttention(head_dim, num_query_heads, num_key_value_heads, dropout=0.0, use_bias=True, flash_attention=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, **kwargs)`

## GRU
`keras.layers.GRU(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=True, use_cudnn='auto', **kwargs)`

## GRUCell
`keras.layers.GRUCell(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, reset_after=True, seed=None, **kwargs)`

## HashedCrossing
`keras.layers.HashedCrossing(num_bins, output_mode='int', sparse=False, name=None, dtype=None, **kwargs)`

## Hashing
`keras.layers.Hashing(num_bins, mask_value=None, salt=None, output_mode='int', sparse=False, **kwargs)`

## Identity
`keras.layers.Identity(**kwargs)`

## InputLayer
`keras.layers.InputLayer(shape=None, batch_size=None, dtype=None, sparse=None, ragged=None, batch_shape=None, input_tensor=None, optional=False, name=None, **kwargs)`

## IntegerLookup
`keras.layers.IntegerLookup(max_tokens=None, num_oov_indices=1, mask_token=None, oov_token=-1, vocabulary=None, vocabulary_dtype='int64', idf_weights=None, invert=False, output_mode='int', sparse=False, pad_to_max_tokens=False, name=None, **kwargs)`

## JaxLayer
`keras.layers.JaxLayer(call_fn, init_fn=None, params=None, state=None, seed=None, **kwargs)`

## Lambda
`keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None, **kwargs)`

## Layer
`keras.layers.Layer(*args, **kwargs)`

## LayerNormalization
`keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)`

## LeakyReLU
`keras.layers.LeakyReLU(negative_slope=0.3, **kwargs)`

## LSTM
`keras.layers.LSTM(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, use_cudnn='auto', **kwargs)`

## LSTMCell
`keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, **kwargs)`

## Masking
`keras.layers.Masking(mask_value=0.0, **kwargs)`

## Maximum
`keras.layers.Maximum(**kwargs)`

## MaxNumBoundingBoxes
`keras.layers.MaxNumBoundingBoxes(max_number, fill_value=-1, **kwargs)`

## MaxPool1D
`keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MaxPool2D
`keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MaxPool3D
`keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MaxPooling1D
`keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MaxPooling2D
`keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MaxPooling3D
`keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)`

## MelSpectrogram
`keras.layers.MelSpectrogram(fft_length=2048, sequence_stride=512, sequence_length=None, window='hann', sampling_rate=16000, num_mel_bins=128, min_freq=20.0, max_freq=None, power_to_db=True, top_db=80.0, mag_exp=2.0, min_power=1e-10, ref_power=1.0, **kwargs)`

## Minimum
`keras.layers.Minimum(**kwargs)`

## MixUp
`keras.layers.MixUp(alpha=0.2, data_format=None, seed=None, **kwargs)`

## MultiHeadAttention
`keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim=None, dropout=0.0, use_bias=True, output_shape=None, attention_axes=None, flash_attention=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, **kwargs)`

## Multiply
`keras.layers.Multiply(**kwargs)`

## Normalization
`keras.layers.Normalization(axis=-1, mean=None, variance=None, invert=False, **kwargs)`

## Permute
`keras.layers.Permute(dims, **kwargs)`

## Pipeline
`keras.layers.Pipeline(layers, name=None)`

## PReLU
`keras.layers.PReLU(alpha_initializer='Zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None, **kwargs)`

## RandAugment
`keras.layers.RandAugment(value_range=(0, 255), num_ops=2, factor=0.5, interpolation='bilinear', seed=None, data_format=None, **kwargs)`

## RandomBrightness
`keras.layers.RandomBrightness(factor, value_range=(0, 255), seed=None, **kwargs)`

## RandomColorDegeneration
`keras.layers.RandomColorDegeneration(factor, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomColorJitter
`keras.layers.RandomColorJitter(value_range=(0, 255), brightness_factor=None, contrast_factor=None, saturation_factor=None, hue_factor=None, seed=None, data_format=None, **kwargs)`

## RandomContrast
`keras.layers.RandomContrast(factor, value_range=(0, 255), seed=None, **kwargs)`

## RandomCrop
`keras.layers.RandomCrop(height, width, seed=None, data_format=None, name=None, **kwargs)`

## RandomElasticTransform
`keras.layers.RandomElasticTransform(factor=1.0, scale=1.0, interpolation='bilinear', fill_mode='reflect', fill_value=0.0, value_range=(0, 255), seed=None, data_format=None, **kwargs)`

## RandomErasing
`keras.layers.RandomErasing(factor=1.0, scale=(0.02, 0.33), fill_value=None, value_range=(0, 255), seed=None, data_format=None, **kwargs)`

## RandomFlip
`keras.layers.RandomFlip(mode='horizontal_and_vertical', seed=None, data_format=None, **kwargs)`

## RandomGaussianBlur
`keras.layers.RandomGaussianBlur(factor=1.0, kernel_size=3, sigma=1.0, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomGrayscale
`keras.layers.RandomGrayscale(factor=0.5, data_format=None, seed=None, **kwargs)`

## RandomHue
`keras.layers.RandomHue(factor, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomInvert
`keras.layers.RandomInvert(factor=1.0, value_range=(0, 255), seed=None, data_format=None, **kwargs)`

## RandomPerspective
`keras.layers.RandomPerspective(factor=1.0, scale=1.0, interpolation='bilinear', fill_value=0.0, seed=None, data_format=None, **kwargs)`

## RandomPosterization
`keras.layers.RandomPosterization(factor, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomRotation
`keras.layers.RandomRotation(factor, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, data_format=None, **kwargs)`

## RandomSaturation
`keras.layers.RandomSaturation(factor, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomSharpness
`keras.layers.RandomSharpness(factor, value_range=(0, 255), data_format=None, seed=None, **kwargs)`

## RandomShear
`keras.layers.RandomShear(x_factor=0.0, y_factor=0.0, interpolation='bilinear', fill_mode='reflect', fill_value=0.0, data_format=None, seed=None, **kwargs)`

## RandomTranslation
`keras.layers.RandomTranslation(height_factor, width_factor, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, data_format=None, **kwargs)`

## RandomZoom
`keras.layers.RandomZoom(height_factor, width_factor=None, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, data_format=None, **kwargs)`

## ReLU
`keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0, **kwargs)`

## RepeatVector
`keras.layers.RepeatVector(n, **kwargs)`

## Rescaling
`keras.layers.Rescaling(scale, offset=0.0, **kwargs)`

## Reshape
`keras.layers.Reshape(target_shape, **kwargs)`

## Resizing
`keras.layers.Resizing(height, width, interpolation='bilinear', crop_to_aspect_ratio=False, pad_to_aspect_ratio=False, fill_mode='constant', fill_value=0.0, antialias=False, data_format=None, **kwargs)`

## ReversibleEmbedding
`keras.layers.ReversibleEmbedding(input_dim, output_dim, tie_weights=True, embeddings_initializer='uniform', embeddings_regularizer=None, embeddings_constraint=None, mask_zero=False, reverse_dtype=None, logit_soft_cap=None, **kwargs)`

## RMSNormalization
`keras.layers.RMSNormalization(axis=-1, epsilon=1e-06, **kwargs)`

## RNN
`keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, zero_output_for_mask=False, **kwargs)`

## SeparableConv1D
`keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None, **kwargs)`

## SeparableConv2D
`keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None, **kwargs)`

## SeparableConvolution1D
`keras.layers.SeparableConvolution1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None, **kwargs)`

## SeparableConvolution2D
`keras.layers.SeparableConvolution2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None, **kwargs)`

## SimpleRNN
`keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, seed=None, **kwargs)`

## SimpleRNNCell
`keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, seed=None, **kwargs)`

## Softmax
`keras.layers.Softmax(axis=-1, **kwargs)`

## Solarization
`keras.layers.Solarization(addition_factor=0.0, threshold_factor=0.0, value_range=(0, 255), seed=None, **kwargs)`

## SpatialDropout1D
`keras.layers.SpatialDropout1D(rate, seed=None, name=None, dtype=None)`

## SpatialDropout2D
`keras.layers.SpatialDropout2D(rate, data_format=None, seed=None, name=None, dtype=None)`

## SpatialDropout3D
`keras.layers.SpatialDropout3D(rate, data_format=None, seed=None, name=None, dtype=None)`

## SpectralNormalization
`keras.layers.SpectralNormalization(layer, power_iterations=1, **kwargs)`

## StackedRNNCells
`keras.layers.StackedRNNCells(cells, **kwargs)`

## STFTSpectrogram
`keras.layers.STFTSpectrogram(mode='log', frame_length=256, frame_step=None, fft_length=None, window='hann', periodic=False, scaling='density', padding='valid', expand_dims=False, data_format=None, **kwargs)`

## StringLookup
`keras.layers.StringLookup(max_tokens=None, num_oov_indices=1, mask_token=None, oov_token='[UNK]', vocabulary=None, idf_weights=None, invert=False, output_mode='int', pad_to_max_tokens=False, sparse=False, encoding='utf-8', name=None, **kwargs)`

## Subtract
`keras.layers.Subtract(**kwargs)`

## TextVectorization
`keras.layers.TextVectorization(max_tokens=None, standardize='lower_and_strip_punctuation', split='whitespace', ngrams=None, output_mode='int', output_sequence_length=None, pad_to_max_tokens=False, vocabulary=None, idf_weights=None, sparse=False, ragged=False, encoding='utf-8', name=None, **kwargs)`

## TFSMLayer
`keras.layers.TFSMLayer(filepath, call_endpoint='serve', call_training_endpoint=None, trainable=True, name=None, dtype=None)`

## TimeDistributed
`keras.layers.TimeDistributed(layer, **kwargs)`

## TorchModuleWrapper
`keras.layers.TorchModuleWrapper(module, name=None, output_shape=None, **kwargs)`

## UnitNormalization
`keras.layers.UnitNormalization(axis=-1, **kwargs)`

## UpSampling1D
`keras.layers.UpSampling1D(size=2, **kwargs)`

## UpSampling2D
`keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest', **kwargs)`

## UpSampling3D
`keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None, **kwargs)`

## Wrapper
`keras.layers.Wrapper(layer, **kwargs)`

## ZeroPadding1D
`keras.layers.ZeroPadding1D(padding=1, data_format=None, **kwargs)`

## ZeroPadding2D
`keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None, **kwargs)`

## ZeroPadding3D
`keras.layers.ZeroPadding3D(padding=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs)`
