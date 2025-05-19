def create_sequences(data, targets, seq_length=3000, step=100):
    sequences = []
    target_sequences = []
    for i in range(0, len(data) - seq_length + 1, step):
        sequences.append(data[i:i+seq_length])
        target_sequences.append(targets[i:i+seq_length])
    return np.array(sequences), np.array(target_sequences)

st = pd.read_csv('data_extracted/1.csv', header=None, names=["time", 'signal', "chans"])

# Calculate relative features
st['signal_diff'] = st['signal'].diff().fillna(0)
st['signal_rolling_mean'] = st['signal'].rolling(window=50, center=True).mean().bfill().ffill()
st['signal_relative'] = st['signal'] - st['signal_rolling_mean']
st['signal_zscore'] = (st['signal'] - st['signal_rolling_mean']) / st['signal'].rolling(window=50, center=True).std().bfill().ffill()

# X = st[["signal_relative", "signal_zscore"]].values
X = st[['signal']].values
y = st["chans"].values

scaler = RobustScaler()
X = scaler.fit_transform(X)

seq_len = 3000
X_seq, y_seq = create_sequences(X, y, seq_length=seq_len)

y_seq_categorical = tf.keras.utils.to_categorical(y_seq, num_classes=3)  # Assuming max 2 channels

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_categorical, test_size=0.2, random_state=42)

def build_model():
    # здесь должна быть реализация модели
    # для остальных моделей я использую такой код и тут хорошо бы использовать его (или переделать):
    def peak_aware_loss(y_true, y_pred):
        # Create weights - higher for actual channel openings
        weights = tf.where(y_true[:, :, 1:] > 0, 1.0, 1.0)  # [batch, seq, 2]
        weights = tf.concat([tf.ones_like(weights[:, :, :1]), weights], axis=-1)  # [batch, seq, 3]
        
        # Calculate base loss
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Apply weights
        weighted_loss = loss * tf.reduce_mean(weights, axis=-1)
        return tf.reduce_mean(weighted_loss)
    
    # Custom metrics for class-specific performance
    def class_recall(class_idx):
        def recall(y_true, y_pred):
            y_true_class = y_true[:, :, class_idx]
            y_pred_class = tf.argmax(y_pred, axis=-1) == class_idx
            y_pred_class = tf.cast(y_pred_class, tf.float32)
            
            true_positives = tf.reduce_sum(y_true_class * y_pred_class)
            possible_positives = tf.reduce_sum(y_true_class)
            
            return true_positives / (possible_positives + tf.keras.backend.epsilon())
        recall.__name__ = f'recall_class_{class_idx}'
        return recall
    
    def class_precision(class_idx):
        def precision(y_true, y_pred):
            y_true_class = y_true[:, :, class_idx]
            y_pred_class = tf.argmax(y_pred, axis=-1) == class_idx
            y_pred_class = tf.cast(y_pred_class, tf.float32)
            
            true_positives = tf.reduce_sum(y_true_class * y_pred_class)
            predicted_positives = tf.reduce_sum(y_pred_class)
            
            return true_positives / (predicted_positives + tf.keras.backend.epsilon())
        precision.__name__ = f'precision_class_{class_idx}'
        return precision
    
    # Focused metrics compilation - only what you care about
    model.compile(optimizer='adam',
                loss=peak_aware_loss,
                metrics=[
                    'accuracy',
                    # Class 0 metrics (if you want them)
                    class_recall(0), class_precision(0),
                    # The important ones - class 1 and 2
                    class_recall(1), class_precision(1),
                    class_recall(2), class_precision(2),
                ])
    
    return model

model = build_model(input_shape=(seq_len, X_seq.shape[2]))
history = model.fit(X_train, y_train,
                   epochs=20,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=[
                       tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
                   ])


