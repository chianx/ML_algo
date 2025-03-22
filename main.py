import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import ADASYN  # Changed from SMOTE to ADASYN

def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    sequence_length = 100  # 29 previous + 1 current
    sequences, targets = [], []

    # Create sequences of 30 rows (including the current signal)
    for i in range(sequence_length - 1, len(data)):
        sequences.append(data.iloc[i - sequence_length + 1:i + 1].drop(columns=['my_indicator', 'correct']).values)
        targets.append(data.iloc[i]['correct'])

    sequences = np.array(sequences)
    targets = np.array(targets)

    # Normalize features
    scaler = MinMaxScaler()
    num_features = sequences.shape[2]
    sequences = sequences.reshape(-1, num_features)
    sequences = scaler.fit_transform(sequences)
    sequences = sequences.reshape(-1, sequence_length, num_features)

    # Convert target labels to categorical
    label_map = {-1: 0, 0.5: 1, 1: 2}
    targets = np.array([label_map[x] for x in targets])
    targets = to_categorical(targets, num_classes=3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42, shuffle=True)

    # Apply ADASYN to balance the dataset
    adasyn = ADASYN(random_state=1)
    X_train_res, y_train_res = adasyn.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
    X_train_res = X_train_res.reshape(-1, sequence_length, num_features)
    y_train_res = to_categorical(y_train_res, num_classes=3)

    return X_train_res, X_test, y_train_res, y_test, scaler

def build_lstm_model(input_shape):
    sequence_input = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(sequence_input)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(16, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    # x = Dense(16, activation='relu')(x)
    out = Dense(3, activation='softmax')(x)

    model = Model(inputs=sequence_input, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Load dataset
data = pd.read_excel("DOGEUSDT_5m_250d.xlsx").drop(columns=["timestamp"])
data = data[data['my_indicator'].isin(['BUY', 'SELL'])]

# Separate BUY and SELL data
buy_data = data[data['my_indicator'] == 'BUY']
sell_data = data[data['my_indicator'] == 'SELL']

# Preprocess separately for BUY and SELL models
X_train_buy, X_test_buy, y_train_buy, y_test_buy, buy_scaler = preprocess_data(buy_data)
X_train_sell, X_test_sell, y_train_sell, y_test_sell, sell_scaler = preprocess_data(sell_data)

# Train BUY model
buy_model = build_lstm_model((X_train_buy.shape[1], X_train_buy.shape[2]))
buy_model.fit(X_train_buy, y_train_buy, epochs=65, batch_size=32, validation_data=(X_test_buy, y_test_buy))
buy_model.save("buy_model.keras")

# Train SELL model
sell_model = build_lstm_model((X_train_sell.shape[1], X_train_sell.shape[2]))
sell_model.fit(X_train_sell, y_train_sell, epochs=65, batch_size=32, validation_data=(X_test_sell, y_test_sell))
sell_model.save("sell_model.keras")

# Model evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    label_map = {-1: 0, 0.5: 1, 1: 2}

    label_counts = {label: (y_test_labels == idx).sum() for label, idx in label_map.items()}
    pred_counts = {label: (y_pred == idx).sum() for label, idx in label_map.items()}
    correct_counts = {label: ((y_pred == y_test_labels) & (y_test_labels == idx)).sum() for label, idx in label_map.items()}

    print(f"\n🔹 **{model_name} Model Summary**")
    print(f"   - Total {model_name} signals: {len(y_test)}")
    print("\n🔹 **Label Distribution in Test Set**:")
    for label, count in label_counts.items():
        print(f"   - {label}: {count}")
    print("\n🔹 **Predictions Made by Model**:")
    for label, count in pred_counts.items():
        print(f"   - Predicted as {label}: {count}")
    print("\n🔹 **Correct Predictions**:")
    for label, count in correct_counts.items():
        print(f"   - Correctly Predicted {label}: {count} / {label_counts[label]} ({(count/label_counts[label])*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=['-1', '0.5', '1']))

# Evaluate both models
evaluate_model(buy_model, X_test_buy, y_test_buy, "BUY")
evaluate_model(sell_model, X_test_sell, y_test_sell, "SELL")
