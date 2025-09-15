from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data = load_wine()
X = data.data
y = data.target
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Acurácia Rede Neural:", acc)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train.argmax(axis=1))
y_pred_rf = rf.predict(X_test)
print("Acurácia RandomForest:", accuracy_score(y_test.argmax(axis=1), y_pred_rf))
