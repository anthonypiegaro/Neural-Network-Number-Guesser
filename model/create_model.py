from utils.build_model import build_model
from utils.clean_data import clean_data
from utils.load_mnist_data import load_mnist_data
from utils.test_model import test_model

X_train, y_train, X_test, y_test = load_mnist_data()

X_train = clean_data(X_train)
X_test = clean_data(X_test)

model = build_model()

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Log accuracy
accuracy = test_model(model, X_test, y_test)
print(accuracy)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
