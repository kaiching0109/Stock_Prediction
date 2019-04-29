import keras
from keras.models import Sequential
# from keras.layers import Dense
from keras import layers
# from sklearn.matrics import confusion_matrix

class netural_network():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.classifier = Sequential()
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test
        self.set_input_layer("tanh")
        self.set_hidden_layer("tanh")
        self.set_output_layer("relu")

    # """
    # Weight initialization
    # """
    # def set_weights(self, shape):
    #     weights = tf.random_normal(shape, stddev=0.1)
    #     return tf.Variable(weights)

    """
        Here initializes the weights and the input layer
        Choices in output_dim:
        1. using k-fold
        2. simply use average of node of input and output layers
        in our case, it is 1(price) as input and 1(signal) as output,
        so (1 + 1)/2 which is 1 if we choose average.
    """
    def set_input_layer(self, acti_func):
        self.classifier.add(layers.Dense(1, kernel_initializer="uniform", activation = acti_func, input_dim=1))

    def set_hidden_layer(self, acti_func):
        # self.classifier.add(Dense(output_dim=1, init="uniform",
        #     activation=acti_func))
        self.classifier.add(layers.Dense(1, kernel_initializer="uniform", activation = acti_func))

    def set_output_layer(self, acti_func):
        self.classifier.add(layers.Dense(1, kernel_initializer="uniform", activation = acti_func))
    """
        optimizer_algo is what approach we use to calculate the cost
        Eg: adam Gass_decent
        loss_func is to optimize optimizer_algo
        Eg: Ordinary Least Squares, Logic lost

        batch_size and epoch should be experiment
    """
    def compile(self, optimizer_algo="adam", loss_func="mean_squared_error", epoch=100):
        self.classifier.compile(optimizer=optimizer_algo, loss=loss_func,
            metrics=["accuracy"])

        """
        The batch size defines the number of samples that will be propagated through the network.
        For instance, let's say you have 1050 training samples and you want to \
        set up a batch_size equal to 100. The algorithm takes the first 100 \
        samples (from 1st to 100th) from the training dataset and trains the \
        network. Next, it takes the second 100 samples (from 101st to 200th) \
        and trains the network againself.
        """
        batch_size = (self.X_train.shape)[0]
        self.classifier.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epoch)
        return self.classifier.evaluate(self.X_train, self.Y_train)

    def predict(self):
        y_pred_train = self.classifier.predict(self.X_train)
        y_pred_test = self.classifier.predict(self.X_test)
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        return y_pred_train, y_pred_test
    #
    # def get_confusion_matrix(condition):
    #     y_pred = (y_pred > condition)
    #     return confusion_matrix(x_test, y_pred)
