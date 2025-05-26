from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import logging
import warnings
warnings.filterwarnings('ignore')
from log_file import phase_1
logger = phase_1("main")


class MNIST:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            self.Y = self.df.iloc[:, 0].values
            self.X = self.df.iloc[:, 1:].values

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.Y, test_size=0.2, random_state=42
            )

            logger.info(f"X_train shape:{self.X_train.shape}")
            logger.info(f"X_test shape: {self.X_test.shape}")
            logger.info(f"y_train shape:{self.y_train.shape}")
            logger.info(f"y_test shape:{self.y_test.shape}")

            self.find_best_k()

            self.LR()
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.warning(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")



    def find_best_k(self):
        try:
            val_acc = []
            k_values = np.arange(3, 11, 2)
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(self.X_train, self.y_train)
                val_acc.append(accuracy_score(self.y_test, knn.predict(self.X_test)))
            best_k = k_values[val_acc.index(max(val_acc))]
            logger.info(f"Best K: {best_k} with Accuracy: {max(val_acc)}")
            self.knn_reg = KNeighborsClassifier(n_neighbors=best_k)
            self.knn_reg.fit(self.X_train, self.y_train)
            logger.info(f"____TRAINING PERFORMANCE______ of KNN:  {accuracy_score(self.y_train, self.knn_reg.predict(self.X_train))}")
            logger.info(f"____TRAINING PERFORMANCE______ of KNN WITH CONFUSION MATRIX:\n{confusion_matrix(self.y_train, self.knn_reg.predict(self.X_train))}")
            logger.info(f"___TRAINING PERFORMANCE______OF KNN CLASSIFICATION REPORT:\n{classification_report(self.y_train, self.knn_reg.predict(self.X_train))}")
            y_pred = self.knn_reg.predict(self.X_test)
            logger.info(f"____TESTING PERFORMANCE______ of KNN:  {accuracy_score(self.y_test, y_pred)}")
            logger.info(f"____TESTING PERFORMANCE______ of KNN WITH CONFUSION MATRIX:\n{confusion_matrix(self.y_test, y_pred)}")
            logger.info(f"____TESTING PERFORMANCE______ of KNN WITH CLASSIFICATION REPORT:\n{classification_report(self.y_test, y_pred)}")
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def LR(self):
        try:
            log_reg = LogisticRegression()
            log_reg.fit(self.X_train, self.y_train)
            logger.info(f"____TRAINING PERFORMANCE______ of LR:  {accuracy_score(self.y_train, log_reg.predict(self.X_train))}")
            logger.info(f"____TRAINING PERFORMANCE______ of LR WITH CONFUSION MATRIX: \n{confusion_matrix(self.y_train, log_reg.predict(self.X_train))}")
            logger.info(f"___TRAINING PERFORMANCE______OF LR CLASSIFICATION REPORT: \n{classification_report(self.y_train, log_reg.predict(self.X_train))}")
            y_pred1 = log_reg.predict(self.X_test)
            logger.info(f"____TESTING PERFORMANCE______ of LR:  {accuracy_score(self.y_test, y_pred1)}")
            logger.info(f"____TESTING PERFORMANCE______ of LR WITH CONFUSION MATRIX:\n{confusion_matrix(self.y_test, y_pred1)}")
            logger.info(f"____TESTING PERFORMANCE______ of LR WITH CLASSIFICATION REPORT:\n {classification_report(self.y_test, y_pred1)}")

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

    def predict_image(self, index):
        try:
            if index < 0 or index >= len(self.df):
                raise ValueError("Invalid index. Choose a valid row index from dataset.")
            image = self.df.iloc[index, 1:].to_numpy().reshape(28, 28)
            image_flattened = image.reshape(1, 784) / 255.0
            prediction = self.knn_reg.predict(image_flattened)

            # Display the image
            plt.figure(figsize=(5, 3))
            plt.imshow(image)
            plt.title(f"Predicted Label: {prediction[0]}")
            plt.axis("off")
            plt.show()

            return prediction[0]
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.info(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


if __name__ == "__main__":
    try:
        path = "mnist_test.csv"
        obj = MNIST(path)


        # Predict and display an image
        obj.predict_image(8)

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.info(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")