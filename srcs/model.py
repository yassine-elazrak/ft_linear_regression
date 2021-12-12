import numpy as np
import os
import csv
import json
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path_file):
        self.is_file(path_file)
        self.path_file = path_file
        self.X = []
        self.Y = []

    def is_file(self, path_file):
        if not os.path.isfile(path_file):
            raise ValueError("this file does not exist")
        
    def MinMaxScaler(self, nums):
        return  (nums - self.min) / (self.max - self.min)
    
    def setup(self):
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X_ = self.X
        self.Y_ = self.Y
        self.X = self.X.reshape((1, len(self.X)))
        self.Y = self.Y.reshape((1, len(self.Y)))
        self.max = np.amax(self.X)
        self.min = np.amin(self.X)
        self.X = self.MinMaxScaler(self.X)
        # print("data X",self.X)
        
###best val   0.01  10000000
     
    def read_csv(self):
        with open(self.path_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                self.X.append(float(row[0]))
                self.Y.append(float(row[1]))
            self.setup()



class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history_losses = []
        self.history_yhat = []

    def save(self, filename):
        with open(filename, 'w') as f:
            payload = {
                "weights": self.W.tolist(),
                "x_max": self.X_max,
                "x_min": self.X_min,
            }
        json.dump(payload, f)

    
    def ft_load(cls, filename):
        data = json.load(filename)
        return data

    def predict(self, X):
        return np.dot(X.T, self.W).T

    def cost(self, yhat, y):
        C = (1 / (2 * self.m) ) * np.sum(np.power(yhat - y, 2))
        return C

    def gradient_descent(self, W, X, Y, yhat):
        D = np.dot(X, (yhat - Y).T) / self.m
        W = W - self.learning_rate * D
        return W
    def setup_matrix(self,X):
        if not isinstance(X, np.array):
            X = np.array(X)
            
    def ft_predict(self, X):
        ones = np.ones((1, X.shape[1]))
        X = np.append(X, ones, axis=0)
        return np.dot(X.T, self.W).T

    def train(self, X, Y, dataset):
        ones = np.ones((1, X.shape[1]))
        X = np.append(X, ones, axis=0)
        self.m = X.shape[1]
        self.n = X.shape[0]
        self.W = np.zeros((self.n, 1))
        fig, x1 = plt.subplots()
        yhat = self.predict(X)
        # lr = x1.plot(yhat[0], dataset.X_, color="r") 
        # plt.ion()
        
        for _ in range(self.epochs + 1):
            
            yhat = self.predict(X)
            cost = self.cost(yhat, Y)
            # self.history_yhat.append(y)
            if _ % 100 == 0 :
                plt.clf()
                ft_plt(dataset, yhat)
                plt.pause(0.01)
                print("plt ===> clear")
            self.W = self.gradient_descent(self.W, X, Y, yhat)
        # plt.ioff()    
        # plt.show()
        return self.W
# class Plt:
    
#     @classmethod
#     def plt_loss(cls, loss):
#         pass
    
#     @classmethod
#     def plt_predict(cls, x, y):
       
#         plt.scatter([1, 2, 3, 2, 5, 3, 7, 8, 9, 10, 11, 12],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#         plt.plot([])
#         # plt.scatter(x,y, label="high bp low heartrate",color="r")
        
#         plt.ylabel("y_labl")
#         plt.xlabel("x_labl")
#         plt.title("plot_data")
#         plt.legend()
#         plt.show()


def ft_plt(dataset, y):
    
    plt.scatter(dataset.Y_, dataset.X_)
    plt.plot(y[0], dataset.X_, color="r") 
    plt.ylabel("y_labl")
    plt.xlabel("x_labl")
    plt.title("plot_data")
    # plt.legend()
  
    # plt.show()
    # plt.clf()
    
        


if __name__ == '__main__':
    path_file = "./data/data.csv"
    dataset = Dataset(path_file)
    dataset.read_csv()
    lr = LinearRegression()
    w = lr.train(dataset.X, dataset.Y, dataset)
    # print(w)
    # x = np.array([0.21004509, 1])
    X = lr.ft_predict(dataset.X)
    
    # # for i in range(len(dataset.Y[0])):
    # #     print(int(X[0][i]),"=", int(dataset.Y[0][i]),end="\n")
    # # Plt.plt_predict(dataset.X, dataset.Y)
    # plt.scatter(dataset.Y_, dataset.X_)
    # plt.plot(X[0], dataset.X_ )
    #     # plt.scatter(x,y, label="high bp low heartrate",color="r")
        
    # plt.ylabel("y_labl")
    # plt.xlabel("x_labl")
    # plt.title("plot_data")
    # plt.legend()
    # plt.show()
