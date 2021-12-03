# from random import Random
import random
seed = 1337


def ft_train_test_split(*arrays, test_size=0.25, random_state=None, shuffle=True):
    if shuffle:
        ft_shuffle(array[0])
        ft_shuffle(array[1])
        
    _slice = int(len(arrays[0]) * test_size)
    X_train = arrays[0][_slice:]
    X_test = arrays[0][:_slice]
    y_train = arrays[0][:_slice]
    y_test = arrays[0][_slice:]
    
    return (X_train, X_test, y_train, y_test)


def ft_shuffle(arr):
    random.Random(seed).shuffle(arr)




array = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
X_train, X_test, y_train, y_test = ft_train_test_split(array)
print(X_train, X_test, y_train, y_test)
