# import pandas as pd

# obj = pd.read_pickle(r'classes.npy')

# import os
# import pickle

# filein0 = open('classes.npy', "rb")
# t0 = pickle.load(filein0)
# arr = t0[1]
import numpy as np

#data = np.random.normal(0, 1, 100)
data = ["","medical","furniture"]
np.save('listClasses.npy', data)
#This works!!!
# data = np.load('classes.npy', allow_pickle=True)
data = np.load('listClasses.npy', allow_pickle=True)
print(data)

# PICKLE_FILE = 'classes.npy'


# def main():
#     # append data to the pickle file
#     # add_to_pickle(PICKLE_FILE, 123)
#     # add_to_pickle(PICKLE_FILE, 'Hello')
#     # add_to_pickle(PICKLE_FILE, None)
#     # add_to_pickle(PICKLE_FILE, b'World')
#     # add_to_pickle(PICKLE_FILE, 456.789)
#     # load & show all stored objects
#     for item in read_from_pickle(PICKLE_FILE):
#         print(repr(item))
#     os.remove(PICKLE_FILE)


# def add_to_pickle(path, item):
#     with open(path, 'ab') as file:
#         pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


# def read_from_pickle(path):
#     with open(path, 'rb') as file:
#         try:
#             while True:
#                 yield pickle.load(file)
#         except EOFError:
#             pass


# if __name__ == '__main__':
#     main()