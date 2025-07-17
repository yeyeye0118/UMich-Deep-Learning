import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    print(num_test)
    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      print(i)
                #x[i,:]被广播了，输入的X是测试集
                #L1距离
     #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))  L2距离
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred






import data_utils
Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('C:/Users/17737/Desktop/UMich_DLfCV/lecture2/project/data/cifar10/')


#print(Xtr)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
Xte_rows_sub= Xte_rows[0:200,:]

nn = NearestNeighbor()
nn.train(Xtr_rows,Ytr)
ans_Yte=nn.predict(Xte_rows_sub)
Yte_sub=Yte[0:200]

print ('accuracy: %f' %np.mean(ans_Yte == Yte_sub) )

#鉴于10000要跑十多分钟，用了sub子集跑前200个 得到的精度26%