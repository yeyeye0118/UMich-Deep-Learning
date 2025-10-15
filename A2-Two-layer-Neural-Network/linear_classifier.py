"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional
import numpy as np
import math

def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] #分类数C
    num_train = X.shape[0]   #照片数N
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #######################################################################
                # TODO:                                                               #
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.        
                # 计算损失函数的 SVM 项的梯度，并将其存储在 dW 上 （第 1 部分）
                # 与其先计算损失，然后计算导数，不如在计算损失的同时计算导数很简单。        #
                #######################################################################
                # Replace "pass" statement with your code
                dW[:,j] +=X[i]
                dW[:,y[i]] -=X[i]
                #######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    
    dW /= num_train
    dW += 2*reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    # 确保 dtype 一致，提升数值稳定性（特别是做梯度检查时用 float64）
    #W = W.to(X.dtype) 
    N = X.shape[0]
    scores = X @ W
    correct_scores = scores.gather(1, y.unsqueeze(1))
    margins = torch.relu(scores - correct_scores + 1.0)
    margins.scatter_(1, y.unsqueeze(1), 0.0)
    loss = margins.sum().div_(X.shape[0]) + reg * W.square().sum()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code
    
    # ---------------- Gradient ----------------
    # binary mask: 违反约束的类标记 1
    binary = margins.gt(0).to(X.dtype)            # (N,C)

    # 每个样本违反的类别数 (N,)
    row_sum = binary.sum(dim=1)

    # 正确类的梯度系数 = -违反类别数
    binary[torch.arange(N), y] = -row_sum

    # 计算 dW
    dW = (X.T @ binary).div_(N) + (2.0 * reg * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    从训练数据中随机采样 batch_size 个元素及其对应的标签，
    用于本轮的 gradient descent。
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: 将采样到的数据存到 X_batch 中，并把对应的标签存到 y_batch 中； #
    # 采样后，X_batch 的形状应为 (batch_size, dim)，                     #
    # y_batch 的形状应为 (batch_size,)。                                    #
    #                                                                       #
    # 提示：使用 torch.randint 来生成索引。                                #
    #########################################################################
    # 在这里用你的代码替换 "pass"
    indices = torch.randint(0, num_train-1, (batch_size,))
    X_batch=X[indices]
    y_batch=y[indices]
    #########################################################################
    #                       代码结束                                        #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,  # 损失函数（比如 SVM loss 或 Softmax loss）
    W: torch.Tensor,      # 权重矩阵 (D, C)，D是特征维度，C是类别数
    X: torch.Tensor,      # 训练数据 (N, D)，N是样本数
    y: torch.Tensor,      # 标签 (N,)
    learning_rate: float = 1e-3, # 学习率
    reg: float = 1e-5,           # 正则化强度
    num_iters: int = 100,        # 迭代次数
    batch_size: int = 200,       # 每次迭代用多少训练样本
    verbose: bool = False,       # 是否打印训练过程
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = int(torch.max(y).item()) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item()) #loss.item() 会把一个只有单个数值的张量 转成 Python 标量（float 或 int）。

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # Replace "pass" statement with your code
        W = W - grad*learning_rate
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    使用此线性分类器的训练权重来预测
    数据点。

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Replace "pass" statement with your code
    scores = X @ W   # (N, D) @ (D, C) -> (N, C)
    # 取每行最大值的索引作为预测类别
    y_pred = torch.argmax(scores, dim=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    返回 SVM 模型的候选超参数。你需要为每个参数至少提供两个取值，
    并且总的网格搜索组合数应小于 25。

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
                      学习率候选值，例如 [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
                                正则化强度候选值，例如 [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO:   add your own hyper parameter lists.                             #
    # 你的任务：在此添加你自己的超参数列表。                                      #
    ###########################################################################
    # Replace "pass" statement with your code
    # 用你的代码替换 "pass" 语句
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    regularization_strengths = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    #                           代码结束                                       #
    ###########################################################################

    return learning_rates, regularization_strengths



def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    训练一个 LinearClassifier 实例，并返回训练好的实例及其在训练/验证集上的准确率。

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
                              新建的 LinearClassifier 实例，训练/验证都将在此实例上进行
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
                        包含 ['X_train', 'y_train', 'X_val', 'y_val'] 键的字典，
                        用于训练分类器
    - lr (float): learning rate parameter for training a SVM instance.
                  用于训练 SVM 的学习率参数
    - reg (float): a regularization weight for training a SVM instance.
                   用于训练 SVM 的正则化权重
    - num_iters (int, optional): a number of iterations to train
                                 训练的迭代次数（可选参数）

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
                              训练好的 LinearClassifier 实例，
                              使用 (['X_train', 'y_train'], lr, reg) 迭代 num_iter 次得到
    - train_acc (float): training accuracy of the svm_model
                         SVM 模型在训练集上的准确率
    - val_acc (float): validation accuracy of the svm_model
                       SVM 模型在验证集上的准确率
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
                     # that are correctly classified.
                     # 准确率即被正确分类的数据点所占比例
    val_acc = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    # 编写代码，在训练集上训练一个线性 SVM，并计算它在训练集和验证集上的准确率
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    # 提示：当你确认验证代码正常工作后，应使用最终的 num_iters 重新运行验证代码。
    # 在此之前，可以先用较小的 num_iters 进行测试。
    ###########################################################################
    # Feel free to uncomment this, at the very beginning,
    # and don't forget to remove this line before submitting your final version
    # 你可以在一开始取消注释下面这行，
    # 但请不要忘记在提交最终版本前删除它
    # num_iters = 100

    # Replace "pass" statement with your code
    # 用你的代码替换 "pass" 语句
    #
    X_train,y_train = data_dict['X_train'], data_dict['y_train']
    X_val,y_val = data_dict['X_val'],data_dict['y_val']
    cls.train(X_train,y_train,lr,reg,num_iters)

    y_train_pre = cls.predict(X_train)
    train_acc = (y_train==y_train_pre).float().mean().item()

    y_val_pre = cls.predict(X_val)
    val_acc = (y_val==y_val_pre).float().mean().item()

    ############################################################################
    #                            END OF YOUR CODE                              #
    #                            代码结束                                       #
    ############################################################################

    return cls, train_acc, val_acc



# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  
    Softmax 损失函数，naive（朴素）实现，使用显式循环。
    
    When you implement the regularization over W, please DO NOT multiply the regularization term by 1/2 (no coefficient).
    在实现对 W 的正则化时，请不要乘以 1/2（没有系数）。

    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.
    输入数据维度为 D，总共有 C 类，我们对 N 个样本的小批量进行操作。

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
      W: 权重张量，形状为 (D, C)
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
      X: 小批量数据，形状为 (N, D)
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
      y: 标签张量，形状为 (N,)，y[i]=c 表示第 i 个样本的标签为 c，0 <= c < C
    - reg: (float) regularization strength
      reg: 正则化强度

    Returns a tuple of:
    - loss as single float
      损失，单个浮点数
    - gradient with respect to weights W; a tensor of same shape as W
      权重 W 的梯度，与 W 形状相同的张量
    """
    # Initialize the loss and gradient to zero.
    # 初始化损失和梯度为 0
    loss = 0.0
    dW = torch.zeros_like(W)  # dW 用于存储梯度，形状与 W 相同

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # TODO: 使用显式循环计算 softmax 损失及其梯度
    # 将损失存储在 loss 中，将梯度存储在 dW 中
    # 注意数值稳定性（参考 CS231n Numeric Stability）
    # 不要忘记正则化
    #W[D,C],X[N,D],X[i] [1,D]
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]
    L = np.zeros(N)
    for i in range(N):
        f=(X[i]@W).T.squeeze()  #[C,1]-->[C]
        f-=torch.max(f)   
        p=torch.exp(f)/torch.sum(torch.exp(f))
        L[i]=-math.log(p[y[i]])

        dW[:,y[i]]+=-X[i].T
        for j in range(C):
            dW[:,j]+=p[j]*X[i].T

    data_loss=L.mean()
    reg_loss=reg*torch.sum(W**2)
    dW /=N
    dW_reg_loss=2*reg*W
    dW +=dW_reg_loss
    loss = data_loss+reg_loss
    # Replace "pass" statement with your code
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW  # 返回损失和梯度



def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    #W[D,C],X[N,D],X[i] [1,D]
        # ===============================================================
    # Forward pass
    # 前向计算
    # ===============================================================
    scores = X @ W                     # (N, C) 计算所有样本的分类分数
    scores -= torch.max(scores, dim=1, keepdim=True).values  
    # 数值稳定性：每个样本减去最大值，避免 exp 溢出

    exp_scores = torch.exp(scores)     # (N, C)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)  
    # softmax 概率 (N, C)

    # 取出正确类别的概率
    correct_logprobs = -torch.log(probs[torch.arange(X.shape[0]), y])  
    # (N,) 交叉熵损失

    data_loss = torch.mean(correct_logprobs)  # 平均损失
    reg_loss = reg * torch.sum(W ** 2)        # 正则化损失
    loss = data_loss + reg_loss               # 总损失

    # ===============================================================
    # Backward pass
    # 反向传播计算梯度
    # ===============================================================
    dscores = probs.clone()                   # (N, C)
    dscores[torch.arange(X.shape[0]), y] -= 1  # p_j - 1 for correct class
    dscores /= X.shape[0]                      # 平均化

    dW = X.T @ dscores                        # (D, N) @ (N, C) -> (D, C)
    dW += 2 * reg * W                         # 正则化项梯度



    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    regularization_strengths = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths
