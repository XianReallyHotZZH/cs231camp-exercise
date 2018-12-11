import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)    # 1*C 的行向量，得分
    correct_class_score = scores[y[i]]  # 正确分类的得分
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 折叶损失
      if margin > 0:
        loss += margin
        #====== add part =======
        dW[:, y[i]] += -X[i, :]
        dW[:, j] += X[i, :]
        #=======================

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss. 参数正则化损失项
  loss += reg * np.sum(W * W)   

  # 均化dW
  dW /= num_train
  # 考虑进正则化项
  dW += 2 * reg * W 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  X：N*D
  y：N*1
  W：D*C
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero (D*C)

  num_train = X.shape[0]    # N
  num_classes = W.shape[1]  # C

  scores = X.dot(W)         # N*C

  # 得到每一个样本正确分类的得分 (N*1)
  y_correct_class_scores = scores[np.arange(num_train),y].reshape(-1,1)
  # 折叶损失对应的mask（每一样本，贡献损失的错误分类的mask，包含了正确类） N*C
  mask = (scores - y_correct_class_scores + 1) > 0 
  # 折叶损失(包含了正确类) N*C
  zheye_scores = (scores - y_correct_class_scores + 1) * mask
  # 对应于当前样本的总的标量损失(需要减去多加的正确类的损失)
  loss = (np.sum(zheye_scores) - num_train*1) / num_train
  # 加上参数正则化项(L2)
  loss += reg * np.sum(W*W)

  #print('=========loss:%f========='%loss)

  # 初始化ds N*C
  ds = np.ones_like(zheye_scores)
  # 有效的zheye_scores梯度为1，无效的为0
  ds *= mask
  # 更改以下正确分类的情况(由于mask每一行包含了一个正确分类的1，所以后面求和要减1)
  ds[np.arange(num_train),y] = -1 * (np.sum(mask,axis=1)-1)
  # 更新梯度(损失函数部分)
  dW = X.T.dot(ds) / num_train
  # 更新梯度（参数正则化部分）
  dW += 2 * reg * W


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
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
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
