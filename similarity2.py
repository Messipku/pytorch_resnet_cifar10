import torch
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

data_filename = "/tmp/model_tests.10000.pt"

# https://arxiv.org/abs/1810.11750
# Maximum Matching Similarity
def calc_similarity1(x, y, epsilon):
    x = x.numpy()
    y = y.numpy()
    nr_cols = (x.shape[1] + y.shape[1])

    while True:
        try:
            y_basis = linalg.orth(y)
            x_cols = []
            for i in range(x.shape[1]):
                vec = x[:, i:i+1]
                norm = np.sqrt((vec ** 2).sum())
                if norm <= np.finfo(type(norm)).eps * x.max() * x.shape[0]:
                    continue
                vec = vec / norm
                weights = np.matmul(vec.T, y_basis)
                proj = np.matmul(y_basis, weights.T)
                remain = vec - proj
                norm = np.sqrt((remain ** 2).sum())
                if norm >= epsilon:
                    x_cols.append(i)

            x_basis = linalg.orth(x)
            y_cols = []
            for j in range(y.shape[1]):
                vec = y[:, j:j+1]
                norm = np.sqrt((vec ** 2).sum())
                if norm <= np.finfo(type(norm)).eps * y.max() * y.shape[0]:
                    continue
                vec = vec / norm
                weights = np.matmul(vec.T, x_basis)
                proj = np.matmul(x_basis, weights.T)
                remain = vec - proj
                norm = np.sqrt((remain ** 2).sum())
                if norm >= epsilon:
                    y_cols.append(j)

            x = np.delete(x, x_cols, axis=1)
            y = np.delete(y, y_cols, axis=1)
            if not x_cols and not y_cols:
                break
        except ValueError:
            return 0

    return (x.shape[1] + y.shape[1]) / nr_cols

# https://arxiv.org/abs/1905.00414
# Linear CKA
def calc_similarity2(x, y):
    a = torch.matmul(y.t(), x)
    b = torch.matmul(x.t(), x)
    c = torch.matmul(y.t(), y)
    a = (a ** 2).sum()
    b = (b ** 2).sum().sqrt()
    c = (c ** 2).sum().sqrt()
    z = (a / (b * c)).item()
    return z
#基于曲线积分的相似度
def calc_similarity3(x,y):
    sum1=0
    sum2=0
    x=sorted((x-min(x))/(max(x)-min(x)))
    y=sorted((y-min(y))/(max(y)-min(y)))
    lx=len(x)
    ly=len(y)
    if lx<ly:
        for i in range(ly):
            sum1=sum1+min(x[(i*lx)//ly],y[i])
            sum2=sum2+max(x[(i*lx)//ly],y[i])
    else:
        for i in range(lx):
            sum1=sum1+min(x[i],y[(i*ly)//lx])
            sum2=sum2+max(x[i],y[(i*ly)//lx])
    return sum1/sum2
def main():
    X, Y = torch.load(data_filename)

    X_sum = [x.sum(dim=0) for x in X]
    Y_sum = [y.sum(dim=0) for y in Y]

    

    for x, y in zip(X_sum[0:], X_sum[1:]):
        #plt.plot(sorted(list(x.numpy())))
        #plt.plot(sorted(list(y.numpy())))
        #plt.show()
        sim3=calc_similarity3(x.numpy(),y.numpy())
        print(f"{sim3}")

    #for x, y in zip(X[0:], Y[0:]):
    #    print(f"x.shape = {x.shape}, y.shape = {y.shape}")
    #    sim1 = calc_similarity1(x, y, 0.3)
    #    #sim2 = calc_similarity2(x, y)
    #    print(f"Similarity1: {sim1}")
    #    print('')

if __name__ == "__main__":
    main()
