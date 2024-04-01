import numpy as np


def eig_invpower(A, v0, pre, p=0):
    uk = v0
    flag = 1
    val_old = 0
    n = 0
    if p != 0:
        A = A - p * np.eye(len(A))
    LU, La, Ua, order0, order1 = Doolittle_pivot(np.asarray(A))  # PA=LU
    while flag:
        n = n + 1
        vk = solveLineq(La, Ua, np.asarray(uk)[order1, :])
        val = vk[np.argmax(np.abs(vk))]
        uk = np.asmatrix(vk.reshape(len(A), 1)) / val
        print(np.asarray(uk).flatten())
        if (np.abs(1 / val - val_old) < pre):
            flag = 0
        val_old = 1 / val
    print('min eigenvalue:', 1 / val + p)
    print('eigenvector:', np.asarray(uk).flatten())
    print('iteration:', n)
    return 1 / val + p, uk


def Doolittle_pivot(A):  # A为np.array，而不是np.matrix
    n = len(A)
    LU = A.copy()
    order1 = np.arange(n)
    for r in range(n):
        ut = LU[:r, r].reshape(r, 1)
        si = A[r:, r] - np.sum(ut * LU[r:, :r].T, axis=0)
        ir = np.argmax(np.abs(si))
        if ir != 0:
            LU[[r, r + ir], :] = LU[[r + ir, r], :]
            order1[[r, r + ir]] = order1[[r + ir, r]]
        lt = LU[r, :r].reshape(r, 1)
        LU[r, r:] = LU[r, r:] - np.sum(lt * LU[:r, r:], axis=0)
        if r == n - 1:
            continue
        LU[r + 1:, r] = (LU[r + 1:, r] - np.sum(ut * LU[r + 1:, :r].T, axis=0)) / LU[r, r]
    U = np.triu(LU)
    L = np.tril(LU) - np.diag(np.diag(LU)) + np.eye(n)
    order0 = []
    [order0.insert(i, np.where(order1 == i)[0][0]) for i in range(n)]
    return LU, L, U, order0, order1


def solveLineq(L, U, b):  # b为np.array，而不是np.matrix
    rows = len(b)
    y = np.zeros(rows)
    y[0] = b[0] / L[0, 0]
    for k in range(1, rows):
        y[k] = (b[k] - np.sum(L[k, :k] * y[:k])) / L[k, k]
    x = np.zeros(rows)
    k = rows - 1
    x[k] = y[k] / U[k, k]
    for k in range(rows - 2, -1, -1):
        x[k] = (y[k] - np.sum(x[k + 1:] * U[k, k + 1:])) / U[k, k]
    return x


if __name__ == '__main__':
    A = np.matrix([[5, -1, 1],
                   [-1, 2, 0],
                   [1, 0, 3]], dtype='float')
    v0 = np.matrix([[1], [1], [1]], dtype='float')
    pre = 1e-10
    val, uk = eig_invpower(A, v0, pre, 1.2679)
