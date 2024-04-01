import numpy as np


def eig_power(A, v0, pre):
    u = v0
    flag = 1
    val_old = 0
    n = 0
    while flag:
        n = n + 1
        vk = A * u
        val = vk[np.argmax(np.abs(vk))]
        u = vk / val
        if (np.abs(val - val_old) < pre):
            flag = 0
        val_old = val
        print(np.asarray(u).flatten(), val)
    print('max eigenvalue:', val)
    print('eigenvector:', np.asarray(u).flatten())
    print('iteration:', n)
    return val, u


if __name__ == '__main__':
    A = np.matrix([[5, -1, 1],
                   [-1, 2, 0],
                   [1, 0, 3]], dtype='float')
    v0 = np.matrix([[1], [1], [1]], dtype='float')
    pre = 1e-8
    val, u = eig_power(A, v0, pre)
