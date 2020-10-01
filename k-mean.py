import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import xlrd
# lấy dữ liệu từ excel
file_location = "E:/python/customers_data.xlsx"
wb = xlrd.open_workbook(file_location)
sheet = wb.sheet_by_index(0)
dataEX = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
data = np.array(dataEX, dtype= int)

# tạo tâm cụm ngẫu nhiên
def kmeans_init_centers(data, n_cluster):
    return data[np.random.choice(data.shape[0], n_cluster, replace=False)]

# tính khoảng cách giữa các điểm với tâm cụm, gán vào cụm gần nhất
def kmeans_predict_labels(data, centers):
    D = cdist(data, centers)

    return np.argmin(D, axis = 1)

# cập nhật tâm cụm
def kmeans_update_centers(data, labels, n_cluster):
    centers = np.zeros((n_cluster, data.shape[1]))
    for k in range(n_cluster):

        Xk = data[labels == k, :]

        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

# kiểm tra điều kiện dừng thuật toán, khi tâm cụm mới trùng với tâm cụm cũ
def kmeans_has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

def kmeans_visualize(data, centers, labels, n_cluster, title):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in range(n_cluster):
        X = data[labels == i]  # lấy dữ liệu của cụm i
        plt.plot(X[:, 0], X[:, 1], plt_colors[i] + '^', markersize=4,
                 label='cluster_' + str(i))  # Vẽ cụm i lên đồ thị
        plt.plot(centers[i][0], centers[i][1], plt_colors[i + 2] + 'o', markersize=10,
                 label='center_' + str(i))  # Vẽ tâm cụm i lên đồ thị
    plt.legend()
    plt.pause(1)
    plt.show()

def kmeans(init_centes, init_labels, data, n_cluster):
    centers = init_centes
    labels = init_labels
    while True:
            labels = kmeans_predict_labels(data, centers)
            new_centers = kmeans_update_centers(data, labels, n_cluster)
            if kmeans_has_converged(centers, new_centers):
                break
            centers = new_centers
            #kmeans_visualize(data, centers, labels, n_cluster, 'K-Means = ')
    return (centers, labels)
n_cluster = 4
init_centers = kmeans_init_centers(data, n_cluster)
init_labels = np.zeros(data.shape[0])
centers, labels = kmeans(init_centers, init_labels, data, n_cluster)

# in danh sách
for i in range(n_cluster):
    X = data[labels == i]

    print('Tâm của cụm ',i,centers[i])
    print('Các phần tử của cụm',i)
    print(X)

# kiểm tra chất lượng cụm 0
distA = np.mean(cdist(data[labels == 0], data[labels == 0]))
distA1 = np.mean(cdist(data[labels == 1], data[labels == 0]))
distA2 = np.mean(cdist(data[labels == 2], data[labels == 0]))
distA3 = np.mean(cdist(data[labels == 3], data[labels == 0]))
distB = min(distA1, distA2, distA3)
test = (distB - distA)/max(distA,distB)
print('---------------------')
print('Kiểm tra độ tốt cụm 0')
print('Số phần tử cụm 0: ', len(list(data[labels == 0])))
print(distA)
print(distB)
print('k = ',test) # khi 'test' dần tiến đến 1 có nghĩa là chất lượng phân cụm tốt, -1 <= test <= 1
