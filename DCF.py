# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la
from collections import defaultdict
from math import log


class DCF:
    code_len = 4
    alpha = 2
    beta = 2
    threshold = 1e-4
    max_iter = 50
    user = {}
    item = {}
    id2user = {}
    id2item = {}
    u_i_r = defaultdict(dict)
    i_u_r = defaultdict(dict)
    minVal = 1
    maxVal = 5

    train_data_path = '../data/ML100k_new/ML100k_train_new.txt'
    valid_data_path = '../data/ML100k_new/ML100k_valid_new.txt'
    test_data_path = '../data/ML100k_new/ML100k_test_new.txt'


    def init_model(self):
        self.generate_hash()
        self.rating_matrix, self.rating_matrix_bin, self.globalmean = self.get_rating_matrix()
        self.B = np.sign(np.array(np.random.randn(len(self.user), self.code_len) / (self.code_len ** 0.5)))
        self.D = np.sign(np.array(np.random.randn(len(self.item), self.code_len) / (self.code_len ** 0.5)))
        self.X = self.update_X_Y(self.B)
        self.Y = self.update_X_Y(self.D)
        self.loss, self.last_delta_loss = 0.0, 0.0

    def trainSet(self):
        with open(self.train_data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                r = 2 * self.code_len * (float(int(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01) - self.code_len
                yield (int(u), int(i), float(r))

    def valid_test_Set(self, path):
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                # r = 2 * self.code_len * (float(int(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01) - self.code_len
                yield (int(u), int(i), float(r))

    def generate_hash(self):
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            self.u_i_r[user_id][item_id] = rating
            self.i_u_r[item_id][user_id] = rating
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id

    def get_rating_matrix(self):
        rating_matrix = np.zeros((len(self.user), len(self.item)))   # (943, 1596)
        globalmean = 0.0
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            globalmean += rating
            rating_matrix[self.user[user_id]][self.item[item_id]] = int(rating)
        rating_matrix_bin = (rating_matrix > 0).astype('int')
        globalmean = globalmean / (len(self.user))
        return rating_matrix, rating_matrix_bin, globalmean

    def calDCG_k(self, dictdata, k):
        nDCG = []
        for key in dictdata.keys():
            listdata = dictdata[key]
            real_value_list = sorted(listdata, key=lambda x: x[1], reverse=True)
            idcg = 0.0
            predict_value_list = sorted(listdata, key=lambda x: x[0], reverse=True)
            dcg = 0.0
            if len(listdata) >= k:
                for i in range(k):
                    idcg += (pow(2, real_value_list[i][1]) - 1) / (log(i + 2, 2))
                    dcg += (pow(2, predict_value_list[i][1]) - 1) / (log(i + 2, 2))
                if(idcg != 0):
                    nDCG.append(float(dcg / idcg))
            else:
                continue
        ave_ndcg = np.mean(nDCG)
        # print(nDCG)
        return ave_ndcg

    def train_model(self):
        iteration = 0
        last_loss = 0.0
        while(iteration < self.max_iter):
            master_flag = 0
            iteration += 1
            # print('update B')
            for u in range(len(self.user)):
                while(1):
                    flag = 0
                    bu = self.B[u, :]
                    # print(self.B[:, u])
                    for k in range(self.code_len):
                        dk = self.D[:, k]
                        buk_hat = np.sum((self.rating_matrix[u, :] - np.dot(self.D, bu.T)) * dk * self.rating_matrix_bin[u, :])\
                                  + self.alpha * self.X[u, k] + len(self.u_i_r[self.id2user[u]]) * bu[k]
                        buk_new = np.sign(self.K(buk_hat, bu[k]))
                        if(bu[k] != buk_new):
                            flag = 1
                            bu[k] = buk_new
                    if(flag == 0):
                        break
                    self.B[u, :] = bu
                    master_flag = 1
            # print('update D')
            for i in range(len(self.item)):
                while(1):
                    flag = 0
                    di = self.D[i, :]
                    for k in range(self.code_len):
                        bk = self.B[:, k]
                        dik_hat = np.sum((self.rating_matrix[:, i] - np.dot(self.B, di.T)) * bk * self.rating_matrix_bin[:, i])\
                                  + self.beta * self.Y[i, k] + len(self.i_u_r[self.id2item[i]]) * di[k]
                        dik_new = np.sign(self.K(dik_hat, di[k]))
                        if(di[k] != dik_new):
                            flag = 1
                            di[k] = dik_new
                    if(flag == 0):
                        break
                    self.D[i, :] = di
                    master_flag = 1
            self.X = self.update_X_Y(self.B)
            self.Y = self.update_X_Y(self.D)
            self.loss = np.sum((self.rating_matrix - np.dot(self.B, (self.D).T)) ** 2 * self.rating_matrix_bin) \
                        - 2 * self.alpha * np.trace(np.dot((self.B).T, self.X)) - 2 * self.beta * np.trace(np.dot((self.D).T, self.Y))
            valid_ndcg_10 = self.valid_test_model(self.valid_data_path)
            delta_loss = self.loss - last_loss
            print('iteration %d: loss = %.5f, delta_loss = %.5f valid_NDCG@10=%.5f' %
                  (iteration, self.loss, delta_loss, valid_ndcg_10))
            if(master_flag == 0):
                break
            if(abs(delta_loss) < self.threshold or abs(delta_loss) == abs(self.last_delta_loss)):
                break
            self.last_delta_loss = delta_loss
            last_loss = self.loss
        test_ndcg_10 = self.valid_test_model(self.test_data_path)
        print('test NGCD@10 = %.5f' %(test_ndcg_10))

    def K(self, x, y):
        return x if x != 0 else y

    def containUser(self, user_id):
        if user_id in self.user:
            return True
        else:
            return False

    def containItem(self, item_id):
        if item_id in self.item:
            return True
        else:
            return False

    def valid_test_model(self, path):
        pre_true_dict = defaultdict(list)
        for index, line in enumerate(self.valid_test_Set(path)):
            user_id, item_id, rating = line
            if(self.containUser(user_id) and self.containItem(item_id)):
                bu = self.B[self.user[user_id], :]
                di = self.D[self.item[item_id], :]
                pre = np.dot(bu, di)
            elif(self.containUser(user_id) and not self.containItem(item_id)):
                pre = sum(self.u_i_r[user_id].values()) / float(len(self.u_i_r[user_id]))
            elif(not self.containUser(user_id) and self.containItem(item_id)):
                pre = sum(self.i_u_r[item_id].values()) / float(len(self.i_u_r[item_id]))
            else:
                pre = self.globalmean
            pre_true_dict[user_id].append([pre, rating])
        ndcg_10 = self.calDCG_k(pre_true_dict, 10)
        return ndcg_10

    def gram_schmidt(self, X):
        Q, R = la.qr(X)
        return Q

    def update_X_Y(self, Z):
        temp_Z = Z.T
        Z_bar = temp_Z - temp_Z.mean(axis=1)[:, np.newaxis]  #(943, code_len)
        # print(Z_bar.shape)
        SVD_Z = la.svd(Z_bar, full_matrices=False)
        Q = SVD_Z[2].T
        # print(Q.shape)
        if Q.shape[1] < self.code_len:
            Q = self.gram_schmidt(np.c_[Q, np.ones((Q.shape[0], self.code_len - Q.shape[1]))])
        P = la.svd(np.dot(Z_bar, Z_bar.T))[0]
        Z_new = np.sqrt(temp_Z.shape[1]) * np.dot(P, Q.T)
        return Z_new.T

    def main(self):
        self.init_model()
        self.train_model()



if __name__ == '__main__':
    dcf = DCF()
    dcf.main()
