# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la
from collections import defaultdict
from math import log
import random
from sklearn.metrics import roc_auc_score



class DBPR:
    code_len = 8
    alpha = 0.01
    beta = 0.01
    threshold = 1e-4
    max_iter = 50
    user = {}
    item = {}
    id2user = {}
    id2item = {}

    train_data_path = '../data/ML100k_new/ML100k_train_new.txt'
    valid_data_path = '../data/ML100k_new/ML100k_valid_new.txt'
    test_data_path = '../data/ML100k_new/ML100k_test_new.txt'

    def init_model(self):
        self.generate_hash()
        self.rating_matrix, self.z_pos, self.z_neg = self.generate_rating_matrix()
        
    def init_parameter(self):
        self.B = np.sign(np.array(np.random.randn(len(self.user), self.code_len) / (self.code_len ** 0.5)))
        self.D = np.sign(np.array(np.random.randn(len(self.item), self.code_len) / (self.code_len ** 0.5)))
        self.X = self.update_X_Y(self.B)
        self.Y = self.update_X_Y(self.D)
        self.D_pos_bar = np.zeros(len(self.user))
        self.D_neg_bar = np.zeros(len(self.user))
        self.loss, self.last_delta_loss = 0.0, 0.0

    def trainSet(self):
        with open(self.train_data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                yield (int(u), int(i), float(r))

    def validSet(self):
        with open(self.valid_data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                yield (int(u), int(i), float(r))

    def testSet(self):
        with open(self.test_data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                yield (int(u), int(i), float(r))

    def generate_hash(self):
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id

    def generate_rating_matrix(self):
        rating_matrix = np.zeros((len(self.user), len(self.item)))
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            rating_matrix[self.user[user_id], self.item[item_id]] = 1
        z_pos = 1 / np.sum(rating_matrix, axis = 1)
        z_neg = 1 / np.sum(1 - rating_matrix, axis = 1)
        return rating_matrix, z_pos, z_neg

    def train_model(self):
        iteration = 0
        last_loss = 0.0
        while(iteration < self.max_iter):
            master_flag = 0
            iteration += 1
#             print('update B')
            for u in range(len(self.user)):
                du_pos_bar = self.z_pos[u] * np.dot(self.rating_matrix[u, :], self.D)
                du_neg_bar = self.z_neg[u] * np.dot(1 - self.rating_matrix[u, :], self.D)
                bu = self.B[u, :]
                xu = self.X[u, :]
                self.D_pos_bar[u] = np.dot(bu, du_pos_bar)
                self.D_neg_bar[u] = np.dot(bu, du_neg_bar)
                while(1):
                    flag = 0
                    for k in range(self.code_len):
                        dkd_pos_bar = self.z_pos[u] * np.dot(self.rating_matrix[u, :] * self.D[:, k].T, self.D)
                        dkd_neg_bar = self.z_neg[u] * np.dot((1 - self.rating_matrix[u, :]) * self.D[:, k].T, self.D)
                        buk_hat = (- np.dot(bu, du_neg_bar)+ bu[k] * du_neg_bar[k]) * du_pos_bar[k] - (np.dot(bu, du_pos_bar) - bu[k] * du_pos_bar[k]) * du_neg_bar[k] + (np.dot(bu, dkd_pos_bar) - 2 * self.code_len * du_pos_bar[k] - bu[k]) + (np.dot(bu, dkd_neg_bar) + 2 * self.code_len * du_neg_bar[k] - bu[k]) - self.alpha * xu[k]
                        buk_new = np.sign(self.K(-buk_hat, bu[k]))
                        if(bu[k] != buk_new):
                            flag = 1
                            bu[k] = buk_new
                    if(flag == 0):
                        break
                    self.B[u, :] = bu
                    master_flag = 1
#             print('update D')
            z_pos_bar = np.dot(self.z_pos, self.rating_matrix)
            z_neg_bar = np.dot(self.z_neg, 1 - self.rating_matrix)
            for i in range(len(self.item)):
                while(1):
                    flag = 0
                    di = self.D[i, :]
                    yi = self.Y[i, :]
                    for k in range(self.code_len):
                        bik_pos_bar = np.dot(self.z_pos * self.B[:, k], self.rating_matrix[:, i])
                        bik_neg_bar = np.dot(self.z_neg * self.B[:, k], 1 - self.rating_matrix[:, i])
                        first_part = np.dot(self.B[:, k] * (1 - self.rating_matrix[:, i]) * self.z_neg, np.dot(self.B, di.T) - self.D_pos_bar)
                        second_part = np.dot(self.B[:, k] * self.rating_matrix[:, i] * self.z_pos, np.dot(self.B, di.T) - self.D_neg_bar)
                        dik_hat = first_part + second_part - (z_pos_bar[i] + z_neg_bar[i]) * di[k] + 2 * self.code_len * (bik_neg_bar - bik_pos_bar) - self.beta * yi[k]
                        # print(dik_hat)
                        dik_new = np.sign(self.K(-dik_hat, di[k]))
                        if(di[k] != dik_new):
                            flag = 1
                            di[k] = dik_new
                    if(flag == 0):
                        break
                    self.D[i, :] = di
                    master_flag = 1
            self.X = self.update_X_Y(self.B)
            self.Y = self.update_X_Y(self.D)
            self.loss = np.sum((2 * self.code_len * self.rating_matrix - np.dot(self.B, (self.D).T)) ** 2 * self.rating_matrix) - 2 * self.alpha * np.trace(np.dot((self.B).T, self.X)) - 2 * self.beta * np.trace(np.dot((self.D).T, self.Y))
            valid_auc = self.valid_model()
            delta_loss = self.loss - last_loss
            print('iteration %d: loss = %.5f, delta_loss = %.5f valid_AUC=%.5f' %
                  (iteration, self.loss, delta_loss, valid_auc))
            if(master_flag == 0):
                break
            if(abs(delta_loss) < self.threshold or abs(delta_loss) == abs(self.last_delta_loss)):
                break
            self.last_delta_loss = delta_loss
            last_loss = self.loss
        test_auc = self.test_model()
        print('test AUC = %.5f' %(test_auc))

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

    def valid_model(self):
        valid_matrix = np.zeros((len(self.user), len(self.item)))
        for index, line in enumerate(self.validSet()):
            user_id, item_id, rating = line
            if(self.containUser(user_id) and self.containItem(item_id)):
                valid_matrix[self.user[user_id], self.item[item_id]] = 1
        predict_matrix = np.dot(self.B, self.D.T) * valid_matrix
        predict = predict_matrix.reshape(-1)
        valid = valid_matrix.reshape(-1)
        auc_score = roc_auc_score(valid, predict)
        return auc_score

    def test_model(self):
        test_matrix = np.zeros((len(self.user), len(self.item)))
        for index, line in enumerate(self.testSet()):
            user_id, item_id, rating = line
            if(self.containUser(user_id) and self.containItem(item_id)):
                test_matrix[self.user[user_id], self.item[item_id]] = 1
        predict_matrix = np.dot(self.B, self.D.T) * test_matrix
        predict = predict_matrix.reshape(-1)
        test = test_matrix.reshape(-1)
        auc_score = roc_auc_score(test, predict)
        return auc_score

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
        self.init_parameter()
        self.train_model()



if __name__ == '__main__':
    dbpr = DBPR()
    dbpr.main()
