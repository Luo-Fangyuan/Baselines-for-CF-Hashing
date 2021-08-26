# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la
from collections import defaultdict
from math import log


class PPH:
    code_len = 4
    lamda = 0.01
    lr = 0.01
    threshold = 1e-4
    max_iter = 2000
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
        self.globalmean = self.generate_hash()
        self.B = np.array(np.random.rand(len(self.user), self.code_len))
        self.D = np.array(np.random.rand(len(self.item), self.code_len))
        self.loss, self.last_delta_loss = 0.0, 0.0

    def trainSet(self):
        with open(self.train_data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                # r = 2 * self.code_len * (float(int(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01) - self.code_len
                yield (int(u), int(i), float(r))

    def valid_test_Set(self, path):
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(',')
                # r = 2 * self.code_len * (float(int(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01) - self.code_len
                yield (int(u), int(i), float(r))

    def generate_hash(self):
        globalmean = 0.0
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            globalmean += rating
            self.u_i_r[user_id][item_id] = rating
            self.i_u_r[item_id][user_id] = rating
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id
        globalmean = globalmean / (len(self.user))
        return globalmean

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
        return ave_ndcg

    def train_model(self):
        iteration = 0
        last_loss = 0.0
        while(iteration < self.max_iter):
            iteration += 1
            self.loss = 0.0
            for index, line in enumerate(self.trainSet()):
                user_id, item_id, rating = line
                bu = self.B[self.user[user_id], :]
                di = self.D[self.item[item_id], :]
                error = rating - 0.5 * self.maxVal - np.dot(bu, di)
                self.loss += error ** 2
                self.B[self.user[user_id], :] += self.lr * error * di - 2 * self.lamda * bu * (np.linalg.norm(bu) ** 2 - 0.5 * self.maxVal)
                self.D[self.item[item_id], :] += self.lr * error * bu - 2 * self.lamda * di * (np.linalg.norm(di) ** 2 - 0.5 * self.maxVal)
            for u in range(len(self.user)):
                self.loss += (np.linalg.norm(self.B[u, :]) ** 2 - 0.5 * self.maxVal) ** 2
            for i in range(len(self.item)):
                self.loss += (np.linalg.norm(self.D[i, :]) ** 2 - 0.5 * self.maxVal) ** 2
            self.loss = self.lamda * self.loss
            
            valid_ndcg_10 = self.valid_test_model(self.valid_data_path)
            delta_loss = self.loss - last_loss
            print('iteration %d: loss = %.5f, delta_loss = %.5f valid_NDCG@10=%.5f' %(iteration, self.loss, delta_loss, valid_ndcg_10))
            last_loss = self.loss
            if(abs(delta_loss) < self.threshold or self.loss > last_loss):
                break
            self.last_delta_loss = delta_loss
        test_ndcg_10 = self.valid_test_model(self.test_data_path)
        print('test NGCD@10 = %.5f' %(test_ndcg_10))

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
                bu = np.sign(self.B[self.user[user_id], :])
                di = self.D[self.item[item_id], :]
                mean = np.mean(di)
                std = np.std(di)
                if(np.linalg.norm(di) < mean - std):
                    pre = np.dot(bu, np.sign(di)) - 2
                if(np.linalg.norm(di) >= mean - std and np.linalg.norm(di) <= mean + std):
                    pre = np.dot(bu, np.sign(di))
                if(np.linalg.norm(di) > mean + std):
                    pre = pre = np.dot(bu, np.sign(di)) + 2
            elif(self.containUser(user_id) and not self.containItem(item_id)):
                pre = sum(self.u_i_r[user_id].values()) / float(len(self.u_i_r[user_id]))
            elif(not self.containUser(user_id) and self.containItem(item_id)):
                pre = sum(self.i_u_r[item_id].values()) / float(len(self.i_u_r[item_id]))
            else:
                pre = self.globalmean
            pre_true_dict[user_id].append([pre, rating])
        ndcg_10 = self.calDCG_k(pre_true_dict, 10)
        return ndcg_10


    def main(self):
        self.init_model()
        self.train_model()



if __name__ == '__main__':
    pph = PPH()
    pph.main()
