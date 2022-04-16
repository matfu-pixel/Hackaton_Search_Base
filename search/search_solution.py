import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg 
from .search import Base
import utils.util_funcs as uf
from sklearn.cluster import KMeans

class SearchSolution(Base):
    def __init__(self, data_file='./data/train_data.pickle', 
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url

    def set_base_from_pickle(self):

        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data') 
            gdown.download(self.data_url, self.data_file, quiet=False) 

        with open(self.data_file, 'rb') as f:
            data = pickle.load(f) 
    
        self.reg_matrix = [None] * len(data['reg'])
        self.ids = {} 
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)        
        self.pass_dict = data['pass']

        print('Init start...')
        K = 2
        self.kmeans = KMeans(n_clusters=K, random_state=42, verbose=True, n_init=10, max_iter=100).fit(self.reg_matrix)
        print('Myaso end')
        self.ans = [[] for _ in range(K)]
        for i, k in enumerate(self.kmeans.labels_):
            self.ans[k].append(i)
    
    def cal_base_speed(self, base_speed_path='./base_speed.pickle') -> float:
        samples = cfg.samples 
        N, C, C_time, T_base = 0, 0, 0, 0
        for i, tup in enumerate(tqdm(self.pass_dict.items(), total=samples)):

            idx, passes = tup
            for q  in passes:
                t0 = time.time()
                c_output = self.search(query=q) 
                t1 = time.time()
                T_base += (t1 - t0)

                C_set = [True for tup in c_output if tup[0] == idx]
                if len(C_set):
                    C += 1
                    C_time += (t1 - t0) 
                N += 1

            if i > samples:
                break

        base_speed = T_base / N
        with open(base_speed_path, 'wb') as f:
            pickle.dump(base_speed, f)

    def search(self, query: np.array) -> List[Tuple]:
        tmp = []
        for it in self.ans[self.kmeans.predict([query])[0]]:
            tmp.append((it, self.reg_matrix[it] @ query))
        return tmp
    

    def insert_base(self, feature: np.array) -> None:
        pass
        self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0) 

    def cos_sim(self, query: np.array) -> np.array:
        pass

