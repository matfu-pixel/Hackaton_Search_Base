import numpy as np
from .search import Base
from typing import List, Tuple
import faiss
import pickle, os, time, gdown
import numpy as np
from tqdm import tqdm
from config import Config as cfg 

 
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
 
        dim = 512
        self.index = faiss.index_factory(dim, "IVF1000,Flat", faiss.METRIC_INNER_PRODUCT)
        print(self.index.is_trained)
        self.index.train(self.reg_matrix.astype('float32'))
 
        print(self.index.is_trained)
        print(self.index.ntotal)
        self.index.add(self.reg_matrix.astype('float32'))
        print(self.index.ntotal)
 
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
        print(f"Base Line Speed: {base_speed}")
        print(f"Base Line Accuracy: {C / N * 100}")
        # with open(base_speed_path, 'wb') as f:
        #     pickle.dump(base_speed, f)
 
    def search(self, query: np.array) -> List[Tuple]:
        #print(query)
        self.index.nprobe = 100
        tmp = np.zeros((1, len(query)), dtype='float32')
        tmp[0] = np.array(query)
        D, I = self.index.search(tmp, 1000)
        return [(I[0][i], D[0][i]) for i in range(len(D))]
 
    def insert_base(self, feature: np.array) -> None:
        self.index.add(feature)

    def cos_sim():
        pass
