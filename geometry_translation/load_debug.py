import numpy as np

if __name__ == '__main__':
    data = '../data/tdrive_sample_s5/learning/train/0_0_direction.npy'
    ab = np.load(data)
    print(ab.shape)
