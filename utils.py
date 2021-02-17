import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F


def data_2_init_cond(H, M, L, lmd):
	batch_dim = np.shape(H)[0]
	Nc = 3
	alpha = (np.sqrt(Nc) * torch.from_numpy(H.astype(np.float32)) * L /
			 (2 * np.pi))
	beta = torch.from_numpy(M.astype(np.float32)) * L**3 * np.pi / np.sqrt(Nc)
	init_cond = torch.zeros((batch_dim, 2))
	init_cond[:, 0] = (alpha * np.exp(-1) + beta * np.exp(-3) -
					   (lmd * alpha**3 / 2) * 1 * np.exp(-3))
	init_cond[:, 1] = (-alpha * np.exp(-1) - 3 * beta * np.exp(-3) +
					   (3 / 2) * (lmd * alpha**3) * np.exp(-3) - 0.5 *
					   (alpha**3) * lmd * np.exp(-3))
	return init_cond

def create_batch_data(pos_H, pos_M, neg_H, neg_M, batch_dim):
	data_H = np.zeros(batch_dim)
	data_M = np.zeros(batch_dim)
	label = np.ones(batch_dim)  # 1 = False, 0 = True
	pos_data_num = int(batch_dim / 2)
	pos_idx = np.floor(np.random.uniform(0,
										 np.shape(pos_H)[0],
										 pos_data_num)).astype(int)
	neg_idx = np.floor(
		np.random.uniform(0,
						  np.shape(pos_H)[0],
						  batch_dim - pos_data_num)).astype(int)
	data_H[:pos_data_num] = pos_H[pos_idx]
	data_H[pos_data_num:] = neg_H[neg_idx]
	data_M[:pos_data_num] = pos_M[pos_idx]
	data_M[pos_data_num:] = neg_M[neg_idx]
	label[:pos_data_num] = np.zeros(pos_data_num)
	return data_H, data_M, label

def tanaka(x, epsilon=None, tanh=None, dev=.01):
	left = -arctan_scaled(x, center=-epsilon, scale=1 / dev, tanh=tanh)
	right = arctan_scaled(x, center=epsilon, scale=1 / dev, tanh=tanh)
	lf = left + right
	return (lf + 2) / 2

def arctan_scaled(x, center=None, scale=10, tanh=None):
	y = scale * (x - center)
	y = tanh(y)
	return y









