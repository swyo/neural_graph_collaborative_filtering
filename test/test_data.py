#!/usr/bin/env python
import sys
import unittest
from typing import List, Dict

from swyo.utils import Config

sys.path.append('../NGCF')
from utility.load_data import Data


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opt = Config('../config/ngcf.json')

    def test_init(self):
        def assert_not_empty(dic: Dict[int, List[int]]):
            for key, val in dic.items():
                self.assertIsInstance(key, int)
                self.assertIsInstance(val, list)
                self.assertTrue(val != [])
        generator = Data(path=self.opt.data_path + self.opt.dataset, batch_size=self.opt.batch_size)
        self.assertEqual(generator.R.shape, (generator.n_users, generator.n_items))
        self.assertEqual(generator.R.nnz, generator.n_train)
        assert_not_empty(generator.train_items)
        assert_not_empty(generator.test_set)
        distinct_n_users_train = len(generator.train_items)
        distinct_n_users_test = len(generator.test_set)
        self.assertLessEqual(distinct_n_users_train, generator.n_users)
        self.assertLessEqual(distinct_n_users_test, generator.n_users)
        if distinct_n_users_train == distinct_n_users_test and distinct_n_users_train == generator.n_users:
            print("Train and Test users are sampling for each users.")

    def test_get_adj_mat(self):
        """Create adjacent matrices."""
        generator = Data(path=self.opt.data_path + self.opt.dataset, batch_size=self.opt.batch_size)
        adj, norm_adj, mean_adj = generator.get_adj_mat()
        m, n = generator.R.shape
        self.assertEqual(adj.shape, (m + n, m + n))
        self.assertEqual(m + n, norm_adj.sum())

    def test_negative_pool(self):
        """Create negative pool by 100 sampling negative items for each users.

        Description:
            Sampling negative items with replacement, so duplicates exist.
        """
        generator = Data(path=self.opt.data_path + self.opt.dataset, batch_size=self.opt.batch_size)
        self.assertEqual(generator.neg_pools, {})
        generator.negative_pool()
        self.assertEqual(len(generator.neg_pools), generator.R.shape[0])
        for uid, neg_samples in generator.neg_pools.items():
            self.assertEqual(len(neg_samples), 100)

    def test_sparsity_split(self):
        """Split users into 4 clusters(folds) by balancing total numbers of ratings."""
        generator = Data(path=self.opt.data_path + self.opt.dataset, batch_size=self.opt.batch_size)
        split_uids, split_states = generator.get_sparsity_split()
        self.assertEqual(sum([len(uids) for uids in split_uids]), generator.R.shape[0])
        for uids, state in zip(split_uids, split_states):
            n_ratings = sum([len(generator.train_items[uid]) + len(generator.test_set[uid]) for uid in uids])
            true = int(state[state.find('#all rates') + 11:][1: -1])
            self.assertEqual(n_ratings, true)

    def test_sample(self):
        """Take random samples for given batch_size from train examples.

        Description:
            If batch_size is larger than n_trains, do sampling with replacement.
        """
        generator = Data(path=self.opt.data_path + self.opt.dataset, batch_size=self.opt.batch_size)
        users, pos_items, neg_items = generator.sample()
        self.assertTrue(len(users) == len(pos_items) == len(neg_items) == generator.batch_size)
