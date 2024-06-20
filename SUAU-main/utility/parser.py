"""
@Time    : 2024/3/11 08:21
@Author  : YuZhang
@File    : parser.py
"""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run Reproduct")
    parse.add_argument('--seed', type=int, default=2024, help='random seed')
    parse.add_argument('--gpu', type=int, default=0, help='indicates which gpu to use')
    parse.add_argument('--cuda', type=bool, default=True, help='use gpu or not')
    parse.add_argument('--log', type=str, default='None', help='init log file name')
    parse.add_argument('--alpha', type=float, default=2, help='align alpha')
    parse.add_argument('--t', type=float, default=2.0, help='')
    parse.add_argument('--dataset_path', type=str, default='./dataset/', help='dataset path')
    parse.add_argument('--dataset', type=str, default='douban-book', help='dataset')
    parse.add_argument('--dataset_type', type=str, default='.txt', help='dataset type')
    parse.add_argument('--top_K', type=str, default='[10, 20, 30, 40, 50]', help='rating number')
    parse.add_argument('--train_epoch', type=int, default=300, help='train epoch')
    parse.add_argument('--early_stop', type=int, default=10, help='if no more higher recall, stop')
    parse.add_argument('--embedding_size', type=int, default=64, help='')
    parse.add_argument('--train_batch_size', type=int, default=2048, help='')
    parse.add_argument('--test_batch_size', type=int, default=200, help='')
    parse.add_argument('--learn_rate', type=float, default=0.001, help='')
    parse.add_argument('--GCN_layer', type=int, default=1, help='')
    parse.add_argument('--test_frequency', type=int, default=1, help='')
    parse.add_argument('--gamma', type=float, default=2.5, help='control uniform')
    parse.add_argument('--beta', type=float, default=0.1, help='control similar item')
    parse.add_argument('--seed_start', action='store_true', help='When you use type=bool and pass a value, the parser interprets any provided value as True, and if the argument is not provided, it defaults to False. ')
    parse.add_argument('--sim_item', type=int, default=2, help='similar item number')
    parse.add_argument('--sparsity_test', type=int, default=0, help='whether inter sparsity test')
    return parse.parse_args()