"""
@Time    : 2024/3/11 08:11
@Author  : YuZhang
@File    : SUAU.py
"""
import torch
from torch import nn
import utility.losses as losses
import utility.trainer as trainer
import utility.tools as tools
import torch.nn.functional as F

class SUAU(nn.Module):
    def __init__(self, args, dataset, device):
        super(SUAU, self).__init__()
        self.model_name = "UniqueAU"
        self.args = args
        self.sim_item = self.args.sim_item
        self.dataset = dataset
        self.device = device
        self.gamma = float(args.gamma)
        self.t = float(self.args.t)
        self.alpha = float(self.args.alpha)
        self.activation = nn.Sigmoid()
        self.beta = args.beta

        self.user_embedding = nn.Embedding(num_embeddings=self.dataset.num_users,
                                           embedding_dim=int(self.args.embedding_size))
        self.item_embedding = nn.Embedding(num_embeddings=self.dataset.num_items,
                                           embedding_dim=int(self.args.embedding_size))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.MLP = nn.Linear(int(self.args.embedding_size), int(self.args.embedding_size))

        self.adj_mat = self.dataset.sparse_adjacency_matrix()

        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)


    def find_similar_item(self):
        user_counts = self.dataset.train_mat.getnnz(axis=1)

        all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        self.dataset.item_counts = self.dataset.train_mat.getnnz(axis=0)

        self.dataset.item_users_dict = {}

        for user, item in zip(self.dataset.train_mat.row, self.dataset.train_mat.col):
            if item not in self.dataset.item_users_dict:
                self.dataset.item_users_dict[item] = set()
            self.dataset.item_users_dict[item].add(user)

        self.dataset.similar_item = {}
        for item in range(self.dataset.num_items):
            users = self.dataset.item_users_dict[item]
            users = list(users)
            min_u = [users[0]]
            min_value = user_counts[users[0]]
            for u in users:
                if min_value > user_counts[u] and user_counts[u] > 1:
                    min_u = [u]
                    min_value = user_counts[u]
                elif min_value == user_counts[u]:
                    min_u.append(u)
            if len(min_u) == 1:
                self.dataset.similar_item[item] = set(self.dataset.train_dict[min_u[0]])
            else:
                self.dataset.similar_item[item] = set()
                for u in min_u:
                    self.dataset.similar_item[item].update(self.dataset.train_dict[u])

        self.dataset.item_indices = []
        self.dataset.similar_item_indices = []

        for item, similar_items in self.dataset.similar_item.items():

            random_similar_items = torch.tensor(list(similar_items - {item}))

            embedding1, embedding_group = all_item_gcn_embed[item], all_item_gcn_embed[random_similar_items]
            random_similar_items = self.rank_similarity(embedding1, embedding_group, random_similar_items)
            random_similar_items = random_similar_items[0]

            t = self.sim_item

            if len(random_similar_items) < t:
                self.dataset.similar_item_indices.extend(random_similar_items)
                self.dataset.item_indices.extend([item] * len(random_similar_items))
            else:
                random_similar_item = random_similar_items[:t]
                # for u in random_similar_user:
                self.dataset.similar_item_indices.extend(random_similar_item)
                self.dataset.item_indices.extend([item] * t)

        return self.dataset.item_indices, self.dataset.similar_item_indices


    def cosine_similarity(self, embedding1, embedding2):
        similarity = F.cosine_similarity(embedding1, embedding2.unsqueeze(0), dim=-1)
        return similarity

    def rank_similarity(self, embedding, group_embeddings, original_indices):
        similarities = self.cosine_similarity(embedding, group_embeddings)
        sorted_indices = torch.argsort(similarities, descending=True)
        sorted_original_indices = original_indices[sorted_indices]
        return sorted_original_indices

    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = []
        for layer in range(int(self.args.GCN_layer)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        return user_emb, item_emb

    def aggregate_info(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = []
        for layer in range(int(3)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        return user_emb, item_emb

    def forward(self, user, positive, flag='main'):
        all_user_gcn_embed, all_item_gcn_embed = self.aggregate()
        # all_user_gcn_embed_info, all_item_gcn_embed_info = self.aggregate_info()
        if flag == 'main':
            user_gcn_embed, item_gcn_embed = all_user_gcn_embed[user.long()], all_item_gcn_embed[positive.long()]
            unique_user, user_count = torch.unique(user, return_counts=True)
            unique_positive, positive_count = torch.unique(positive, return_counts=True)
            user_gcn_embed_u, pos_gcn_embed_u = all_user_gcn_embed[unique_user.long()], all_item_gcn_embed[unique_positive.long()]
            align_loss = losses.get_align_loss(user_gcn_embed, item_gcn_embed, alpha=self.alpha)
            uniform_loss = self.gamma * (losses.get_uniform_loss(user_gcn_embed_u, t=self.t) +
                                         losses.get_uniform_loss(pos_gcn_embed_u, t=self.t)) / 2

            # user_gcn_embed_info, item_gcn_embed_info = all_user_gcn_embed_info[user.long()], all_item_gcn_embed_info[positive.long()]
            # ssl_loss = 0.00001 * losses.InfoNCE(user_gcn_embed_info, item_gcn_embed_info, 0.2)

            loss_list = [align_loss, uniform_loss]
        if flag == 'item':
            # user in here means similar_item
            item1_gcn_embed, item2_gcn_embed = all_item_gcn_embed[user.long()], all_item_gcn_embed[positive.long()]

            unique_item1, item1_count = torch.unique(user, return_counts=True)
            unique_item2, item2_count = torch.unique(positive, return_counts=True)

            item1_gcn_embed_i, item2_gcn_embed_i = all_item_gcn_embed[unique_item1.long()], all_item_gcn_embed[unique_item2.long()]

            align_loss = losses.get_align_loss(item1_gcn_embed, item2_gcn_embed, alpha=self.alpha) * self.beta
            uniform_loss = self.gamma * (losses.get_uniform_loss(item1_gcn_embed_i, t=self.t) +
                                         losses.get_uniform_loss(item2_gcn_embed_i, t=self.t)) / 2 * self.beta
            loss_list = [align_loss, uniform_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_gcn_embed, all_item_gcn_embed = self.aggregate()
        rating = self.activation(torch.matmul(all_user_gcn_embed[user.long()], all_item_gcn_embed.t()))
        return rating

class Trainer():
    def __init__(self, args, dataset, device, logger):
        self.model = SUAU(args, dataset, device)
        self.dataset = dataset
        self.device = device
        self.logger = logger
        self.args = args
    def train(self):
        trainer.training(self.model, self.args, self.dataset, self.device, self.logger)