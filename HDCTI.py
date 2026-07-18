#coding:utf8
import json
import os
from util.gpu import configure_cuda_environment
configure_cuda_environment()

import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior() #禁用2.0版本的方法
from base.herbRecommender import herbRecommender
from scipy.sparse import coo_matrix,hstack
#import tensorflow as tf
import numpy as np
from math import sqrt
import pandas as pds
import scipy.sparse as sp
# from spektral.layers import GraphAttention
from time import strftime,localtime,time
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout

import networkx as nx
from util.graph import bipartite_pagerank
from util.model_components import (
    EarlyStoppingTracker,
    build_regularization_loss,
    context_interaction_pair_scores,
    resolve_context_terms,
    resolve_context_mask_training,
    resolve_counterfactual_context,
    resolve_early_stopping,
    resolve_global_token_attention,
    resolve_hyperedge_attention,
    resolve_herb_context_attention,
    resolve_pair_decoder,
    resolve_support_router,
    target_conditioned_herb_contexts,
)
from util.support_router import (
    inverse_softplus,
    monotonic_context_gate,
    softplus,
)
from util.hyperedge_attention import (
    hyperedge_specificity_prior,
    ordered_incidence_ids,
)
from util.global_token_attention import (
    global_token_attention_complexity,
    summarize_global_token_attention,
)
from sklearn.metrics import average_precision_score, roc_auc_score
# tf.compat.v1.set_random_seed(4321)
from util.io import FileIO

TARGET_HERB_ATTENTION_MODES = {
    'target_attention',
    'target_residual_attention',
}

class HDCTI(herbRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(HDCTI, self).__init__(conf,trainingSet,testSet,fold)


    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix，对称矩阵
            if int(pair[2])!=0:
                row += [self.data.compound[pair[0]]]
                col += [self.data.protein[pair[1]]]
                entries += [1]
        # u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_diseases),dtype=np.float32)
        u_i_adj = coo_matrix((entries,(row,col)), shape=(self.num_compounds, self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            # symmetric matrix
            #x=random.randint(0,1008)
            #y=random.randint(0,1192)
            #row += [x]
            #col += [y]
            row += [self.data.herb[pair[0]]]
            col += [self.data.compound[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_compounds),dtype=np.float32)
        return u_i_adj

    def buildcpAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.cpassociation:
            # symmetric matrix
            row += [self.data.compound[pair[0]]]
            col += [self.data.protein[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_compounds,self.num_proteins),dtype=np.float32)
        return u_i_adj

    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            # symmetric matrix
            #x=random.randint(0,7257)
            #y=random.randint(0,11070)
            #row += [x]
            #col += [y]
            row += [self.data.protein[pair[0]]]
            col += [self.data.disease[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_proteins,self.num_diseases),dtype=np.float32)
        return u_i_adj
    def buildJointAdjacency(self):
        indices = [[self.data.herb[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_herbs, self.num_diseases])
        return norm_adj

    def buildGraphAndPageRank(self, adjacency_matrix):
        G = nx.Graph()
        coo = adjacency_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            G.add_edge(i, j, weight=v)
        pr = nx.pagerank(G, alpha=0.85)
        return pr
    # def buildJointAdjacency(self):    #建立联合邻接矩阵
    #     indices = [[self.data.compound[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
    #     values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
    #               for item in self.data.trainingData]
    #     norm_adj = tf.SparseTensor(indices=indices, values=values,
    #                                dense_shape=[self.num_compounds, self.num_proteins])
    #     return norm_adj

    def initModel(self):
        super(HDCTI, self).initModel()

        def safe_reciprocal(values):
            values = np.asarray(values, dtype=np.float32)
            result = np.zeros_like(values)
            np.divide(1.0, values, out=result, where=values != 0)
            return result

        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #A=A.dot(A.transpose().dot(A))
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()



        
        # 计算 hc 的转置
        H_c = hc.transpose()
        # 计算 H_c 中每一行的和，并将其形状调整为 (1, -1)
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        # 计算 H_c 中每一列的和，并将其形状调整为 (1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        # 计算边的归一化矩阵
        temp1 = (H_c.multiply(safe_reciprocal(D_hc_e))).transpose()
        # 计算节点的归一化矩阵
        temp2 = (H_c.transpose().multiply(safe_reciprocal(D_hc_v))).transpose()
        # 将边的矩阵转换为 COO 格式
        edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        node = temp2.tocoo()
        # 获取边和节点的索引
        edge_indices = np.asarray([edge.row, edge.col], dtype=np.int64).T
        node_indices = np.asarray([node.row, node.col], dtype=np.int64).T
        # 构建稀疏张量表示边和节点
        H_e = tf.sparse_reorder(
            tf.SparseTensor(edge_indices, edge.data.astype(np.float32), edge.shape)
        )
        H_n = tf.sparse_reorder(
            tf.SparseTensor(node_indices, node.data.astype(np.float32), node.shape)
        )
        compound_herbs = [[] for _ in range(self.num_compounds)]
        hc_coo = hc.tocoo()
        for herb_index, compound_index in zip(hc_coo.row, hc_coo.col):
            compound_herbs[int(compound_index)].append(int(herb_index))
        compound_herbs = [sorted(set(indices)) for indices in compound_herbs]
        self.compound_herbs = compound_herbs
        self.max_compound_herbs = max(
            [len(indices) for indices in compound_herbs] + [1]
        )
        herb_sentinel = self.num_herbs
        self.compound_herb_indices = np.full(
            (self.num_compounds, self.max_compound_herbs),
            herb_sentinel,
            dtype=np.int32,
        )
        self.compound_herb_mask = np.zeros(
            (self.num_compounds, self.max_compound_herbs), dtype=np.float32
        )
        for compound_index, herb_indices in enumerate(compound_herbs):
            degree = len(herb_indices)
            if degree:
                self.compound_herb_indices[compound_index, :degree] = herb_indices
                self.compound_herb_mask[compound_index, :degree] = 1.0
        hc_attention_edge_ids = np.asarray(hc_coo.row, dtype=np.int32)
        hc_attention_node_ids = np.asarray(hc_coo.col, dtype=np.int32)
        hc_attention_ids = ordered_incidence_ids(
            hc_attention_edge_ids, hc_attention_node_ids
        )
        hc_edge_degrees = np.bincount(
            hc_attention_edge_ids, minlength=self.num_herbs
        ).astype(np.float32)
        hc_specificity_prior = hyperedge_specificity_prior(
            hc_edge_degrees, self.num_compounds
        )
        
        # 获取 pd 的矩阵
        P_d = pd
        # 计算 P_d 中每一行的和，并将其形状调整为 (1, -1)
        D_P_v = P_d.sum(axis=1).reshape(1, -1)
        # 计算 P_d 中每一列的和，并将其形状调整为 (1, -1)
        D_P_e = P_d.sum(axis=0).reshape(1, -1)
        # 计算边的矩阵
        temp1 = (P_d.multiply(safe_reciprocal(D_P_e))).transpose()
        # 计算节点的矩阵
        temp2 = (P_d.transpose().multiply(safe_reciprocal(D_P_v))).transpose()
        # 将边的矩阵转换为 COO 格式
        pd_edge = temp1.tocoo()
        # 将节点的矩阵转换为 COO 格式
        pd_node = temp2.tocoo()
        # 获取边和节点的索引
        A_pde = np.asarray([pd_edge.row, pd_edge.col], dtype=np.int64).T
        A_pdn = np.asarray([pd_node.row, pd_node.col], dtype=np.int64).T
        # 构建稀疏张量表示边和节点
        P_e = tf.sparse_reorder(
            tf.SparseTensor(A_pde, pd_edge.data.astype(np.float32), pd_edge.shape)
        )
        P_n = tf.sparse_reorder(
            tf.SparseTensor(A_pdn, pd_node.data.astype(np.float32), pd_node.shape)
        )
        pd_coo = pd.tocoo()
        pd_attention_edge_ids = np.asarray(pd_coo.col, dtype=np.int32)
        pd_attention_node_ids = np.asarray(pd_coo.row, dtype=np.int32)
        pd_attention_ids = ordered_incidence_ids(
            pd_attention_edge_ids, pd_attention_node_ids
        )
        pd_edge_degrees = np.bincount(
            pd_attention_edge_ids, minlength=self.num_diseases
        ).astype(np.float32)
        pd_specificity_prior = hyperedge_specificity_prior(
            pd_edge_degrees, self.num_proteins
        )
        self.hyperedge_attention_structure = {
            'hc': {
                'nodes': int(self.num_compounds),
                'hyperedges': int(self.num_herbs),
                'incidences': int(hc_attention_edge_ids.size),
                'specificity_prior_min': float(np.min(hc_specificity_prior)),
                'specificity_prior_max': float(np.max(hc_specificity_prior)),
            },
            'pd': {
                'nodes': int(self.num_proteins),
                'hyperedges': int(self.num_diseases),
                'incidences': int(pd_attention_edge_ids.size),
                'specificity_prior_min': float(np.min(pd_specificity_prior)),
                'specificity_prior_max': float(np.max(pd_specificity_prior)),
            },
        }




        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer =tf.keras.initializers.glorot_normal()
        self.n_layer = 2
        self.weights={}
        self.attention_weights = {}
        self.context_terms = resolve_context_terms(self.config)
        self.use_context_interaction = any(self.context_terms.values())
        self.herb_context_attention = resolve_herb_context_attention(self.config)
        self.counterfactual_context = resolve_counterfactual_context(self.config)
        self.context_mask_training = resolve_context_mask_training(self.config)
        self.support_router = resolve_support_router(self.config)
        self.hyperedge_attention = resolve_hyperedge_attention(self.config)
        self.global_token_attention = resolve_global_token_attention(self.config)
        if (
            self.hyperedge_attention['enabled']
            and self.global_token_attention['enabled']
        ):
            raise ValueError(
                'Global token attention and factorized hyperedge attention '
                'cannot be enabled in the same experiment.'
            )
        if self.support_router['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Support routing requires experiment.protocol=strict.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError('Support routing requires static Hctx-P.')
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Support routing requires HerbOnly context terms.'
                )
            if self.context_mask_training['enabled']:
                raise ValueError(
                    'Support routing cannot be combined with context-masked training.'
                )
            if self.data.pseudo_cold_info is None:
                raise ValueError(
                    'Support routing requires deterministic pseudo-cold graph masking.'
                )
        if self.hyperedge_attention['enabled']:
            print(
                'Hyperedge attention: mode=%s H-C=%s P-D=%s '
                'temperature=%g prior_scale=%g.' % (
                    self.hyperedge_attention['mode'],
                    'on' if self.hyperedge_attention['hc_enabled'] else 'off',
                    'on' if self.hyperedge_attention['pd_enabled'] else 'off',
                    self.hyperedge_attention['temperature'],
                    self.hyperedge_attention['prior_scale'],
                )
            )
        if self.global_token_attention['enabled']:
            if self.emb_size % self.global_token_attention['heads'] != 0:
                raise ValueError(
                    'num.factors must be divisible by '
                    'global.token.attention.heads.'
                )
            print(
                'HILGA global token attention: H-C=%s P-D=%s tokens=%d '
                'heads=%d temperature=%g.' % (
                    'on' if self.global_token_attention['hc_enabled'] else 'off',
                    'on' if self.global_token_attention['pd_enabled'] else 'off',
                    self.global_token_attention['tokens'],
                    self.global_token_attention['heads'],
                    self.global_token_attention['temperature'],
                )
            )

        self.compound_cp_support_degrees = np.zeros(
            self.num_compounds, dtype=np.float32
        )
        for compound_id, _, label in self.data.cpassociation:
            if float(label) > 0:
                self.compound_cp_support_degrees[
                    self.data.compound[str(compound_id)]
                ] += 1.0
        self.compound_context_available = (
            np.sum(self.compound_herb_mask, axis=1) > 0
        ).astype(np.float32)
        if (
            self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
            and not self.context_terms['herb_protein']
        ):
            raise ValueError(
                '%s requires context.herb_protein=True.' %
                self.herb_context_attention['mode']
            )
        if self.counterfactual_context['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Counterfactual herb context requires experiment.protocol=strict.'
                )
            if self.herb_context_attention['mode'] != 'static':
                raise ValueError(
                    'Counterfactual herb context requires static Hctx-P.'
                )
            if self.context_terms != {
                    'compound_disease': False,
                    'herb_protein': True,
                    'herb_disease': False}:
                raise ValueError(
                    'Counterfactual herb context requires HerbOnly context terms.'
                )
            from util.counterfactual_context import (
                build_exact_degree_counterfactuals,
            )
            compound_memberships = {
                str(self.data.id2compound[index]): {
                    str(self.data.id2herb[herb]) for herb in herbs
                }
                for index, herbs in enumerate(self.compound_herbs)
                if herbs
            }
            matching = build_exact_degree_counterfactuals(
                compound_memberships.keys(),
                compound_memberships,
                draws=self.counterfactual_context['draws'],
                seed=self.counterfactual_context['seed'],
            )
            donor_indices = np.tile(
                np.arange(self.num_compounds, dtype=np.int32)[:, None],
                (1, self.counterfactual_context['draws']),
            )
            donor_eligible = np.zeros(self.num_compounds, dtype=np.float32)
            for compound_id, donors in matching['assignments'].items():
                index = self.data.compound[str(compound_id)]
                donor_indices[index] = np.asarray([
                    self.data.compound[str(donor)] for donor in donors
                ], dtype=np.int32)
                donor_eligible[index] = 1.0
            self.counterfactual_donor_indices = donor_indices
            self.counterfactual_donor_eligible = donor_eligible
            self.counterfactual_u_idx = tf.placeholder(
                tf.int32, name='counterfactual_u_idx'
            )
            self.counterfactual_eligible_mask = tf.placeholder(
                tf.float32, name='counterfactual_eligible_mask'
            )
            import hashlib
            assignment_hash = hashlib.sha256(
                donor_indices.tobytes() + donor_eligible.tobytes()
            ).hexdigest()
            self.counterfactual_context['assignment_sha256'] = assignment_hash
            self.counterfactual_context['eligible_compounds'] = int(
                np.sum(donor_eligible)
            )
            print(
                'CHCR: enabled weight=%g margin=%g draws=%d seed=%d '
                'eligible_compounds=%d/%d assignment_hash=%s' % (
                    self.counterfactual_context['weight'],
                    self.counterfactual_context['margin'],
                    self.counterfactual_context['draws'],
                    self.counterfactual_context['seed'],
                    self.counterfactual_context['eligible_compounds'],
                    self.num_compounds,
                    assignment_hash[:12],
                )
            )
        else:
            self.counterfactual_donor_indices = None
            self.counterfactual_donor_eligible = None
            self.counterfactual_u_idx = None
            self.counterfactual_eligible_mask = None
        if self.context_mask_training['enabled']:
            if getattr(self.data, 'protocol', 'legacy') != 'strict':
                raise ValueError(
                    'Context-masked training requires experiment.protocol=strict.'
                )
            print(
                'CMIT: enabled side=%s weight=%g' % (
                    self.context_mask_training['side'],
                    self.context_mask_training['weight'],
                )
            )
        self.pair_decoder = resolve_pair_decoder(self.config)
        print(
            'Candidate context terms: C-Dctx=%s, Hctx-P=%s, Hctx-Dctx=%s' % (
                'on' if self.context_terms['compound_disease'] else 'off',
                'on' if self.context_terms['herb_protein'] else 'off',
                'on' if self.context_terms['herb_disease'] else 'off',
            )
        )
        print(
            'Herb context mode: %s (max incident herbs=%d)' % (
                self.herb_context_attention['mode'], self.max_compound_herbs
            )
        )
        if self.support_router['enabled']:
            print(
                'Support router: mode=%s pseudo_cold_ratio=%.4f seed=%d '
                'initial_slope=%g.' % (
                    self.support_router['mode'],
                    self.support_router['pseudo_cold_ratio'],
                    self.support_router['seed'],
                    self.support_router['initial_slope'],
                )
            )
        print('Pair decoder: %s' % self.pair_decoder['type'])
        attention_size = 64
        self.compound_attention_weights = []
        self.protein_attention_weights = []

        num_heads = 2  # Number of attention heads
        head_dim = self.emb_size // num_heads  # Dimension of each attention head
        attention_max_nodes = None
        if self.config.contains('attention.max.nodes'):
            attention_max_nodes = int(self.config['attention.max.nodes'])
        use_compound_full_attention = attention_max_nodes is None or self.num_compounds <= attention_max_nodes
        use_protein_full_attention = attention_max_nodes is None or self.num_proteins <= attention_max_nodes
        global_attention_conflicts = (
            self.global_token_attention['hc_enabled']
            and use_compound_full_attention
        ) or (
            self.global_token_attention['pd_enabled']
            and use_protein_full_attention
        )
        if global_attention_conflicts:
            raise ValueError(
                'HILGA replaces dense full-node attention; set '
                'attention.max.nodes=0 for HILGA experiments.'
            )
        if not use_compound_full_attention:
            print('Skipping compound full self-attention: nodes=%d > attention.max.nodes=%d' %
                  (self.num_compounds, attention_max_nodes))
        if not use_protein_full_attention:
            print('Skipping protein full self-attention: nodes=%d > attention.max.nodes=%d' %
                  (self.num_proteins, attention_max_nodes))



        if getattr(self.data, 'protocol', 'legacy') == 'strict':
            pr_compound_embeddings, _ = bipartite_pagerank(cp)
            pr_protein_embeddings, _ = bipartite_pagerank(pd)
            print('Strict PageRank: fold training C-P graph with type-safe bipartite node IDs.')
        else:
            pr_compound = self.buildGraphAndPageRank(cp)
            pr_protein = self.buildGraphAndPageRank(pd)
            pr_compound_embeddings = np.array([pr_compound.get(i, 0) for i in range(self.num_compounds)])
            pr_protein_embeddings = np.array([pr_protein.get(i, 0) for i in range(self.num_proteins)])
        pr_compound_embeddings = np.reshape(pr_compound_embeddings, (self.num_compounds, 1))
        pr_protein_embeddings = np.reshape(pr_protein_embeddings, (self.num_proteins, 1))


        initializer = tf.variance_scaling_initializer(scale=2.0)

        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_1_%d' % (i + 1))
            self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_2_%d' % (i + 1))
            self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
            self.attention_weights['compound' ] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='compound')
            self.attention_weights['protein'] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='protein')
            for h in range(num_heads):
                self.attention_weights['compound_q_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_q_%d_%d' % (i + 1, h))
                self.attention_weights['compound_k_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_k_%d_%d' % (i + 1, h))
                self.attention_weights['compound_v_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='compound_v_%d_%d' % (i + 1, h))

                self.attention_weights['protein_q_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_q_%d_%d' % (i + 1, h))
                self.attention_weights['protein_k_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_k_%d_%d' % (i + 1, h))
                self.attention_weights['protein_v_%d_%d' % (i + 1, h)] = tf.Variable(
                    initializer([self.emb_size, head_dim]), name='protein_v_%d_%d' % (i + 1, h))
            if self.hyperedge_attention['hc_enabled']:
                self.weights['hc_hyper_node_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='hc_hyper_node_%d' % (i + 1),
                )
                self.weights['hc_hyper_edge_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='hc_hyper_edge_%d' % (i + 1),
                )
            if self.hyperedge_attention['pd_enabled']:
                self.weights['pd_hyper_node_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='pd_hyper_node_%d' % (i + 1),
                )
                self.weights['pd_hyper_edge_%d' % (i + 1)] = tf.Variable(
                    tf.zeros([self.emb_size]),
                    name='pd_hyper_edge_%d' % (i + 1),
                )

        for i in range(2):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            # self.weights['gating_bias%d' % (i + 1)] = tf.Variable(tf.zeros([1, self.emb_size]),
            #                                                       name='g_W_b_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                       name='g_W_b_%d_1' % (i + 1))

        if self.use_context_interaction:
            # Zero initialization keeps the first forward pass equal to the dot-product baseline.
            self.weights['context_compound_disease'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_compound_disease'
            )
            self.weights['context_herb_protein'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_herb_protein'
            )
            self.weights['context_herb_disease'] = tf.Variable(
                tf.zeros([self.emb_size]), name='context_herb_disease'
            )
        if self.support_router['enabled']:
            self.weights['support_router_raw_slope'] = tf.Variable(
                np.asarray([
                    inverse_softplus(self.support_router['initial_slope'])
                ], dtype=np.float32),
                name='support_router_raw_slope',
            )
        if self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES:
            self.weights['target_herb_projection'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32),
                name='target_herb_projection',
            )
            self.weights['target_protein_projection'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32),
                name='target_protein_projection',
            )
        if self.herb_context_attention['mode'] == 'target_residual_attention':
            # V2 starts exactly from static Hctx-P and learns only a pair-specific delta.
            self.weights['context_target_herb_residual'] = tf.Variable(
                tf.zeros([self.emb_size]),
                name='context_target_herb_residual',
            )

        if self.pair_decoder['type'] == 'bilinear':
            # Identity initialization makes the first forward pass equal to Dot.
            self.weights['decoder_bilinear'] = tf.Variable(
                tf.eye(self.emb_size, dtype=tf.float32), name='decoder_bilinear'
            )
        elif self.pair_decoder['type'] == 'mlp':
            hidden_size = self.pair_decoder['hidden_size']
            self.weights['decoder_mlp_hidden'] = tf.Variable(
                initializer([self.emb_size * 4, hidden_size]), name='decoder_mlp_hidden'
            )
            self.weights['decoder_mlp_hidden_bias'] = tf.Variable(
                tf.zeros([hidden_size]), name='decoder_mlp_hidden_bias'
            )
            # A zero output layer gives an exact Dot residual initialization.
            self.weights['decoder_mlp_output'] = tf.Variable(
                tf.zeros([hidden_size, 1]), name='decoder_mlp_output'
            )
            self.weights['decoder_mlp_output_bias'] = tf.Variable(
                tf.zeros([1]), name='decoder_mlp_output_bias'
            )

        self.global_token_diagnostic_tensors = {}
        if self.global_token_attention['enabled']:
            token_count = self.global_token_attention['tokens']
            for side in ('hc', 'pd'):
                if not self.global_token_attention[side + '_enabled']:
                    continue
                self.global_token_diagnostic_tensors[side] = []
                for layer in range(1, self.n_layer + 1):
                    prefix = '%s_global_token' % side
                    self.weights['%s_assignment_%d' % (prefix, layer)] = tf.Variable(
                        initializer([self.emb_size, token_count]),
                        name='%s_assignment_%d' % (prefix, layer),
                    )
                    for projection in ('q', 'k', 'v', 'output'):
                        self.weights['%s_%s_%d' % (
                            prefix, projection, layer
                        )] = tf.Variable(
                            initializer([self.emb_size, self.emb_size]),
                            name='%s_%s_%d' % (prefix, projection, layer),
                        )
                    self.weights['%s_gamma_%d' % (prefix, layer)] = tf.Variable(
                        tf.zeros([1]), name='%s_gamma_%d' % (prefix, layer)
                    )

        def multi_head_attention_compound(embeddings, attention_weights, num_heads, head_dim):
            if not use_compound_full_attention:
                return embeddings
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['compound_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['compound_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['compound_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention

        def multi_head_attention_protein(embeddings, attention_weights, num_heads, head_dim):
            if not use_protein_full_attention:
                return embeddings
            attention_heads = []

            for h in range(num_heads):
                q = tf.matmul(embeddings, attention_weights['protein_q_%d_%d' % (i + 1, h)])
                k = tf.matmul(embeddings, attention_weights['protein_k_%d_%d' % (i + 1, h)])
                v = tf.matmul(embeddings, attention_weights['protein_v_%d_%d' % (i + 1, h)])

                attn_logits = tf.matmul(q, k, transpose_b=True)
                attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(head_dim)), axis=-1)
                attn_output = tf.matmul(attn_weights, v)

                attention_heads.append(attn_output)

            # Concatenate all attention heads
            concat_attention = tf.concat(attention_heads, axis=-1)
            return concat_attention


        def self_gating(em,channel):

            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))

        def unsorted_segment_softmax(logits, segment_ids, segment_count):
            maxima = tf.math.unsorted_segment_max(
                logits, segment_ids, segment_count
            )
            shifted = logits - tf.gather(maxima, segment_ids)
            exponentials = tf.exp(shifted)
            denominators = tf.math.unsorted_segment_sum(
                exponentials, segment_ids, segment_count
            )
            return exponentials / (
                tf.gather(denominators, segment_ids) + 1e-12
            )

        def factorized_hyperedge_propagation(
                node_embeddings,
                forward_edge_ids,
                forward_node_ids,
                reverse_edge_ids,
                reverse_node_ids,
                edge_count,
                node_count,
                specificity_prior,
                node_score_vector,
                edge_score_vector,
                name):
            temperature = float(self.hyperedge_attention['temperature'])
            prior_scale = float(self.hyperedge_attention['prior_scale'])
            node_logits = tf.reduce_sum(
                node_embeddings * node_score_vector, axis=1
            )
            node_to_edge_values = unsorted_segment_softmax(
                tf.gather(node_logits, forward_node_ids) / temperature,
                forward_edge_ids,
                edge_count,
            )
            node_to_edge = tf.SparseTensor(
                tf.stack([forward_edge_ids, forward_node_ids], axis=1),
                node_to_edge_values,
                [edge_count, node_count],
            )
            edge_embeddings = tf.sparse_tensor_dense_matmul(
                node_to_edge,
                node_embeddings,
                name=name + '_node_to_edge',
            )
            edge_logits = tf.reduce_sum(
                edge_embeddings * edge_score_vector, axis=1
            ) + prior_scale * specificity_prior
            edge_to_node_values = unsorted_segment_softmax(
                tf.gather(edge_logits, reverse_edge_ids) / temperature,
                reverse_node_ids,
                node_count,
            )
            edge_to_node = tf.SparseTensor(
                tf.stack([reverse_node_ids, reverse_edge_ids], axis=1),
                edge_to_node_values,
                [node_count, edge_count],
            )
            propagated_nodes = tf.sparse_tensor_dense_matmul(
                edge_to_node,
                edge_embeddings,
                name=name + '_edge_to_node',
            )
            return edge_embeddings, propagated_nodes

        def hyperedge_induced_global_attention(
                node_embeddings,
                edge_embeddings,
                active_edge_mask,
                side,
                layer):
            token_count = self.global_token_attention['tokens']
            head_count = self.global_token_attention['heads']
            head_dim = self.emb_size // head_count
            temperature = self.global_token_attention['temperature']
            prefix = '%s_global_token' % side

            normalized_edges = tf.math.l2_normalize(edge_embeddings, axis=1)
            assignment_logits = tf.matmul(
                normalized_edges,
                self.weights['%s_assignment_%d' % (prefix, layer)],
            )
            edge_mask = tf.reshape(active_edge_mask, [-1, 1])
            assignment_logits += (1.0 - edge_mask) * -1e9
            edge_assignments = tf.nn.softmax(
                assignment_logits / temperature,
                axis=0,
                name='%s_assignment_layer_%d' % (prefix, layer),
            )
            token_embeddings = tf.matmul(
                edge_assignments,
                edge_embeddings,
                transpose_a=True,
                name='%s_pool_layer_%d' % (prefix, layer),
            )

            queries = tf.matmul(
                tf.math.l2_normalize(node_embeddings, axis=1),
                self.weights['%s_q_%d' % (prefix, layer)],
            )
            keys = tf.matmul(
                tf.math.l2_normalize(token_embeddings, axis=1),
                self.weights['%s_k_%d' % (prefix, layer)],
            )
            values = tf.matmul(
                token_embeddings,
                self.weights['%s_v_%d' % (prefix, layer)],
            )
            queries = tf.transpose(
                tf.reshape(queries, [-1, head_count, head_dim]), [1, 0, 2]
            )
            keys = tf.transpose(
                tf.reshape(keys, [token_count, head_count, head_dim]), [1, 0, 2]
            )
            values = tf.transpose(
                tf.reshape(values, [token_count, head_count, head_dim]), [1, 0, 2]
            )
            node_attention = tf.nn.softmax(
                tf.matmul(queries, keys, transpose_b=True)
                / (tf.sqrt(float(head_dim)) * temperature),
                axis=-1,
                name='%s_node_attention_layer_%d' % (prefix, layer),
            )
            attended = tf.matmul(node_attention, values)
            attended = tf.reshape(
                tf.transpose(attended, [1, 0, 2]), [-1, self.emb_size]
            )
            update = tf.matmul(
                attended,
                self.weights['%s_output_%d' % (prefix, layer)],
                name='%s_update_layer_%d' % (prefix, layer),
            )
            residual_scale = tf.tanh(
                self.weights['%s_gamma_%d' % (prefix, layer)][0]
            )
            self.global_token_diagnostic_tensors[side].append({
                'edge_assignments': edge_assignments,
                'node_attention': node_attention,
                'token_embeddings': token_embeddings,
            })
            return tf.add(
                node_embeddings,
                residual_scale * update,
                name='%s_residual_layer_%d' % (prefix, layer),
            )

        hc_attention_edge_ids_tf = tf.constant(
            hc_attention_ids['forward_edge_ids'], dtype=tf.int64
        )
        hc_attention_node_ids_tf = tf.constant(
            hc_attention_ids['forward_node_ids'], dtype=tf.int64
        )
        hc_attention_reverse_edge_ids_tf = tf.constant(
            hc_attention_ids['reverse_edge_ids'], dtype=tf.int64
        )
        hc_attention_reverse_node_ids_tf = tf.constant(
            hc_attention_ids['reverse_node_ids'], dtype=tf.int64
        )
        hc_specificity_prior_tf = tf.constant(
            hc_specificity_prior, dtype=tf.float32
        )
        hc_active_edge_mask_tf = tf.constant(
            (hc_edge_degrees > 0).astype(np.float32), dtype=tf.float32
        )
        pd_attention_edge_ids_tf = tf.constant(
            pd_attention_ids['forward_edge_ids'], dtype=tf.int64
        )
        pd_attention_node_ids_tf = tf.constant(
            pd_attention_ids['forward_node_ids'], dtype=tf.int64
        )
        pd_attention_reverse_edge_ids_tf = tf.constant(
            pd_attention_ids['reverse_edge_ids'], dtype=tf.int64
        )
        pd_attention_reverse_node_ids_tf = tf.constant(
            pd_attention_ids['reverse_node_ids'], dtype=tf.int64
        )
        pd_specificity_prior_tf = tf.constant(
            pd_specificity_prior, dtype=tf.float32
        )
        pd_active_edge_mask_tf = tf.constant(
            (pd_edge_degrees > 0).astype(np.float32), dtype=tf.float32
        )

        compound_embeddings = self_gating(self.compound_embeddings, 1)
        protein_embeddings = self_gating(self.protein_embeddings, 2)

        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = protein_embeddings
        all_protein_embeddings = [protein_embeddings]
        all_hc_embeddings = []
        all_pd_embeddings = []

        for i in range(self.n_layer):
            if self.hyperedge_attention['hc_enabled']:
                new_hc_edge, new_compound_embeddings = (
                    factorized_hyperedge_propagation(
                        compound_embeddings,
                        hc_attention_edge_ids_tf,
                        hc_attention_node_ids_tf,
                        hc_attention_reverse_edge_ids_tf,
                        hc_attention_reverse_node_ids_tf,
                        self.num_herbs,
                        self.num_compounds,
                        hc_specificity_prior_tf,
                        self.weights['hc_hyper_node_%d' % (i + 1)],
                        self.weights['hc_hyper_edge_%d' % (i + 1)],
                        'hc_factorized_attention_layer_%d' % (i + 1),
                    )
                )
            else:
                new_hc_edge = tf.sparse_tensor_dense_matmul(
                    H_e,
                    compound_embeddings,
                    name='hc_node_to_edge_layer_%d' % (i + 1),
                )
                new_compound_embeddings = tf.sparse_tensor_dense_matmul(
                    H_n,
                    new_hc_edge,
                    name='hc_edge_to_node_layer_%d' % (i + 1),
                )
            if self.hyperedge_attention['pd_enabled']:
                new_pd_edge, new_protein_embeddings = (
                    factorized_hyperedge_propagation(
                        protein_embeddings,
                        pd_attention_edge_ids_tf,
                        pd_attention_node_ids_tf,
                        pd_attention_reverse_edge_ids_tf,
                        pd_attention_reverse_node_ids_tf,
                        self.num_diseases,
                        self.num_proteins,
                        pd_specificity_prior_tf,
                        self.weights['pd_hyper_node_%d' % (i + 1)],
                        self.weights['pd_hyper_edge_%d' % (i + 1)],
                        'pd_factorized_attention_layer_%d' % (i + 1),
                    )
                )
            else:
                new_pd_edge = tf.sparse_tensor_dense_matmul(
                    P_e,
                    protein_embeddings,
                    name='pd_node_to_edge_layer_%d' % (i + 1),
                )
                new_protein_embeddings = tf.sparse_tensor_dense_matmul(
                    P_n,
                    new_pd_edge,
                    name='pd_edge_to_node_layer_%d' % (i + 1),
                )
            new_compound_embeddings = new_compound_embeddings * pr_compound_embeddings
            new_protein_embeddings = new_protein_embeddings * pr_protein_embeddings
        
            new_compound_embeddings = multi_head_attention_compound(new_compound_embeddings, self.attention_weights, num_heads,
                                                            head_dim)
            new_compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            if self.global_token_attention['hc_enabled']:
                new_compound_embeddings = hyperedge_induced_global_attention(
                    new_compound_embeddings,
                    new_hc_edge,
                    hc_active_edge_mask_tf,
                    'hc',
                    i + 1,
                )
        
            # 计算蛋白质节点的新嵌入
            new_protein_embeddings = multi_head_attention_protein(new_protein_embeddings, self.attention_weights, num_heads,
                                                          head_dim)
            new_protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
            if self.global_token_attention['pd_enabled']:
                new_protein_embeddings = hyperedge_induced_global_attention(
                    new_protein_embeddings,
                    new_pd_edge,
                    pd_active_edge_mask_tf,
                    'pd',
                    i + 1,
                )
        
            # 添加节点的注意力机制
        
            attn_weights_compound = tf.nn.softmax(
                tf.matmul(new_compound_embeddings, self.attention_weights['compound']))
            attn_weights_protein = tf.nn.softmax(tf.matmul(new_protein_embeddings, self.attention_weights['protein']))
        
            # 使用注意力权重对邻居节点进行加权求和
            new_compound_embeddings = tf.nn.leaky_relu(tf.matmul(attn_weights_compound * new_compound_embeddings,
                                                self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            new_protein_embeddings = tf.nn.leaky_relu(tf.matmul(attn_weights_protein * new_protein_embeddings,
                                               self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
        
            compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights['layer_%d' % (i + 1)]) + compound_embeddings)
            
            protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_protein_embeddings, self.weights['layer_1_%d' % (i + 1)]) + protein_embeddings)
        
            compound_embeddings = tf.nn.leaky_relu(new_compound_embeddings)
            protein_embeddings = tf.nn.leaky_relu(new_protein_embeddings)
            # compound_embeddings = tf.nn.leaky_relu(compound_embeddings)
            # protein_embeddings = tf.nn.leaky_relu(protein_embeddings)
        
            compound_embeddings = tf.math.l2_normalize(compound_embeddings,axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings,axis=1)
            new_hc_edge=tf.math.l2_normalize(new_hc_edge,axis=1)
            new_pd_edge=tf.math.l2_normalize(new_pd_edge,axis=1)
        
        
        
            all_compound_embeddings+=[compound_embeddings]
            all_protein_embeddings+=[protein_embeddings]
            all_hc_embeddings+=[new_hc_edge]
            all_pd_embeddings+=[new_pd_edge]


        compound_embeddings = tf.reduce_sum(all_compound_embeddings,axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        # compound_embeddings = tf.math.l2_normalize(compound_embeddings)
        # protein_embeddings = tf.math.l2_normalize(protein_embeddings)
        compound_embeddings = tf.nn.leaky_relu(compound_embeddings, alpha=0.2)
        protein_embeddings = tf.nn.leaky_relu(protein_embeddings,alpha=0.2)
        # a = tf.reduce_sum(all_compound_embeddings, axis=0)
        # b = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge=tf.reduce_sum(all_hc_embeddings,axis=0)
        pd_edge=tf.reduce_sum(all_pd_embeddings,axis=0)

        # new_hc_dege = tf.nn.leaky_relu(hc_edge, alpha=0.2)
        # new_pd_edge = tf.nn.leaky_relu(pd_edge, alpha=0.2)



        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")

        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)
        #self.neg_disease_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(A.toarray(),dtype=tf.float32), self.u_idx)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings

        self.final_hcedge=hc_edge
        self.final_pdedge=pd_edge

        # Aggregate only the candidate's incident H-C/P-D hyperedges. No H-D or C-P labels are used here.
        self.final_compound_context = tf.math.l2_normalize(
            tf.sparse_tensor_dense_matmul(
                H_n, self.final_hcedge, name='hc_candidate_context'
            ),
            axis=1,
        )
        self.final_protein_context = tf.math.l2_normalize(
            tf.sparse_tensor_dense_matmul(
                P_n, self.final_pdedge, name='pd_candidate_context'
            ),
            axis=1,
        )

        # self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        # self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_uembedding, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_iembedding, self.v_idx)
        self.u_context_embedding = tf.nn.embedding_lookup(self.final_compound_context, self.u_idx)
        self.v_context_embedding = tf.nn.embedding_lookup(self.final_protein_context, self.v_idx)
        self.support_context_gate = None
        if self.support_router['enabled']:
            pair_support_degrees = tf.gather(
                tf.constant(
                    self.compound_cp_support_degrees, dtype=tf.float32
                ),
                self.u_idx,
            )
            pair_context_available = tf.gather(
                tf.constant(
                    self.compound_context_available, dtype=tf.float32
                ),
                self.u_idx,
            )
            support_slope = tf.nn.softplus(
                self.weights['support_router_raw_slope'][0]
            )
            self.support_context_gate = pair_context_available * tf.exp(
                -support_slope * tf.log1p(pair_support_degrees)
            )
        self.target_herb_attention_weights = None
        self.target_herb_context_embedding = None
        if self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES:
            padded_herb_edges = tf.concat(
                [
                    self.final_hcedge,
                    tf.zeros([1, self.emb_size], dtype=tf.float32),
                ],
                axis=0,
            )
            pair_herb_indices = tf.gather(
                tf.constant(self.compound_herb_indices, dtype=tf.int32),
                self.u_idx,
            )
            pair_herb_mask = tf.gather(
                tf.constant(self.compound_herb_mask, dtype=tf.float32),
                self.u_idx,
            )
            pair_herb_edges = tf.gather(padded_herb_edges, pair_herb_indices)
            pair_herb_edges.set_shape(
                [None, self.max_compound_herbs, self.emb_size]
            )
            herb_keys = tf.tensordot(
                pair_herb_edges,
                self.weights['target_herb_projection'],
                axes=[[2], [0]],
            )
            herb_keys.set_shape([None, self.max_compound_herbs, self.emb_size])
            protein_queries = tf.matmul(
                self.v_embedding, self.weights['target_protein_projection']
            )
            attention_logits = tf.reduce_sum(
                herb_keys * tf.expand_dims(protein_queries, axis=1), axis=2
            )
            attention_logits.set_shape([None, self.max_compound_herbs])
            attention_logits /= (
                tf.sqrt(tf.cast(self.emb_size, tf.float32))
                * self.herb_context_attention['temperature']
            )
            masked_logits = tf.where(
                pair_herb_mask > 0,
                attention_logits,
                tf.fill(tf.shape(attention_logits), tf.constant(-1e9, tf.float32)),
            )
            masked_logits = tf.reshape(
                masked_logits, [-1, self.max_compound_herbs]
            )
            attention_weights = tf.nn.softmax(masked_logits, axis=-1) * pair_herb_mask
            attention_weights /= (
                tf.reduce_sum(attention_weights, axis=1, keepdims=True) + 1e-12
            )
            self.target_herb_attention_weights = attention_weights
            self.target_herb_context_embedding = tf.math.l2_normalize(
                tf.reduce_sum(
                    tf.expand_dims(attention_weights, axis=2) * pair_herb_edges,
                    axis=1,
                ),
                axis=1,
            )
            if self.herb_context_attention['mode'] == 'target_attention':
                self.u_context_embedding = self.target_herb_context_embedding

        #self.v_embedding = self.final_pdedge


    def buildBasePairLogits(self, compound_embedding=None, protein_embedding=None):
        compound_embedding = (
            self.u_embedding if compound_embedding is None else compound_embedding
        )
        protein_embedding = (
            self.v_embedding if protein_embedding is None else protein_embedding
        )
        dot_logits = tf.reduce_sum(
            tf.multiply(compound_embedding, protein_embedding), 1
        )
        decoder_type = self.pair_decoder['type']
        if decoder_type == 'dot':
            return dot_logits
        if decoder_type == 'bilinear':
            transformed_compound = tf.matmul(
                compound_embedding, self.weights['decoder_bilinear']
            )
            return tf.reduce_sum(transformed_compound * protein_embedding, axis=1)
        if decoder_type == 'mlp':
            pair_features = tf.concat(
                [
                    compound_embedding,
                    protein_embedding,
                    compound_embedding * protein_embedding,
                    tf.abs(compound_embedding - protein_embedding),
                ],
                axis=1,
            )
            hidden = tf.nn.leaky_relu(
                tf.matmul(pair_features, self.weights['decoder_mlp_hidden'])
                + self.weights['decoder_mlp_hidden_bias'],
                alpha=0.2,
            )
            residual = tf.reshape(
                tf.matmul(hidden, self.weights['decoder_mlp_output'])
                + self.weights['decoder_mlp_output_bias'],
                [-1],
            )
            return dot_logits + residual
        raise ValueError('Unsupported pair decoder: %s' % decoder_type)

    def buildPairLogits(self, compound_embedding=None, protein_embedding=None):
        compound_embedding = (
            self.u_embedding if compound_embedding is None else compound_embedding
        )
        protein_embedding = (
            self.v_embedding if protein_embedding is None else protein_embedding
        )
        logits = self.buildBasePairLogits(
            compound_embedding, protein_embedding
        )
        if self.context_terms['compound_disease']:
            logits += tf.reduce_sum(
                compound_embedding * self.v_context_embedding
                * self.weights['context_compound_disease'], axis=1
            )
        if self.context_terms['herb_protein']:
            herb_protein_logits = tf.reduce_sum(
                self.u_context_embedding * protein_embedding
                * self.weights['context_herb_protein'], axis=1
            )
            if self.support_router['enabled']:
                herb_protein_logits *= self.support_context_gate
            logits += herb_protein_logits
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                context_delta = (
                    self.target_herb_context_embedding - self.u_context_embedding
                )
                logits += tf.reduce_sum(
                    context_delta * protein_embedding
                    * self.weights['context_target_herb_residual'], axis=1
                )
        if self.context_terms['herb_disease']:
            logits += tf.reduce_sum(
                self.u_context_embedding * self.v_context_embedding
                * self.weights['context_herb_disease'], axis=1
            )
        return logits

    def supportContextGateValues(self, state, compound_indices):
        if not self.support_router['enabled']:
            return None
        compound_indices = np.asarray(compound_indices, dtype=np.int64)
        raw_slope = np.asarray(
            state['weights']['support_router_raw_slope']
        ).reshape(-1)
        slope = softplus(raw_slope[0])
        return monotonic_context_gate(
            self.compound_cp_support_degrees[compound_indices],
            self.compound_context_available[compound_indices],
            slope,
        )

    def buildContextMaskedTrainingLoss(self):
        if not self.context_mask_training['enabled']:
            return tf.constant(0.0, dtype=tf.float32)
        side = self.context_mask_training['side']
        compound_embedding = (
            self.u_context_embedding if side in {'compound', 'both'}
            else self.u_embedding
        )
        protein_embedding = (
            self.v_context_embedding if side in {'protein', 'both'}
            else self.v_embedding
        )
        masked_logits = self.buildPairLogits(
            compound_embedding=compound_embedding,
            protein_embedding=protein_embedding,
        )
        raw_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.neg_disease_embedding,
            logits=masked_logits,
        ))
        return self.context_mask_training['weight'] * raw_loss

    def buildRegularizationLoss(self):
        weight_decay = float(self.config['weight.reg']) if self.config.contains('weight.reg') else 0.01
        return build_regularization_loss(
            tf,
            self.weights,
            self.final_uembedding,
            self.final_iembedding,
            weight_decay,
            self.regU,
            self.regI,
        )

    def buildCounterfactualContextLoss(self):
        if not self.counterfactual_context['enabled']:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, zero, zero

        counterfactual_context_embedding = tf.nn.embedding_lookup(
            self.final_compound_context, self.counterfactual_u_idx
        )
        factual_context_logits = tf.reduce_sum(
            self.u_context_embedding * self.v_embedding
            * self.weights['context_herb_protein'], axis=1
        )
        counterfactual_context_logits = tf.reduce_sum(
            counterfactual_context_embedding * self.v_embedding
            * self.weights['context_herb_protein'], axis=1
        )
        if self.support_router['enabled']:
            factual_context_logits *= self.support_context_gate
            counterfactual_context_logits *= self.support_context_gate
        positive_mask = tf.cast(
            self.neg_disease_embedding > 0.5, tf.float32
        )
        active_mask = positive_mask * self.counterfactual_eligible_mask
        margins = factual_context_logits - counterfactual_context_logits
        violations = tf.nn.relu(
            self.counterfactual_context['margin'] - margins
        ) * active_mask
        raw_loss = tf.reduce_sum(violations)
        weighted_loss = self.counterfactual_context['weight'] * raw_loss
        active_count = tf.reduce_sum(active_mask)
        active_mean_margin = tf.reduce_sum(margins * active_mask) / tf.maximum(
            active_count, 1.0
        )
        return weighted_loss, active_count, active_mean_margin

    def fetchModelState(self):
        return self.sess.run(
            {
                'compound': self.final_uembedding,
                'protein': self.final_iembedding,
                'compound_context': self.final_compound_context,
                'protein_context': self.final_protein_context,
                'herb_edge': self.final_hcedge,
                'disease_edge': self.final_pdedge,
                'weights': self.weights,
            },
            feed_dict={self.isTraining: 0},
        )

    def targetConditionedHerbContexts(
            self, state, compound_indices, protein_indices):
        if self.herb_context_attention['mode'] not in TARGET_HERB_ATTENTION_MODES:
            return None, None
        weights = state['weights']
        return target_conditioned_herb_contexts(
            state['herb_edge'],
            self.compound_herb_indices,
            self.compound_herb_mask,
            state['protein'],
            compound_indices,
            protein_indices,
            weights['target_herb_projection'],
            weights['target_protein_projection'],
            temperature=self.herb_context_attention['temperature'],
        )

    def evaluateValidation(
            self, state, metric, mask_compound=False, mask_protein=False):
        if not self.validationData:
            raise ValueError('Early stopping is enabled but no inner validation records were provided.')

        compound_indices = []
        protein_indices = []
        labels = []
        for compound_id, protein_id, rating in self.validationData:
            compound_key = str(compound_id)
            protein_key = str(protein_id)
            if compound_key not in self.data.compound or protein_key not in self.data.protein:
                raise ValueError(
                    'Validation pair contains an entity outside the model universe: %s, %s.' %
                    (compound_key, protein_key)
                )
            compound_indices.append(self.data.compound[compound_key])
            protein_indices.append(self.data.protein[protein_key])
            labels.append(1 if float(rating) > 0 else 0)

        labels = np.asarray(labels, dtype=np.int32)
        if len(np.unique(labels)) < 2:
            raise ValueError('Inner validation must contain both positive and negative records.')
        zero_weight = np.zeros(self.emb_size, dtype=np.float32)
        weights = state['weights']
        if mask_compound or mask_protein:
            from util.model_components import context_masked_pair_scores
            logits = context_masked_pair_scores(
                state['compound'],
                state['protein'],
                state['compound_context'],
                state['protein_context'],
                compound_indices,
                protein_indices,
                weights.get('context_compound_disease', zero_weight),
                weights.get('context_herb_protein', zero_weight),
                weights.get('context_herb_disease', zero_weight),
                mask_compound=mask_compound,
                mask_protein=mask_protein,
                enabled_terms=self.context_terms,
                decoder_type=self.pair_decoder['type'],
                decoder_weights=weights,
            )
        else:
            target_compound_contexts, _ = self.targetConditionedHerbContexts(
                state, compound_indices, protein_indices
            )
            pair_compound_contexts = (
                target_compound_contexts
                if self.herb_context_attention['mode'] == 'target_attention'
                else None
            )
            residual_compound_contexts = (
                target_compound_contexts
                if self.herb_context_attention['mode'] == 'target_residual_attention'
                else None
            )
            logits = context_interaction_pair_scores(
                state['compound'],
                state['protein'],
                state['compound_context'],
                state['protein_context'],
                compound_indices,
                protein_indices,
                weights.get('context_compound_disease', zero_weight),
                weights.get('context_herb_protein', zero_weight),
                weights.get('context_herb_disease', zero_weight),
                enabled_terms=self.context_terms,
                decoder_type=self.pair_decoder['type'],
                decoder_weights=weights,
                pair_compound_contexts=pair_compound_contexts,
                residual_compound_contexts=residual_compound_contexts,
                target_residual_weight=weights.get('context_target_herb_residual'),
                herb_protein_scale=self.supportContextGateValues(
                    state, compound_indices
                ),
            )
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        if metric == 'aupr':
            return float(average_precision_score(labels, scores))
        if metric == 'auc':
            return float(roc_auc_score(labels, scores))
        raise ValueError('Unsupported validation metric: %s' % metric)

    def trainModel(self):
        def format_duration(seconds):
            seconds = max(0, int(seconds))
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            if hours:
                return '%dh%02dm%02ds' % (hours, minutes, secs)
            if minutes:
                return '%dm%02ds' % (minutes, secs)
            return '%ds' % secs

        logits = self.buildPairLogits()
        bce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.neg_disease_embedding,
            logits=logits
        ))

        reg_loss = self.buildRegularizationLoss()
        counterfactual_loss, counterfactual_active, counterfactual_margin = (
            self.buildCounterfactualContextLoss()
        )
        context_mask_loss = self.buildContextMaskedTrainingLoss()
        loss = bce_loss + reg_loss + counterfactual_loss + context_mask_loss

        optimizer = tf.train.AdamOptimizer(self.lRate)
        train = optimizer.minimize(loss)


        init = tf.global_variables_initializer()
        self.sess.run(init)

        early_stopping = resolve_early_stopping(self.config)
        if early_stopping['enabled'] and not self.validationData:
            raise ValueError('early.stopping=True requires a non-empty inner validation split.')
        tracker = None
        if early_stopping['enabled']:
            tracker = EarlyStoppingTracker(
                early_stopping['patience'], early_stopping['min_delta']
            )
            print(
                'Early stopping: metric=%s interval=%d patience=%d min_delta=%g validation_pairs=%d' % (
                    early_stopping['metric'].upper(),
                    early_stopping['interval'],
                    early_stopping['patience'],
                    early_stopping['min_delta'],
                    len(self.validationData),
                )
            )

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        model_dir = os.path.join("./saved_model", currentTime)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hdcti_model.ckpt")
        saver = tf.train.Saver(max_to_keep=1)

        total_batches = int(np.ceil(float(self.train_size) / self.batch_size))
        total_steps = max(1, self.maxEpoch * total_batches)
        train_start = time()
        epochs_completed = 0
        for epoch in range(self.maxEpoch):
            epoch_start = time()
            for n, batch in enumerate(self.next_batch_pairwise()):
                batch_start = time()
                herb_idx, i_idx, j_idx = batch
                feed_dict = {
                    self.u_idx: herb_idx,
                    self.neg_idx: j_idx,
                    self.v_idx: i_idx,
                    self.isTraining: 1,
                }
                if self.counterfactual_context['enabled']:
                    compound_batch = np.asarray(herb_idx, dtype=np.int32)
                    draw_index = epoch % self.counterfactual_context['draws']
                    feed_dict[self.counterfactual_u_idx] = (
                        self.counterfactual_donor_indices[
                            compound_batch, draw_index
                        ]
                    )
                    feed_dict[self.counterfactual_eligible_mask] = (
                        self.counterfactual_donor_eligible[compound_batch]
                    )
                _, l, cf_l, cf_active, cf_margin, cmit_l = self.sess.run(
                    [
                        train,
                        loss,
                        counterfactual_loss,
                        counterfactual_active,
                        counterfactual_margin,
                        context_mask_loss,
                    ],
                    feed_dict=feed_dict,
                )
                if not np.isfinite(l):
                    raise ValueError('Training loss became non-finite at epoch %d batch %d: %s' % (epoch + 1, n, l))
                elapsed = time() - train_start
                batch_time = time() - batch_start
                step = min(epoch * total_batches + n + 1, total_steps)
                avg_step_time = elapsed / step
                eta = avg_step_time * (total_steps - step)
                chcr_suffix = ''
                if self.counterfactual_context['enabled']:
                    chcr_suffix = (
                        ' chcr_loss: %s chcr_active: %d chcr_margin: %s '
                        'chcr_draw: %d/%d' % (
                            cf_l,
                            int(cf_active),
                            cf_margin,
                            epoch % self.counterfactual_context['draws'] + 1,
                            self.counterfactual_context['draws'],
                        )
                    )
                cmit_suffix = ''
                if self.context_mask_training['enabled']:
                    cmit_suffix = ' cmit_loss: %s' % cmit_l
                print(
                    'training: %d/%d batch %d/%d loss: %s batch_time: %s '
                    'elapsed: %s eta: %s%s%s' % (
                        epoch + 1,
                        self.maxEpoch,
                        n + 1,
                        total_batches,
                        l,
                        format_duration(batch_time),
                        format_duration(elapsed),
                        format_duration(eta),
                        chcr_suffix,
                        cmit_suffix,
                    )
                )
            print('epoch %d/%d finished in %s' %
                  (epoch + 1, self.maxEpoch, format_duration(time() - epoch_start)))
            epochs_completed = epoch + 1

            should_validate = early_stopping['enabled'] and (
                (epoch + 1) % early_stopping['interval'] == 0
                or epoch + 1 == self.maxEpoch
            )
            if should_validate:
                validation_value = self.evaluateValidation(
                    self.fetchModelState(), early_stopping['metric']
                )
                improved, should_stop = tracker.update(validation_value, epoch + 1)
                if improved:
                    saver.save(self.sess, model_path)
                print(
                    'validation: epoch %d %s=%.6f best=%.6f best_epoch=%d stale=%d/%d%s' % (
                        epoch + 1,
                        early_stopping['metric'].upper(),
                        validation_value,
                        tracker.best_value,
                        tracker.best_epoch,
                        tracker.stale_checks,
                        tracker.patience,
                        ' improved' if improved else '',
                    )
                )
                if should_stop:
                    print('Early stopping triggered at epoch %d.' % (epoch + 1))
                    break

        if early_stopping['enabled']:
            saver.restore(self.sess, model_path)
            self.early_stopping_summary = {
                'best_epoch': tracker.best_epoch,
                'best_value': tracker.best_value,
                'epochs_completed': epochs_completed,
                'metric': early_stopping['metric'],
            }
            print(
                'Restored best validation checkpoint: epoch %d %s=%.6f' % (
                    tracker.best_epoch,
                    early_stopping['metric'].upper(),
                    tracker.best_value,
                )
            )
        else:
            self.early_stopping_summary = None

        state = self.fetchModelState()
        self.context_mask_summary = None
        if self.context_mask_training['enabled'] and self.validationData:
            side = self.context_mask_training['side']
            masked_value = self.evaluateValidation(
                state,
                early_stopping['metric'] if early_stopping['enabled'] else 'aupr',
                mask_compound=side in {'compound', 'both'},
                mask_protein=side in {'protein', 'both'},
            )
            self.context_mask_summary = {
                'side': side,
                'metric': (
                    early_stopping['metric'] if early_stopping['enabled'] else 'aupr'
                ),
                'value': masked_value,
            }
            print('CMIT masked validation: side=%s %s=%.6f' % (
                side,
                self.context_mask_summary['metric'].upper(),
                masked_value,
            ))
        self.u = state['compound']
        self.i = state['protein']
        self.u_context = state['compound_context']
        self.i_context = state['protein_context']
        self.herb_edge = state['herb_edge']
        self.weight = state['weights']
        self.global_token_attention_summary = None
        if self.global_token_attention['enabled']:
            layer_summaries = {}
            for side, tensor_rows in self.global_token_diagnostic_tensors.items():
                layer_summaries[side] = []
                for layer_index, tensors in enumerate(tensor_rows, start=1):
                    values = self.sess.run(
                        tensors, feed_dict={self.isTraining: 0}
                    )
                    row = summarize_global_token_attention(
                        values['edge_assignments'],
                        values['node_attention'],
                        values['token_embeddings'],
                    )
                    raw_gamma = float(np.asarray(self.weight[
                        '%s_global_token_gamma_%d' % (side, layer_index)
                    ]).reshape(-1)[0])
                    row.update({
                        'layer': layer_index,
                        'raw_residual_scale': raw_gamma,
                        'residual_scale': float(np.tanh(raw_gamma)),
                    })
                    layer_summaries[side].append(row)
            structure = {}
            for side, node_key, edge_key in (
                    ('hc', 'nodes', 'hyperedges'),
                    ('pd', 'nodes', 'hyperedges')):
                if not self.global_token_attention[side + '_enabled']:
                    continue
                side_structure = self.hyperedge_attention_structure[side]
                structure[side] = dict(side_structure)
                structure[side].update(global_token_attention_complexity(
                    side_structure[node_key],
                    side_structure[edge_key],
                    self.global_token_attention['tokens'],
                    self.global_token_attention['heads'],
                ))
            self.global_token_attention_summary = {
                'mode': self.global_token_attention['mode'],
                'hc_enabled': self.global_token_attention['hc_enabled'],
                'pd_enabled': self.global_token_attention['pd_enabled'],
                'tokens': self.global_token_attention['tokens'],
                'heads': self.global_token_attention['heads'],
                'temperature': self.global_token_attention['temperature'],
                'structure': structure,
                'layers': layer_summaries,
            }
            print(
                'HILGA residual scales: %s' % ', '.join(
                    '%s-L%d=%.6f' % (
                        side.upper(), row['layer'], row['residual_scale']
                    )
                    for side, rows in layer_summaries.items()
                    for row in rows
                )
            )
            global_attention_path = os.path.join(
                model_dir, 'global_token_attention.json'
            )
            with open(global_attention_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.global_token_attention_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('HILGA metadata: %s' % global_attention_path)
        self.hyperedge_attention_summary = None
        if self.hyperedge_attention['enabled']:
            parameter_summary = {}
            for side in ('hc', 'pd'):
                if not self.hyperedge_attention[side + '_enabled']:
                    continue
                parameter_summary[side] = []
                for layer in range(1, self.n_layer + 1):
                    node_weight = self.weight[
                        '%s_hyper_node_%d' % (side, layer)
                    ]
                    edge_weight = self.weight[
                        '%s_hyper_edge_%d' % (side, layer)
                    ]
                    parameter_summary[side].append({
                        'layer': layer,
                        'node_score_mean_abs': float(
                            np.mean(np.abs(node_weight))
                        ),
                        'edge_score_mean_abs': float(
                            np.mean(np.abs(edge_weight))
                        ),
                    })
            self.hyperedge_attention_summary = {
                'mode': self.hyperedge_attention['mode'],
                'hc_enabled': self.hyperedge_attention['hc_enabled'],
                'pd_enabled': self.hyperedge_attention['pd_enabled'],
                'temperature': self.hyperedge_attention['temperature'],
                'prior_scale': self.hyperedge_attention['prior_scale'],
                'structure': self.hyperedge_attention_structure,
                'parameters': parameter_summary,
            }
            print(
                'Hyperedge attention learned mean abs: %s' % ', '.join(
                    '%s-L%d node=%.6f edge=%.6f' % (
                        side.upper(),
                        row['layer'],
                        row['node_score_mean_abs'],
                        row['edge_score_mean_abs'],
                    )
                    for side, rows in parameter_summary.items()
                    for row in rows
                )
            )
            attention_path = os.path.join(
                model_dir, 'hyperedge_attention.json'
            )
            with open(attention_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.hyperedge_attention_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('Hyperedge attention metadata: %s' % attention_path)
        self.support_router_summary = None
        if self.support_router['enabled']:
            support_slope = softplus(
                np.asarray(
                    self.weight['support_router_raw_slope']
                ).reshape(-1)[0]
            )
            all_support_gates = self.supportContextGateValues(
                state, np.arange(self.num_compounds, dtype=np.int64)
            )
            seen_mask = self.compound_cp_support_degrees > 0
            zero_mask = (
                (self.compound_cp_support_degrees == 0)
                & (self.compound_context_available > 0)
            )
            self.support_router_summary = {
                'mode': self.support_router['mode'],
                'learned_slope': float(support_slope),
                'pseudo_cold': self.data.pseudo_cold_info,
                'zero_support_compounds': int(np.sum(zero_mask)),
                'zero_support_gate_mean': (
                    float(np.mean(all_support_gates[zero_mask]))
                    if np.any(zero_mask) else None
                ),
                'seen_compounds': int(np.sum(seen_mask)),
                'seen_gate_mean': (
                    float(np.mean(all_support_gates[seen_mask]))
                    if np.any(seen_mask) else None
                ),
                'gate_min': float(np.min(all_support_gates)),
                'gate_max': float(np.max(all_support_gates)),
            }
            print(
                'Support router learned slope: %.6f; zero-support gate: %s; '
                'seen gate mean: %s.' % (
                    self.support_router_summary['learned_slope'],
                    '%.6f' % self.support_router_summary[
                        'zero_support_gate_mean'
                    ] if self.support_router_summary[
                        'zero_support_gate_mean'
                    ] is not None else 'n/a',
                    '%.6f' % self.support_router_summary['seen_gate_mean']
                    if self.support_router_summary['seen_gate_mean'] is not None
                    else 'n/a',
                )
            )
            router_path = os.path.join(model_dir, 'support_router.json')
            with open(router_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.support_router_summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write('\n')
            print('Support router metadata: %s' % router_path)
        if self.use_context_interaction:
            context_weight_values = {
                'compound_disease': np.mean(np.abs(self.weight['context_compound_disease'])),
                'herb_protein': np.mean(np.abs(self.weight['context_herb_protein'])),
                'herb_disease': np.mean(np.abs(self.weight['context_herb_disease'])),
            }
            print('Context weight mean abs: C-Dctx %s, Hctx-P %s, Hctx-Dctx %s' % (
                '%.6f' % context_weight_values['compound_disease']
                if self.context_terms['compound_disease'] else 'off',
                '%.6f' % context_weight_values['herb_protein']
                if self.context_terms['herb_protein'] else 'off',
                '%.6f' % context_weight_values['herb_disease']
                if self.context_terms['herb_disease'] else 'off',
            ))
        if self.herb_context_attention['mode'] == 'target_residual_attention':
            print(
                'Target herb residual weight mean abs: %.6f' %
                np.mean(np.abs(self.weight['context_target_herb_residual']))
            )
        self.target_attention_summary = None
        if (
            self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
            and self.validationData
        ):
            validation_compounds = [
                self.data.compound[str(row[0])] for row in self.validationData
            ]
            validation_proteins = [
                self.data.protein[str(row[1])] for row in self.validationData
            ]
            validation_target_contexts, validation_attention = self.targetConditionedHerbContexts(
                state, validation_compounds, validation_proteins
            )
            entropy = -np.sum(
                validation_attention
                * np.log(np.clip(validation_attention, 1e-12, 1.0)),
                axis=1,
            )
            self.target_attention_summary = {
                'validation_pairs': len(validation_compounds),
                'expanded_incidences': int(np.sum(
                    self.compound_herb_mask[validation_compounds]
                )),
                'mean_entropy': float(np.mean(entropy)),
                'mean_max_weight': float(np.mean(np.max(validation_attention, axis=1))),
            }
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                validation_static_contexts = state['compound_context'][
                    validation_compounds
                ]
                validation_protein_embeddings = state['protein'][
                    validation_proteins
                ]
                context_delta = (
                    validation_target_contexts - validation_static_contexts
                )
                residual_logits = np.sum(
                    context_delta
                    * validation_protein_embeddings
                    * self.weight['context_target_herb_residual'],
                    axis=1,
                )
                self.target_attention_summary.update({
                    'mean_delta_norm': float(np.mean(
                        np.linalg.norm(context_delta, axis=1)
                    )),
                    'mean_abs_residual_logit': float(np.mean(
                        np.abs(residual_logits)
                    )),
                })
            print(
                'Target herb attention: validation_pairs=%d incidences=%d '
                'mean_entropy=%.6f mean_max_weight=%.6f' % (
                    self.target_attention_summary['validation_pairs'],
                    self.target_attention_summary['expanded_incidences'],
                    self.target_attention_summary['mean_entropy'],
                    self.target_attention_summary['mean_max_weight'],
                )
            )
            if self.herb_context_attention['mode'] == 'target_residual_attention':
                print(
                    'Target herb residual: mean_delta_norm=%.6f '
                    'mean_abs_logit=%.6f' % (
                        self.target_attention_summary['mean_delta_norm'],
                        self.target_attention_summary['mean_abs_residual_logit'],
                    )
                )
        os.makedirs('./results', exist_ok=True)
        if self.target_attention_summary is not None:
            attention_path = (
                './results/%s_top_herbs%s.tsv' % (
                    self.herb_context_attention['mode'], currentTime
                )
            )
            with open(attention_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    'compound_id\tprotein_id\tlabel\trank\therb_id\tattention\n'
                )
                for pair_index, row in enumerate(self.validationData):
                    compound_index = validation_compounds[pair_index]
                    active_count = int(np.sum(
                        self.compound_herb_mask[compound_index]
                    ))
                    active_weights = validation_attention[pair_index, :active_count]
                    top_positions = np.argsort(active_weights)[::-1][:3]
                    for rank, position in enumerate(top_positions, start=1):
                        herb_index = int(
                            self.compound_herb_indices[compound_index, position]
                        )
                        handle.write('%s\t%s\t%s\t%d\t%s\t%.8f\n' % (
                            row[0], row[1], row[2], rank,
                            self.data.id2herb[herb_index],
                            active_weights[position],
                        ))
            print('Target herb attention details: %s' % attention_path)
        np.savetxt('./results/herbedgeDHCN_herb_embedding' + currentTime + '.txt', self.u)
        np.savetxt('./results/diseaseedgeDHCN_disease_embedding' + currentTime + '.txt', self.i)
        if self.use_context_interaction:
            np.savetxt('./results/compound_herb_context' + currentTime + '.txt', self.u_context)
            np.savetxt('./results/protein_disease_context' + currentTime + '.txt', self.i_context)
        save_path = model_path if early_stopping['enabled'] else saver.save(self.sess, model_path)
        print("模型权重保存成功: %s" % save_path)
    def predictForPairs(self, compound_indices, protein_indices):
        zero_weight = np.zeros(self.emb_size, dtype=np.float32)
        state = {
            'compound': self.u,
            'protein': self.i,
            'compound_context': self.u_context,
            'protein_context': self.i_context,
            'herb_edge': self.herb_edge,
            'weights': self.weight,
        }
        target_compound_contexts, _ = self.targetConditionedHerbContexts(
            state, compound_indices, protein_indices
        )
        pair_compound_contexts = (
            target_compound_contexts
            if self.herb_context_attention['mode'] == 'target_attention'
            else None
        )
        residual_compound_contexts = (
            target_compound_contexts
            if self.herb_context_attention['mode'] == 'target_residual_attention'
            else None
        )
        return context_interaction_pair_scores(
            self.u,
            self.i,
            self.u_context,
            self.i_context,
            compound_indices,
            protein_indices,
            self.weight.get('context_compound_disease', zero_weight),
            self.weight.get('context_herb_protein', zero_weight),
            self.weight.get('context_herb_disease', zero_weight),
            enabled_terms=self.context_terms,
            decoder_type=self.pair_decoder['type'],
            decoder_weights=self.weight,
            pair_compound_contexts=pair_compound_contexts,
            residual_compound_contexts=residual_compound_contexts,
            target_residual_weight=self.weight.get('context_target_herb_residual'),
            herb_protein_scale=self.supportContextGateValues(
                state, compound_indices
            ),
        )

    def predictForRanking(self):
        print('hdctipredict----------------------------------------------------------------------------')
        decoder_type = self.pair_decoder['type']
        pairwise_prediction_required = (
            decoder_type == 'mlp'
            or self.herb_context_attention['mode'] in TARGET_HERB_ATTENTION_MODES
        )
        if pairwise_prediction_required:
            scores = np.empty((self.num_compounds, self.num_proteins), dtype=np.float32)
            batch_size = self.pair_decoder['prediction_batch_size']
            total_pairs = self.num_compounds * self.num_proteins
            for start in range(0, total_pairs, batch_size):
                stop = min(start + batch_size, total_pairs)
                flat_indices = np.arange(start, stop, dtype=np.int64)
                compound_indices = flat_indices // self.num_proteins
                protein_indices = flat_indices % self.num_proteins
                scores.reshape(-1)[start:stop] = self.predictForPairs(
                    compound_indices, protein_indices
                )
            return scores
        if decoder_type == 'dot':
            scores = self.u.dot(self.i.transpose())
        elif decoder_type == 'bilinear':
            scores = (self.u.dot(self.weight['decoder_bilinear'])).dot(self.i.transpose())
        else:
            raise ValueError('Unsupported pair decoder: %s' % decoder_type)

        if self.context_terms['compound_disease']:
            scores += (self.u * self.weight['context_compound_disease']).dot(
                self.i_context.transpose()
            )
        if self.context_terms['herb_protein']:
            herb_protein_scores = (
                self.u_context * self.weight['context_herb_protein']
            ).dot(
                self.i.transpose()
            )
            if self.support_router['enabled']:
                support_gates = self.supportContextGateValues(
                    {
                        'weights': self.weight,
                    },
                    np.arange(self.num_compounds, dtype=np.int64),
                )
                herb_protein_scores *= support_gates[:, None]
            scores += herb_protein_scores
        if self.context_terms['herb_disease']:
            scores += (self.u_context * self.weight['context_herb_disease']).dot(
                self.i_context.transpose()
            )
        return scores
