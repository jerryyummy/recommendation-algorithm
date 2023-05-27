# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from paddle.regularizer import L2Decay

class YYLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes,
                 bot_layer_sizes,top_layer_sizes,layer_sizes_fic,self_interaction=False):
        super(YYLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        self.self_interaction = self_interaction
        self.bot_layer_sizes = bot_layer_sizes
        self.top_layer_sizes = top_layer_sizes
        self.layer_sizes_fic = layer_sizes_fic

        self.wide_part = paddle.nn.Linear( # handle dense feature
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))))

        self.bot_mlp = MLPLayer(
            input_shape=dense_feature_dim,
            units_list=bot_layer_sizes,
            activation="relu")
        
        self.embedding = paddle.nn.Embedding( # 处理稀疏特征
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                     sparse_num_field,
                       layer_sizes)

        self.fic = FIC(sparse_feature_dim,
                       1 + sparse_num_field, layer_sizes_fic)

        self.top_mlp = MLPLayer(
            input_shape= 3,
            units_list=top_layer_sizes)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

        self.fc = paddle.nn.Linear(
            # in_features=self.layer_sizes[-1] + self.sparse_num_field *
            # self.sparse_feature_dim + self.dense_feature_dim,
            in_features = top_layer_sizes[-1],
            out_features = 1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 /
                    math.sqrt(self.top_layer_sizes[-1] + self.sparse_num_field +
                              self.dense_feature_dim))))

    def forward(self, sparse_inputs, dense_inputs):
        y_wide = self.wide_part(dense_inputs) # [2, 1]
        x = self.bot_mlp(dense_inputs)
        batch_size, d = x.shape # [2, 16]
        sparse_embs = [] # len(sparse_embs) = 26
        for s_input in sparse_inputs:
            emb = self.embedding(s_input) #[2,1,16]
            emb = paddle.reshape(
                emb, shape=[batch_size, self.sparse_feature_dim]) # [2,16]
            sparse_embs.append(emb)
        
        # concat dense embedding and sparse embeddings, (batch_size, (sparse_num_field + 1), embedding_size)
        T = paddle.reshape(
            paddle.concat(
                x=sparse_embs + [x], axis=1),
            (batch_size, self.sparse_num_field + 1, d)) # [2, 27, 16]
        y_fic = self.fic(T) # [2,1]

        y_interaction = paddle.concat([y_fic] + [y_wide], axis=1) # [2,2]

        y_dnn = self.dnn(paddle.reshape( # y_dnn.shape = [2, 1]
            paddle.concat( # [2, 26, 16]
                x=sparse_embs, axis=1),
            (batch_size, self.sparse_num_field, d)))

        R = paddle.concat([y_dnn] + [y_interaction], axis=1) # [2,3]
        y = self.top_mlp(R)

        logit = self.fc(y)
        predict = F.sigmoid(logit)

        return predict


class FIC(nn.Layer):
    def __init__(self, sparse_feature_dim, num_field, layer_sizes_fic):
        super(FIC, self).__init__()
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.layer_sizes_fic = layer_sizes_fic # [512, 256, 128]
        self.cnn_layers = []
        last_s = self.num_field # 26
        for i in range(len(layer_sizes_fic)):
            _conv = nn.Conv2D(
                in_channels=last_s * self.num_field,
                out_channels=layer_sizes_fic[i],
                kernel_size=(1, 1),
                weight_attr=paddle.ParamAttr(
                    regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(last_s * self.num_field))),
                bias_attr=False)
            last_s = layer_sizes_fic[i]
            self.add_sublayer('cnn_%d' % i, _conv)
            self.cnn_layers.append(_conv)
        tmp_sum = sum(self.layer_sizes_fic)
        self.fic_linear = paddle.nn.Linear(
            in_features=tmp_sum,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                regularizer=L2Decay(coeff=0.0001),
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(tmp_sum))))
        self.add_sublayer('cnn_fc', self.fic_linear)

    def forward(self, feat_embeddings): # [2, 27, 16]
        Xs = [feat_embeddings]
        last_s = self.num_field
        #m = paddle.nn.Dropout(p=0.5)
        
        for s, _conv in zip(self.layer_sizes_fic, self.cnn_layers):
            # calculate Z^(k+1) with X^k and X^0
            X_0 = paddle.reshape(
                x=paddle.transpose(Xs[0], [0, 2, 1]), # [2, 16, 27]
                shape=[-1, self.sparse_feature_dim, self.num_field, 
                       1])  # None, embedding_size, num_field, 1
            X_k = paddle.reshape(
                x=paddle.transpose(Xs[-1], [0, 2, 1]),
                shape=[-1, self.sparse_feature_dim, 1,
                       last_s])  # None, embedding_size, 1, last_s
            Z_k_1 = paddle.matmul(
                x=X_0, y=X_k)  # None, embedding_size, num_field, last_s

            # compresses Z^(k+1) to X^(k+1)
            Z_k_1 = paddle.reshape(
                x=Z_k_1,
                shape=[-1, self.sparse_feature_dim, last_s * self.num_field
                       ])  # None, embedding_size, last_s*num_field
            Z_k_1 = paddle.transpose(
                Z_k_1, [0, 2, 1])  # None, s*num_field, embedding_size
            Z_k_1 = paddle.reshape(
                x=Z_k_1,
                shape=[
                    -1, last_s * self.num_field, 1, self.sparse_feature_dim
                ]
            )  # None, last_s*num_field, 1, embedding_size  (None, channal_in, h, w)

            X_k_1 = _conv(Z_k_1)

            X_k_1 = paddle.reshape(
                x=X_k_1,
                shape=[-1, s,
                       self.sparse_feature_dim])  # None, s, embedding_size
            #X_k_1 = m(X_k_1)
            Xs.append(X_k_1)
            last_s = s
        # sum pooling
        y_fic = paddle.concat(
            x=Xs[1:], axis=1)  # None, (num_field++), embedding_size
        y_fic = paddle.sum(x=y_fic, axis=-1)  # None, (num_field++)i
        tmp_sum = sum(self.layer_sizes_fic)
        y_fic = self.fic_linear(y_fic)
        y_fic = paddle.sum(x=y_fic, axis=-1, keepdim=True)

        return y_fic
        
class DNN(paddle.nn.Layer): # handle sparse feature
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn

class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, activation=None,
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.mlp = []
        self.activation = activation

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(
                    in_features=unit,
                    out_features=units_list[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.TruncatedNormal(
                            std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(
                    in_features=unit,
                    out_features=units_list[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.TruncatedNormal(
                            std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if self.activation is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs
