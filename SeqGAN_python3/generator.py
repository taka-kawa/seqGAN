import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, learning_rate=0.01, reward_gamma=0.95):
        self.num_emb = num_emb  # 語彙数
        self.batch_size = batch_size  # 1バッチの文章量(64)
        self.emb_dim = emb_dim  # 分散表現の次元数
        self.hidden_dim = hidden_dim  # 隠れ層の次元
        self.sequence_length = sequence_length  # 文章の長さ(20)
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32) # 定数の定義(バッチ)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0  # 温度計数
        self.grad_clip = 5.0  # 勾配

        # 期待報酬を文章の長さ文(単語数)
        # それぞれの経過で報酬を定義
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        # 名前スコープ
        # 関連するノードをグループごとにまとめる
        # name_scopeというのもあるが、variable_scopeは変数を管理するための専用のスコープ
        with tf.variable_scope('generator'):
            # 語彙数と分散表現の次元の行列を初期化
            # 入力層の定義
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            # RNN層の定義
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            # 出力層の定義
            self.g_output_unit = self.create_output_unit(self.g_params)

        # placeholderの定義
        # プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える。
        # ジェネレーターによって生成されたシークエンス
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        # Dとrolloutからの報酬
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])

        # processed for batch
        with tf.device("/cpu:0"):
            # tf.transpose:転置行列を生成
            # tf.nn.embedding_lookup:tf.Variableの一部のみを学習させたい，かつその対象をplaceholderなどで指定したい場合
            # idと埋め込み表現変換するみたいな感じ
            # https://qiita.com/kzmssk/items/ddf2c0f956a5d26e992a
            # seq_length x batch_size x emb_dim
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])

        # 初期の隠れ層の状態を定義
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # 時系列で繰り返す
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            """
            繰り返し使用される関数

            WARNING:この関数後回し
            """
            # hidden_memory_tuple
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            # batch x vocab, logits not prob
            o_t = self.g_output_unit(h_t)
            # 出力
            log_prob = tf.log(tf.nn.softmax(o_t))
            # multinomial:0 or 1を返す(おそらく第一引数で確率的なもの)
            # 次の単語を確率的に選ぶ
            # WARNING:1の部分怖い
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            # batch x emb_dim
            # おそらく単語のembedding
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            # writeでiにvalueを書き込む
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp1, h_t, gen_o, gen_x

        # condがTrueの間bodyを繰り返す
        # https://www.tensorflow.org/api_docs/python/tf/while_loop
        # WARNING
        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        #######################################################################################################
        #  事前学習
        #######################################################################################################
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        # 事前学習
        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        # batch_size x seq_length x vocab_size
        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])

        # 事前学習の損失
        # clip_by_value:上限下限を決めている
        # 損失をバッチで平均(文章の長さ(20)*文章量(64))
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )) / (self.sequence_length * self.batch_size)

        # 訓練更新
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        #######################################################################################################
        #  教師なし訓練
        #######################################################################################################
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        # 勾配とパラメータを渡して更新
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, sess):
        """
        generateするときは入力は特にいらない
        """
        outputs = sess.run(self.gen_x)
        return outputs

    def pretrain_step(self, sess, x):
        """
        訓練だから系列データが必要
        """
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        """
        正規分布なランダム値で初期化
        @param shape 行列の次元のリスト
        """
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        """
        RNNのユニットの定義
        入力層と隠れ層の重みを定義
        @param params パラメータ(入力層とのつなぎ目のパラメータ)
        """
        # 入力層
        # 入力ゲート
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))
        # 忘却ゲート
        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))
        # 出力ゲート
        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))
        # セルの出力
        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc,
        ])

        def unit(x, hidden_memory_tm1):
            """
            RNNのセル
            自前でやってるっぽい
            """
            # 分解できるっぽい
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # 入力ゲート
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) + tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )
            # 忘却ゲート
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) + tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )
            # 出力ゲート
            o = tf.nn.sigmoid(
                tf.matmul(x, self.Wog) + tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )
            # 新しいメモリセル
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) + tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )
            # メモリセルの出力
            c = f * c_prev + i * c_

            # 現在の隠れ層の状態
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        """
        出力層のユニット定義
        """
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state:batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwards):
        return tf.train.AdamOptimizer(*args, **kwards)



