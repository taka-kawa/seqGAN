import numpy as np


class GenDataLoader():
    """
    generatorのデータを扱うクラス
    """
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.token_stream = []
        self.sequence_length = sequence_length


    def create_batches(self, data_file):
        """
        データをバッチに分ける
        @param data_file バッチで分けたいファイルのパス
        """
        self.tooken_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                # lineの前後の空白、改行を除去→空白で区切る(単語ごとに)
                line = line.strip()
                line = line.split()
                # 単語のidに直してる
                parse_line = [int(x) for x in line]
                # WARNING:よくわからん
                if len(parse_line) == self.sequence_length:
                    self.token_stream.append(parse_line)

        # できたバッチの数
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # WARNING:よくわからん
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        # 先頭からバッチに分ける
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)

        # バッチの何番目を指すかのpointer
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        # 使い回す可能性があるためあまりをとってる
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataLoader():
    """
    discriminatorのデータを扱うクラス
    """
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sentences = np.array([])
        self.labels = np.array([])



    def load_train_data(self, positive_file, negative_file):
        """
        discriminatorを訓練するためのデータをロードする
        @param positive_file 実世界データのパス
        @param negative_file generatorが作り出したデータパス
        """
        # ロードデータ
        positive_examples = []
        negative_examples = []
        # positiveファイルロード
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                # WARNING:よくわからん
                if len(parse_line) == self.sequence_length:
                    negative_examples.append(parse_line)
        # ポジティブとネガティブを結合
        self.sentences = np.array(positive_examples + negative_examples)

        # ラベル生成
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # データシャッフル
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # バッチに分ける
        # バッチの数
        self.num_batch = int(len(self.labels) / self.batch_size)
        # WARNING:よくわからん
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        # 文章とラベルのバッチを分ける
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        # バッチの何番目を指すかのpointer
        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

