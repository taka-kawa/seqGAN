import numpy as np
import tensorflow as tf
import random
from dataloader import GenDataLoader, DisDataLoader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64 # バッチ一つの文章量

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda_ = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt' # 正解データ
negative_file = 'save/generator_sample.txt' # 不正解データ
eval_file = 'save/eval_file.txt' # 評価データ
generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    """
    最尤推定を使って、generatorを事前学習するが、1epoch
    @param sess tensorflowのセッション
    @param trainable_model 訓練するモデル
    @param data_loader データローダー(バッチに別れている)
    """
    # 教師との損失を格納する
    supervised_g_losses = []
    # 分けたバッチの最初から始めるためにポインタをリセット
    data_loader.reset_pointer()
    for it in range(data_loader.num_batch):
        # バッチの文章群を取得
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def main():
    """
    実験するときに回す関数

    """
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    # インスタンス生成
    gen_data_loader = GenDataLoader(BATCH_SIZE, SEQ_LENGTH)
    likelihood_data_loader = GenDataLoader(BATCH_SIZE, SEQ_LENGTH) # テスト用
    vocab_size = 5000
    dis_data_loader = DisDataLoader(BATCH_SIZE, SEQ_LENGTH)

    # Gに関して
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    with open('save/target_params.pkl', mode='rb') as f:
        target_params = pickle.load(f, encoding='latin-1')
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # the oracle model

    # Dに関して
    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim, filter_sizes=dis_filter_sizes,
                                  num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda_)

    # tfに関してGPUとか設定
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # モデル定義に基づいてモデル生成
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # 最初に、positiveな例(oracleデータからのサンプリング)の供給のためにoracleモデル(絶対正しいマン)を使う。
    # WARNING:ここでrealデータを使う場合コメントアウト
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    print('Make positive data')
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')

    #######################################################################################################
    #  generatorの事前学習
    #######################################################################################################
    print('Start pre-training')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        # 事前学習の誤差を求める
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    print('Start pre-training discriminator...')
    #######################################################################################################
    #  discriminatorの事前学習
    #######################################################################################################
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(50):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    # 途中系列から先を出力するためのrollout
    rollout = ROLLOUT(generator, 0)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            negative_epoch_file = negative_file + str(total_batch)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_epoch_file)
            dis_data_loader.load_train_data(positive_file, negative_epoch_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

    log.close()

if __name__ == '__main__':
    main()