import keras
import os
# Create your first MLP in Keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy
import pickle
# pickle substitution for large files
import joblib
import argparse

# This is for not eating up the whole RAM
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

import pandas as pd
import time
# tf.reset_default_graph()
# fix random seed for reproducibility


from numpy.random import seed
seed(1) # 1
from tensorflow import set_random_seed
set_random_seed(2) # 2

from utils.models import get_model, get_conv_model

# for deep explain module
from keras import backend as K
from deepexplain.tensorflow import DeepExplain

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# load pima indians dataset

def load_data():
    # deprecated for now
    data_dir = '../data'
    X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized')

    # loading data
    f = open(os.path.join(X_data_dir, 'X_train.pickle'), 'rb')
    X_train = pickle.load(f)
    f.close()

    f = open(os.path.join(X_data_dir, 'X_val.pickle'), 'rb')
    X_val = pickle.load(f)
    f.close()

    f = open(os.path.join(X_data_dir, 'X_test.pickle'), 'rb')
    X_test = pickle.load(f)
    f.close()

    Y_data_dir = os.path.join(data_dir, 'COPDGene_Freeze1_RNAseq_samples', 'SmokCigNow')
    f = open(os.path.join(Y_data_dir, 'Y_train.pickle'), 'rb')
    Y_train = pickle.load(f)
    f.close()

    f = open(os.path.join(Y_data_dir, 'Y_val.pickle'), 'rb')
    Y_val = pickle.load(f)
    f.close()

    f = open(os.path.join(Y_data_dir, 'Y_test.pickle'), 'rb')
    Y_test = pickle.load(f)
    f.close()

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def load_data_all(data_dir, choice):
    data_dir = data_dir
    if choice == 'exon':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_normalized')
    elif choice == 'gene':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_normalized')
    elif choice == 'transcript':
        X_data_dir = os.path.join('/home/zifeng/Research/COPD/data_stranded2', 'COPDGene_Freeze3_RNAseq_transcripts_logCPM_normalized')

    # loading data
    f = open(os.path.join(X_data_dir, 'X_all.pickle'), 'rb')
    X_all = joblib.load(f)
    f.close()

    Y_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_samples', 'SmokCigNow')
    f = open(os.path.join(Y_data_dir, 'Y_all.pickle'), 'rb')
    Y_all = joblib.load(f)
    f.close()

    return X_all, Y_all

def load_data_new(data_dir, choice, data_type):
    data_dir = data_dir
    if choice == 'exon':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_{}'.format(data_type))
    elif choice == 'gene':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_{}'.format(data_type))
    elif choice == 'transcript':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_transcripts_logCPM_{}'.format(data_type))
    else:
        return None, None, None, None

    f = open(os.path.join(X_data_dir, 'X_train.pickle'), 'rb')
    X_train = joblib.load(f)
    f.close

    f = open(os.path.join(X_data_dir, 'X_val.pickle'), 'rb')
    X_val = joblib.load(f)
    f.close

    f = open(os.path.join(X_data_dir, 'X_test.pickle'), 'rb')
    X_test = joblib.load(f)
    f.close

    # here merge X_val and X_test
    X_test = numpy.concatenate((X_test, X_val), axis=0)  

    Y_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_samples', 'SmokCigNow')
    f = open(os.path.join(Y_data_dir, 'Y_train.pickle'), 'rb')
    Y_train = joblib.load(f)
    f.close

    f = open(os.path.join(Y_data_dir, 'Y_val.pickle'), 'rb')
    Y_val = joblib.load(f)
    f.close

    f = open(os.path.join(Y_data_dir, 'Y_test.pickle'), 'rb')
    Y_test = joblib.load(f)  
    f.close

    Y_test = numpy.concatenate((Y_test, Y_val), axis=0)  

    return X_train, X_test, Y_train, Y_test

def load_data_final_test(data_dir, choice, data_type):
    data_dir = data_dir
    if choice == 'exon':
        X_data_dir = os.path.join(data_dir, 'COPDGeneTest_Freeze3_RNAseq_exonicParts_logCPM_{}'.format(data_type))
    elif choice == 'gene':
        X_data_dir = os.path.join(data_dir, 'COPDGeneTest_Freeze3_RNAseq_genes_logCPM_{}'.format(data_type))
    elif choice == 'transcript':
        X_data_dir = os.path.join(data_dir, 'COPDGeneTest_Freeze3_RNAseq_transcripts_logCPM_{}'.format(data_type))
    else:
        return None
    
    f = open(os.path.join(X_data_dir, 'X_final_test.pickle'), 'rb')
    X_final_test = joblib.load(f)
    f.close

    return X_final_test

def load_data_emph(data_dir, choice):
    '''
    This function is for loading emphysema data...
    '''
    data_dir = '../data_emph'
    choice = 'gene'
    if choice == 'exon':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_normalized')
    elif choice == 'gene':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_normalized')
    elif choice == 'transcript':
        X_data_dir = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_transcripts_logCPM_normalized')

    f = open(os.path.join(X_data_dir, 'X.pickle'), 'rb')
    X_train = joblib.load(f)
    f.close

    X_test = X_train
    f.close

    f = open(os.path.join(data_dir, 'emph_bin_5.pickle'), 'rb')
    Y_train = joblib.load(f)
    f.close

    Y_test = Y_train
    f.close

    return X_train, X_test, Y_train, Y_test

def get_weights_stats(model, eps=0.05):
    all_weights = model.get_weights()
    total_sum = 0
    total_abs = 0
    total_size = 0
    non_zero = 0
    for weights in all_weights:
        total_size += weights.size
        total_abs += numpy.sum(numpy.abs(weights))
        total_sum += numpy.sum(weights)
        non_zero += numpy.sum(weights > eps)
    return non_zero, total_sum, total_abs, total_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nodes', default='0', type=str, help='Specify the hidden nodes of 1st layer')
    parser.add_argument('--emph', default=True, type=str2bool, help='Specify use emph data or not')
    parser.add_argument('--save_dir', default='/home/zifeng/Research/COPD/models', type=str, help='Specify where to save the models')
    parser.add_argument('--data_dir', default='/home/zifeng/Research/COPD/data_last', type=str, help='Specify data set directory')
    parser.add_argument('--choice', default="exon", type=str, help='Specify the data type you want to choose')
    parser.add_argument('--data_type', default="normalized", type=str, help='choose from normalized or covAdjusted')
    parser.add_argument('--exp_name', default="exp1", type=str, help='Specify the experiment name')
    parser.add_argument('--patience', default=50, type=int, help='Specify # of epochs')
    #parser.add_argument('--features', default='255124', type=int, help='Specify dim of input features')
    parser.add_argument('--weights_stats', default=True, type=str2bool, help='Specify if calculate weigths stats or not')
    parser.add_argument('--epsilon', default=0.02, type=float, help='Specify how close is to zero')
    parser.add_argument('--epoch', default=100, type=int, help='Specify # of epochs')
    parser.add_argument('--batch_size', default=415, type=int, help='Specify # of mini batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='Specify learning rate')
    parser.add_argument('--l1', default=0.01, type=float, help='Specify l1 regularizer')
    parser.add_argument('--l2', default=0, type=float, help='Specify l2 regularizer')
    parser.add_argument('--dropout', default=0.2, type=float, help='Specify dropout')
    parser.add_argument('--model_type', default='mlp', type=str, help='Specify model you want to use, for now mlp and conv')
    parser.add_argument('--cross_val', default=5, type=int, help='Specify if using cross validation or not, if 1, no cross validation, if other than 1, say x, then x-fold')
    parser.add_argument('--feature_map', default=False, type=str2bool, help='Specify if using the important exons')
    parser.add_argument('--normalize', default=True, type=str2bool, help='Specify if do normalization')
    parser.add_argument('--trans_supervision', default=True, type=str2bool, help='Specify if add transcripts as intermediate supervision')
    parser.add_argument('--map_layer', default=False, type=str2bool, help='Specify if using the map_layer, if so, we need to import the linking_matrix')
    parser.add_argument('--map_type', default='', type=str, help='Specify if using the map_layer, if so, we need to import the linking_matrix')
    parser.add_argument('--map_dir', default='mapping_data_5gene', type=str, help='Specify where to load the mapping datas')
    parser.add_argument('--save_iso', default=False, type=str2bool, help='Specify if save the intermediate estimation of isoforms')
    parser.add_argument('--fs_layer', default=False, type=str2bool, help='Specify if using the feature selection layer')
    parser.add_argument('--save_log', default='./new_log.txt', type=str, help='Specify where to save the log')
    ## This arg is for final test
    parser.add_argument('--final_test', default=False, type=str2bool, help='Specify if create final test result')
    parser.add_argument('--final_test_dir', default='/home/zifeng/Research/COPD/data_last_test', type=str, help='Specify where the final test located')
    parser.add_argument('--final_test_save_dir', default='', type=str)
    ## This is for loading model and train
    parser.add_argument('--no_train', action='store_true', default=False, help="if specified, no train needed, just load a pretrained model")
    parser.add_argument('--load_model_path', default='', type=str, help="where to load the trained model")
    ## This is for timing benchmarking
    parser.add_argument('--timing', action='store_true', default=False)
    ## This is for DeepExplain
    parser.add_argument('--explain', action='store_true', default=False)
    ## This is for loading uncorrelated gene or not
    parser.add_argument('--uncorrelated_gene', action='store_true', default=False)
    args = parser.parse_args()
    


    with open(args.save_log, 'a+') as f:
        f.write('\n')
        for key,value in vars(args).items():
            f.write(key + ' : ' + str(value) + '\n')


    #if args.cross_val == 1:
        '''if not args.emph:
            X_all, X_eval, Y_all, Y_eval = load_data_new(args.data_dir, args.choice)
        else:
            X_all, X_eval, Y_all, Y_eval = load_data_emph(args.data_dir, args.choice)'''
    #else:
        #X_all, Y_all = load_data_all(args.data_dir, args.choice)
        # now we are using new data
    if not args.emph:
        X_all, X_eval, Y_all, Y_eval = load_data_new(args.data_dir, args.choice, args.data_type)
    else:
        X_all, X_eval, Y_all, Y_eval = load_data_emph(args.data_dir, args.choice)

    if args.final_test:
        X_final_test = load_data_final_test(args.final_test_dir, args.choice, args.data_type)

    if args.choice.lower() == 'exon':
        if args.feature_map:
            #f = open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'exon_list2.pickle'), 'rb')
            f = open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir, 'exon_list.pickle'), 'rb')
            feature_list = pickle.load(f)
            f.close()
            X_all = X_all[:, feature_list]
            X_eval = X_eval[:, feature_list]

            if args.final_test:
                X_final_test = X_final_test[:, feature_list]

            if args.trans_supervision:
                #f = open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'tx_list2.pickle'), 'rb')
                f = open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir,'tx_list.pickle'), 'rb')
                feature_list = pickle.load(f)
                f.close()
                tx_all, tx_eval, ty_all, ty_eval = load_data_new(args.data_dir, 'transcript', args.data_type)
                tx_all = tx_all[:, feature_list]
                tx_eval = tx_eval[:, feature_list]
                if args.normalize:
                    tx_all_mean = numpy.mean(tx_all, axis = 0)
                    tx_all_var = numpy.var(tx_all, axis = 0)
                    #print(tx_all_var)
                    #tx_all_var = numpy.sqrt(tx_all_var)
                    tx_all = (tx_all - tx_all_mean) / tx_all_var
                    tx_eval = (tx_eval - tx_all_mean) / tx_all_var
            else:
                tx_all = None
        else:
            X_all = X_all
            X_eval = X_eval



    elif args.choice.lower() == 'gene':
        if args.feature_map:
            if args.uncorrelated_gene:
                print("loading uncorrelated genes!")
                f = open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir, 'gene_list_uncorrelated.pickle'), 'rb')
            else:
                f = open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir, 'gene_list.pickle'), 'rb')
            #feature_list = pickle.load(open(os.path.join('../emph_pvalue_list', 'gene_list.pickle'),'rb'))
            feature_list = pickle.load(f)
            ###### HARD CODE!!!!!
            #feature_list = feature_list[0:5000]
            ###### HARD CODE!!!!!
            f.close()
            X_all = X_all[:, feature_list]
            X_eval = X_eval[:, feature_list]

            if args.final_test:
                X_final_test = X_final_test[:, feature_list]

        else:
            X_all = X_all
            X_eval = X_eval

    elif args.choice.lower() == 'transcript':
        if args.feature_map:
            #f = open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'tx_list2.pickle'), 'rb')
            f = open(os.path.join('/home/zifeng/Research/COPD/',args.map_dir , 'tx_list.pickle'), 'rb')
            feature_list = pickle.load(f)
            f.close()
            X_all = X_all[:, feature_list]
            X_eval = X_eval[:, feature_list]

            if args.final_test:
                X_final_test = X_final_test[:, feature_list]

        else:
            X_all = X_all
            X_eval = X_eval

    elif args.choice.lower() == 'exon+transcript':
        # exon
        X_all, X_eval, Y_all, Y_eval = load_data_new(args.data_dir, 'exon', args.data_type)
        if args.feature_map:
            #f = open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'tx_list2.pickle'), 'rb')
            f = open(os.path.join('/home/zifeng/Research/COPD/',args.map_dir , 'exon_list.pickle'), 'rb')
            feature_list = pickle.load(f)
            f.close()
            X_all = X_all[:, feature_list]
            X_eval = X_eval[:, feature_list]

            if args.final_test:
                X_final_test = load_data_final_test(args.final_test_dir, 'exon', args.data_type)
                X_final_test_exon = X_final_test[:, feature_list]

        else:
            X_all = X_all
            X_eval = X_eval
        X_all_exon = X_all
        X_eval_exon = X_eval

        # transcript
        X_all, X_eval, Y_all, Y_eval = load_data_new(args.data_dir, 'transcript', args.data_type)
        if args.feature_map:
            #f = open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'tx_list2.pickle'), 'rb')
            f = open(os.path.join('/home/zifeng/Research/COPD/',args.map_dir , 'tx_list.pickle'), 'rb')
            feature_list = pickle.load(f)
            f.close()
            X_all = X_all[:, feature_list]
            X_eval = X_eval[:, feature_list]

            if args.final_test:
                X_final_test = load_data_final_test(args.final_test_dir, 'transcript', args.data_type)
                X_final_test_transcript = X_final_test[:, feature_list]

        else:
            X_all = X_all
            X_eval = X_eval

        X_all = numpy.concatenate([X_all_exon, X_all], axis=1)
        X_eval = numpy.concatenate([X_eval_exon, X_eval], axis=1)
        
        if args.final_test:
            X_final_test = numpy.concatenate([X_final_test_exon, X_final_test_transcript], axis=1)


    input_dim = X_all.shape[1]

    if args.normalize:
        X_all_mean = numpy.mean(X_all, axis = 0)
        X_all_var = numpy.var(X_all, axis = 0)
        #X_all_var = numpy.sqrt(X_all_var)
        X_all = (X_all - X_all_mean) / X_all_var
        X_eval = (X_eval - X_all_mean) / X_all_var

        if args.final_test:
            X_final_test = (X_final_test - X_all_mean) / X_all_var



        # This line is just for conv models
        if args.model_type == 'conv':
            X_all = numpy.expand_dims(X_all, axis=2)


    # print(X_all.shape)
    # print(X_all[0])
    # print(X_all[1])
    # print(X_all[2])
    # exit(0)

    if args.map_layer:
        #print("Linking Matrix Added")
        #with open(os.path.join('/home/zifeng/Research/COPD/mapping_data_new', 'linking_matrix2.pickle'), 'rb') as f:
        if args.map_type == "exon2transcript":
            mat_name = 'linking_matrix.pickle'
        elif args.map_type == "exon2gene":
            mat_name = 'eg_linking_matrix.pickle'
        with open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir, mat_name), 'rb') as f:
            linking_matrix = pickle.load(f)
    else:
        linking_matrix = None


    print("This experiment is for {}".format(args.choice))
    print(X_all.shape)

    #input_dim = X_train.shape[1]
    hyper_params = {
        'l1': args.l1,
        'l2': args.l2,
        'dropout': args.dropout,
        'map_layer' : args.map_layer,
        'linking_matrix' : linking_matrix,
        'trans_supervision' : args.trans_supervision,
        'fs_layer' : args.fs_layer,
        'save_iso' : args.save_iso
    }


    layer_params = args.nodes.split()
    node_str = "_".join(layer_params)
    layer_params = [int(x) for x in layer_params if int(x) > 0]

    print('Here is the number of each layer:')
    for i, num in enumerate(layer_params):
        print("Layer {} has {} nodes".format(i, num))

    if args.model_type == 'mlp':
        if not args.save_iso:
            model = get_model(input_dim, layer_params=layer_params, hyper_params=hyper_params)
        else:
            model, model_iso = get_model(input_dim, layer_params=layer_params, hyper_params=hyper_params)
    elif args.model_type == 'conv':
        model = get_conv_model(input_dim)
    # Compile model
    if not args.trans_supervision:
        #adam = keras.optimizers.Nadam(lr=args.lr, beta_1=0.9, beta_2=0.999)

        adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # sgd = SGD(lr=100, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Fit the model
    else:
        adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(optimizer=adam, loss={'out1':'mean_squared_error', 'out2':'binary_crossentropy'}, metrics={'out1':'mae', 'out2': 'accuracy'}, loss_weights=[10.0, 1.0])


        earlystop_callback = EarlyStopping(monitor='val_out2_acc', min_delta=0, patience=(args.epoch*4/5), verbose=1, mode='auto')

    # have a look at model parameters
    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()

    # for name, weight in zip(names, weights):
    #     print(name, weight.shape)
    # exit(0)

    save_whole_path = os.path.join(args.save_dir, args.exp_name, args.choice)
    if not os.path.exists(save_whole_path):
        os.makedirs(save_whole_path)

    # make directories for every fold
    if args.cross_val > 1:
        for i in range(1, args.cross_val+1):
            if not os.path.exists(os.path.join(save_whole_path, '{}-fold'.format(i))):
                os.mkdir(os.path.join(save_whole_path, '{}-fold'.format(i)))

    saved_init_weights = model.get_weights()
    if args.cross_val == 1:
        # First use all training data available to train
        if args.timing:
            start = time.time()
            model.fit(X_all, Y_all, epochs=1, batch_size=args.batch_size, verbose=1)
            duration = time.time() - start
            print("Single Epoch {}, 40 Epochs {}".format(duration, 40*duration))
            exit(0)
        if not args.no_train:
            checkpoint_filepath = os.path.join(save_whole_path, '{}_{}_map_{}_{}_fs_{}_weights.h5'.format(args.choice, node_str, str(args.map_layer), str(args.map_type), str(args.fs_layer)))
            model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_acc',
            mode='max',
            save_best_only=True)
            model.fit(X_all, Y_all, validation_data=[X_eval, Y_eval], epochs=args.epoch, batch_size=args.batch_size, verbose=1, callbacks=[model_checkpoint_callback])
            # load best model
            model.load_weights(checkpoint_filepath)

        # else load the trained model weights
        else:
            model.load_weights(args.load_model_path)
        # evaluate the best model
        
        

        scores = model.evaluate(X_eval, Y_eval)
        Y_pred = model.predict(X_eval)

        Y_pred_ = (Y_pred > 0.5).astype(numpy.int32)
        #print(Y_pred_.shape)
        Y_eval = numpy.array(Y_eval)
        Y_eval = numpy.reshape(Y_eval, (Y_eval.shape[0], 1))
        #print(Y_eval.shape)
        acc = numpy.sum(Y_pred_ == Y_eval) / X_eval.shape[0]
        recall = recall_score(Y_eval, Y_pred_)
        precision = precision_score(Y_eval, Y_pred_)
        f1 = f1_score(Y_eval, Y_pred_)
        auc = roc_auc_score(Y_eval, Y_pred)
        # print model.predict(X_test)
        # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
        print("Eval acc is {}".format(acc))
        print("Eval recall is {}".format(recall))
        print("Eval precision is {}".format(precision))
        print("Eval f1 is {}".format(f1))
        print("Eval auc is {}".format(auc))
        feature_num = X_all.shape[1]
        with open(args.save_log, 'a+') as f:
            f.write("[{}] {} with features:{}, lr:{}, epoch:{}\nacc : {}\nrecall : {}\nprecision : {}\nf1 : {}\nauc : {}\n".format(args.choice, args.nodes, feature_num, args.lr, args.epoch, acc, recall, precision, f1, auc))
        
        if args.fs_layer and (not args.no_train):
            fs_layer_weights = model.get_weights()[0]
            # for debugging usage
            if False:
                shape = fs_layer_weights.shape
                print(shape)
                fs_layer_weights_abs = numpy.abs(fs_layer_weights)
                print(numpy.max(fs_layer_weights_abs))
                print(numpy.min(fs_layer_weights_abs))
                fs_layer_weights_non_zero = fs_layer_weights_abs > 0.02
                fs_layer_weights_non_zero_indices = numpy.where(fs_layer_weights_abs > 0.02)
                with open('suprise_list002_5_geme.pickle', 'wb') as f:
                    pickle.dump(fs_layer_weights_non_zero_indices, f)

                print(numpy.sum(fs_layer_weights_non_zero))
            print("Feature score saved!")
            feature_score_path = os.path.join(save_whole_path, '{}_{}_map_{}_{}_feature_score.pickle'.format(args.choice, node_str, str(args.map_layer), str(args.map_type)))
            with open(feature_score_path, 'wb') as f:
                pickle.dump(fs_layer_weights, f)

        if args.save_iso and (not args.no_train):
            dir_name = args.choice + '_map_' + str(args.map_layer) + '_fs_' + str(args.fs_layer) + '_' + args.nodes
            dir_name = os.path.join('./create_est_iso', dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            Y_eval_to_save = os.path.join(dir_name, 'Y_eval.npy')
            Y_pred_to_save = os.path.join(dir_name, 'Y_pred.npy')
            numpy.save(Y_eval_to_save, Y_eval)
            numpy.save(Y_pred_to_save, Y_pred)
            print("Saving the isoform estimation, as well as all raw prediction value!")
            # do the preprocessing again! for all X!
            X_complete, Y_complete = load_data_all(args.data_dir, args.choice)
            # we definitely use feature map here
            f = open(os.path.join('/home/zifeng/Research/COPD/', args.map_dir, 'exon_list.pickle'), 'rb')
            feature_list = pickle.load(f)
            f.close()
            X_complete = X_complete[:, feature_list]
            if args.normalize:
                X_complete = (X_complete - X_all_mean) / X_all_var
            
            # Get all y first
            Y_pred_complete = model.predict(X_complete)
            joblib.dump(Y_pred_complete, open(os.path.join(dir_name, 'Y_pred_all.pkl'),'wb')) 
            print("Predictions for y saved!")
            # Get all iso then
            iso_est_complete = model_iso.predict(X_complete)
            joblib.dump(iso_est_complete, open(os.path.join(dir_name, 'iso_est_all.pkl'),'wb'))
            print("Isoform estimations saved!")
        
        if not args.no_train:
        # save the whole model to corresponding place
            print("Model weights saved!")
            model.save_weights(os.path.join(save_whole_path, '{}_{}_map_{}_{}_fs_{}_weights.h5'.format(args.choice, node_str, str(args.map_layer), args.map_type, str(args.fs_layer))))
        # The proportion of one is 35%
        #op_train = sum(Y_all) / len(Y_all)
        #print("one proportion in train : {}".format(op_train))
        #op_test = numpy.sum(Y_eval) / Y_eval.shape[0]
        #print("one proportion in train : {}".format(op_test))

        #with open(args.save_log, 'a+') as f:
        if args.explain:
            with DeepExplain(session=K.get_session()) as de:
                input_tensor = model.layers[0].input
                # target_tensor = model.layers[-1].output
                # replicate the model
                fModel = Model(inputs=input_tensor, outputs = model.layers[-1].output)
                target_tensor = fModel(input_tensor)
                xs = X_all
                ys = Y_all[:, None]
                attributions_sal = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
                # save the attributions somewhere
                saliency_path = os.path.join(save_whole_path, '{}_{}_map_{}_{}_saliency.pickle'.format(args.choice, node_str, str(args.map_layer), str(args.map_type)))
                with open(saliency_path, 'wb') as f:
                    pickle.dump(attributions_sal, f)


    else:
        # using K-fold cross validation
        results = []
        eval_results = []
        skf = StratifiedKFold(n_splits=args.cross_val, random_state=1, shuffle=False)
        i = 0
        for train_index, val_index in skf.split(X_all, Y_all):
            i += 1
            model.set_weights(saved_init_weights)
            print("Here is the {}-th training in {}-fold CV".format(i, args.cross_val))
            X_train = X_all[train_index]
            X_val = X_all[val_index]
            Y_train = Y_all[train_index]
            Y_val = Y_all[val_index]

            # checkpoint = ModelCheckpoint(os.path.join(save_whole_path, '{}-fold'.format(i), 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            if not args.trans_supervision:
                model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epoch, batch_size=args.batch_size, verbose=1) #checkpoint])
                scores = model.evaluate(X_val, Y_val)
                results.append(scores[1])
                # Here we use the hold out dataset as a final test
                eval_score = model.evaluate(X_eval, Y_eval)
                eval_results.append(eval_score[1])
            else:
                tx_train = tx_all[train_index]
                tx_val = tx_all[val_index]
                model.fit(X_train, [tx_train, Y_train], validation_data=(X_val, [tx_val, Y_val]), epochs=args.epoch, batch_size=args.batch_size, verbose=1, callbacks=[earlystop_callback])
                scores = model.evaluate(X_val, [tx_val, Y_val])
                results.append(scores[-1])
                #print(scores)
                #break
                # Here we use the hold out dataset as a final test
                eval_score = model.evaluate(X_eval, [tx_eval, Y_eval])
                eval_results.append(eval_score[-1])
            # print("The validation result for the {}-th CV is {.2f}".format(i, scores[1]))
            # check weights that are close to zero here
            #get_sum_weights()
            #get_zero_proportion()
        
        print(results)
        print(eval_results)
        # should save it in a file somewhere for further plotting
        feature_num = X_all.shape[1]
        avg = lambda lst : sum(lst) / float(len(lst))
        with open(args.save_log, 'a+') as f:
            f.write("[{}] {} with features:{}, lr:{}, epoch:{}\ncv_result:{}, avg : {}\nev_result:{}, avg : {}\n".format(args.choice, args.nodes, feature_num, args.lr, args.epoch, results, avg(results), eval_results, avg(eval_results)))
            if args.weights_stats:
                non_zero, total_sum, total_abs, total_size = get_weights_stats(model, eps=args.epsilon)
                stats_str = 'None zero: {}, Sum: {}, Abs Sum: {}, Size: {}'.format(non_zero, total_sum, total_abs, total_size)
                stats_str_norm = 'None zero: {}, Sum: {}, Abs Sum: {}'.format(non_zero/total_size, total_sum/total_size, total_abs/total_size)
                print(stats_str)
                print(stats_str_norm)
                f.write(stats_str + '\n' + stats_str_norm + '\n')
            


    
    if args.final_test:
        print("Starting doing final test!")
        # load test set first
        Y_final_test = model.predict(X_final_test)
        Y_final_test = numpy.squeeze(Y_final_test)
        # save it as csv!
        final_test_save_dir = args.final_test_save_dir #'/home/zifeng/Research/COPD/rebuttal_test_results'
        if not os.path.exists(final_test_save_dir):
            os.makedirs(final_test_save_dir)

        Y_final_test = list(Y_final_test)
        final_result_dict = {'Smoking Probability' : Y_final_test}
        final_result_df = pd.DataFrame(final_result_dict, columns=['Smoking Probability'])
        tsv_name = args.choice + '_map_' + str(args.map_layer) + '_fs_' + str(args.fs_layer) + '_' + node_str + '_final.tsv'
        if args.map_layer:
            tsv_name = args.choice + '_map_' + str(args.map_layer) + '_' + args.map_type + '_fs_' + str(args.fs_layer) + '_' + node_str + '_final.tsv'
        final_result_df.to_csv(os.path.join(final_test_save_dir, tsv_name), sep = '\t', index=False)
