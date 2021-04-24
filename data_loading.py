from csv import reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import joblib
import argparse

def split_train_val_test(sample_list, ratio = [0.8, 0.1]):
    sample_num = len(sample_list)
    np.random.shuffle(sample_list)
    
    train_size = int(sample_num * 0.8)
    val_size = int(sample_num * 0.1)
    test_size = sample_num - train_size - val_size
    
    train_list = sample_list[0:train_size]
    val_list = sample_list[train_size+1:train_size+val_size]
    test_list = sample_list[train_size+val_size+1:]
    
    return train_list, val_list, test_list

def split_train_test(sample_list, ratio=0.8):
    sample_num = len(sample_list)
    np.random.shuffle(sample_list)
    
    train_size = int(sample_num * 0.8)
    test_size = sample_num - train_size
    train_list = sample_list[0:train_size]
    test_list = sample_list[train_size:]
    
    return train_list, test_list
    
    

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])



if __name__ == "__main__":
    # specifying data paths
    parser = argparse.ArgumentParser(description = 'Data Loading Pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default='data_stranded', type=str, help='Specify the data directory')
    parser.add_argument('--type', default='gene', type=str, help='Specify the hidden nodes of 1st layer')
    print("Loading Data")
    args = parser.parse_args()

    if args.type == 'exon':
        data_dir = '../' + args.data_dir
        #COPDGene_Freeze1_RNAseq_genes = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes.csv')
        #COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_covAdjustedWOsv.csv')
        COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_normalized.csv')
        COPDGene_Freeze1_RNAseq_samples = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_samples.csv')
        
        # loading csv data to pandas frame
        # genes = pd.read_csv(COPDGene_Freeze1_RNAseq_genes)
        #exonicParts_logCPM_covAdjusted = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted)
        print("Reading csv files")
        exonicParts_logCPM_normalized = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized)
        samples = pd.read_csv(COPDGene_Freeze1_RNAseq_samples)
        
        # dealing with data
        # so every row represents a patient for now
        # shape (520, 266228)
        #exonicParts_logCPM_covAdjusted_matrix = np.transpose(exonicParts_logCPM_covAdjusted.values[:, 1:])
        print("Loading to matrix")
        exonicParts_logCPM_normalized_matrix = np.transpose(exonicParts_logCPM_normalized.values[:, 1:])
        
        # get smoking status
        # shape (520, )
        smoking_status = samples['SmokCigNow'].values
        # TODO:!!!!!
        # smoking_status_matrix = one_hot(smoking_status, 2)
        
        # total number of examples 
        sample_num_total = exonicParts_logCPM_normalized_matrix.shape[0]
        assert(smoking_status.shape[0] == sample_num_total)
        
        # get train, val, test indices
        pos_list = np.where(smoking_status==1)
        neg_list = np.where(smoking_status==0)
        
        train_list_pos, val_list_pos, test_list_pos = split_train_val_test(np.arange(sample_num_total)[pos_list])
        train_list_neg, val_list_neg, test_list_neg = split_train_val_test(np.arange(sample_num_total)[neg_list])
        #train_list_pos, test_list_pos = split_train_test(np.arange(sample_num_total)[pos_list])
        #train_list_neg, test_list_neg = split_train_test(np.arange(sample_num_total)[neg_list])
        
        train_list = np.concatenate((train_list_pos, train_list_neg), axis=0)
        val_list = np.concatenate((val_list_pos, val_list_neg), axis=0)
        # we keep test as a held-out dataset
        test_list = np.concatenate((test_list_pos, test_list_neg), axis=0)
        
        # saving them to proper place using pickle
        print("Saving to pickle")
        saving_dir = '../' + args.data_dir
        #covAdj_path = os.path.join(saving_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_covAdjusted')
        normalized_path = os.path.join(saving_dir, 'COPDGene_Freeze3_RNAseq_exonicParts_logCPM_normalized')
        smoking_path = os.path.join(saving_dir, 'COPDGene_Freeze3_RNAseq_samples', 'SmokCigNow')
        #if not os.path.exists(covAdj_path):
        #    os.mkdir(covAdj_path)
        if not os.path.exists(normalized_path):
            os.mkdir(normalized_path)
        if not os.path.exists(smoking_path):
            os.mkdir(smoking_path)
        
        # here we should dump the partition list as well
        pickle.dump(train_list, open(os.path.join(saving_dir, 'train_list.pickle'), 'wb'))
        pickle.dump(val_list, open(os.path.join(saving_dir, 'val_list.pickle'), 'wb'))
        pickle.dump(test_list, open(os.path.join(saving_dir, 'test_list.pickle'), 'wb'))
        
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[train_list], open(os.path.join(covAdj_path, 'X_train.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[val_list], open(os.path.join(covAdj_path, 'X_val.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[test_list], open(os.path.join(covAdj_path, 'X_test.pickle'),'wb'))
        
        joblib.dump(exonicParts_logCPM_normalized_matrix[train_list], open(os.path.join(normalized_path, 'X_train.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[val_list], open(os.path.join(normalized_path, 'X_val.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[test_list], open(os.path.join(normalized_path, 'X_test.pickle'),'wb'), protocol=2)
        
        joblib.dump(smoking_status[train_list], open(os.path.join(smoking_path, 'Y_train.pickle'),'wb'))
        joblib.dump(smoking_status[val_list], open(os.path.join(smoking_path, 'Y_val.pickle'),'wb'))
        joblib.dump(smoking_status[test_list], open(os.path.join(smoking_path, 'Y_test.pickle'),'wb'))
        
        # here we directly dump X and Y all
        #print("Saving all the data for now")
        #pickle.dump(exonicParts_logCPM_normalized_matrix, open(os.path.join(normalized_path, 'X_all.pickle'),'wb'))
        #pickle.dump(smoking_status, open(os.path.join(smoking_path, 'Y_all.pickle'),'wb'))

    elif args.type == 'gene':
        data_dir = '../' + args.data_dir
        saving_dir = '../' + args.data_dir
        train_list = pickle.load(open(os.path.join(saving_dir, 'train_list.pickle'), 'rb'))
        val_list = pickle.load(open(os.path.join(saving_dir, 'val_list.pickle'), 'rb'))
        test_list = pickle.load(open(os.path.join(saving_dir, 'test_list.pickle'), 'rb'))
        #COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_covAdjusted.csv')
        COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_normalized.csv')
        
        # loading csv data to pandas frame
    
        #exonicParts_logCPM_covAdjusted = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted)
        exonicParts_logCPM_normalized = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized)
        
        # dealing with data
        # so every row represents a patient for now
        # shape (520, 266228)
        #exonicParts_logCPM_covAdjusted_matrix = np.transpose(exonicParts_logCPM_covAdjusted.values[:, 1:])
        exonicParts_logCPM_normalized_matrix = np.transpose(exonicParts_logCPM_normalized.values[:, 1:])
        
        # get smoking status
        # shape (520, )
        # smoking_status = samples['SmokCigNow'].values
        # TODO:!!!!!
        # smoking_status_matrix = one_hot(smoking_status, 2)
        
        # total number of examples 
        sample_num_total = exonicParts_logCPM_normalized_matrix.shape[0]
        # assert(smoking_status.shape[0] == sample_num_total)
        
        # get train, val, test indices
        #pos_list = np.where(smoking_status==1)
        #neg_list = np.where(smoking_status==0)
        
        #train_list_pos, val_list_pos, test_list_pos = split_train_val_test(np.arange(sample_num_total)[pos_list])
        #train_list_neg, val_list_neg, test_list_neg = split_train_val_test(np.arange(sample_num_total)[neg_list])
        
        #train_list = np.concatenate((train_list_pos, train_list_neg), axis=0)
        #val_list = np.concatenate((val_list_pos, val_list_neg), axis=0)
        #test_list = np.concatenate((test_list_pos, test_list_neg), axis=0)
        
        # saving them to proper place using pickle
        saving_dir = '../' + args.data_dir
        #covAdj_path = os.path.join(saving_dir, 'COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted')
        normalized_path = os.path.join(saving_dir, 'COPDGene_Freeze3_RNAseq_genes_logCPM_normalized')
        #smoking_path = os.path.join(saving_dir, 'COPDGene_Freeze1_RNAseq_samples', 'SmokCigNow')
        
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[train_list], open(os.path.join(covAdj_path, 'X_train.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[val_list], open(os.path.join(covAdj_path, 'X_val.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[test_list], open(os.path.join(covAdj_path, 'X_test.pickle'),'wb'))
        
        joblib.dump(exonicParts_logCPM_normalized_matrix[train_list], open(os.path.join(normalized_path, 'X_train.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[val_list], open(os.path.join(normalized_path, 'X_val.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[test_list], open(os.path.join(normalized_path, 'X_test.pickle'),'wb'), protocol=2)
        
        #pickle.dump(smoking_status[train_list], open(os.path.join(smoking_path, 'Y_train.pickle'),'wb'))
        #pickle.dump(smoking_status[val_list], open(os.path.join(smoking_path, 'Y_val.pickle'),'wb'))
        #pickle.dump(smoking_status[test_list], open(os.path.join(smoking_path, 'Y_test.pickle'),'wb'))
        
        # here we directly dump X and Y all
        #print("Saving all the data for now")
        #pickle.dump(exonicParts_logCPM_normalized_matrix, open(os.path.join(normalized_path, 'X_all.pickle'),'wb'))
        #pickle.dump(smoking_status, open(os.path.join(smoking_path, 'Y_all.pickle'),'wb'))
        
    elif args.type == 'trans':
        data_dir = '../' + args.data_dir
        saving_dir = '../' + args.data_dir
        train_list = pickle.load(open(os.path.join(saving_dir, 'train_list.pickle'), 'rb'))
        val_list = pickle.load(open(os.path.join(saving_dir, 'val_list.pickle'), 'rb'))
        test_list = pickle.load(open(os.path.join(saving_dir, 'test_list.pickle'), 'rb'))
        COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized = os.path.join(data_dir, 'COPDGene_Freeze3_RNAseq_transcripts_logCPM_normalized.csv')
        
        # loading csv data to pandas frame
    
        #exonicParts_logCPM_covAdjusted = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted)
        exonicParts_logCPM_normalized = pd.read_csv(COPDGene_Freeze1_RNAseq_exonicParts_logCPM_normalized)
        
        # dealing with data
        # so every row represents a patient for now
        # shape (520, 266228)
        # exonicParts_logCPM_covAdjusted_matrix = np.transpose(exonicParts_logCPM_covAdjusted.values[:, 1:])
        exonicParts_logCPM_normalized_matrix = np.transpose(exonicParts_logCPM_normalized.values[:, 1:])
        
        # get smoking status
        # shape (520, )
        # smoking_status = samples['SmokCigNow'].values
        # TODO:!!!!!
        # smoking_status_matrix = one_hot(smoking_status, 2)
        
        # total number of examples 
        # sample_num_total = exonicParts_logCPM_covAdjusted_matrix.shape[0]
        # assert(smoking_status.shape[0] == sample_num_total)
        
        # get train, val, test indices
        #pos_list = np.where(smoking_status==1)
        #neg_list = np.where(smoking_status==0)
        
        #train_list_pos, val_list_pos, test_list_pos = split_train_val_test(np.arange(sample_num_total)[pos_list])
        #train_list_neg, val_list_neg, test_list_neg = split_train_val_test(np.arange(sample_num_total)[neg_list])
        
        #train_list = np.concatenate((train_list_pos, train_list_neg), axis=0)
        #val_list = np.concatenate((val_list_pos, val_list_neg), axis=0)
        #test_list = np.concatenate((test_list_pos, test_list_neg), axis=0)
        
        # saving them to proper place using pickle
        saving_dir = '../' + args.data_dir
        #covAdj_path = os.path.join(saving_dir, 'COPDGene_Freeze1_RNAseq_exonicParts_logCPM_covAdjusted')
        normalized_path = os.path.join(saving_dir, 'COPDGene_Freeze3_RNAseq_transcripts_logCPM_normalized')
        
        joblib.dump(exonicParts_logCPM_normalized_matrix[train_list], open(os.path.join(normalized_path, 'X_train.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[val_list], open(os.path.join(normalized_path, 'X_val.pickle'),'wb'), protocol=2)
        joblib.dump(exonicParts_logCPM_normalized_matrix[test_list], open(os.path.join(normalized_path, 'X_test.pickle'),'wb'), protocol=2)
        
        
        #smoking_path = os.path.join(saving_dir, 'COPDGene_Freeze1_RNAseq_samples', 'SmokCigNow')
        
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[train_list], open(os.path.join(covAdj_path, 'X_train.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[val_list], open(os.path.join(covAdj_path, 'X_val.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_covAdjusted_matrix[test_list], open(os.path.join(covAdj_path, 'X_test.pickle'),'wb'))
        
        #pickle.dump(exonicParts_logCPM_normalized_matrix[train_list], open(os.path.join(normalized_path, 'X_train.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_normalized_matrix[val_list], open(os.path.join(normalized_path, 'X_val.pickle'),'wb'))
        #pickle.dump(exonicParts_logCPM_normalized_matrix[test_list], open(os.path.join(normalized_path, 'X_test.pickle'),'wb'))
        
        #pickle.dump(smoking_status[train_list], open(os.path.join(smoking_path, 'Y_train.pickle'),'wb'))
        #pickle.dump(smoking_status[val_list], open(os.path.join(smoking_path, 'Y_val.pickle'),'wb'))
        #pickle.dump(smoking_status[test_list], open(os.path.join(smoking_path, 'Y_test.pickle'),'wb'))
        
        # here we directly dump X and Y all
        #pickle.dump(smoking_status, open(os.path.join(smoking_path, 'Y_all.pickle'),'wb'))
    


    
    
    
    
    
    
