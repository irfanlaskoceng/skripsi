import pandas as pd
import numpy as np
from math import exp

def oneHotEncoding(allDataset,fiturNumeric,fiturKategorikal):
    tmp_one_hot_encoding = allDataset[fiturNumeric]
    for i in fiturKategorikal:
        kategori = np.sort(allDataset[i].unique())
        for k in kategori:
            new_fitur=[]
            for l in allDataset[i].values:
                if l == k:
                    new_fitur.append(1)
                else:
                    new_fitur.append(0)
            tmp_one_hot_encoding.insert(loc=len(tmp_one_hot_encoding.columns), column=i+'_'+str(k), value=new_fitur)
    return tmp_one_hot_encoding

def minMax(data,newMinA,newMaxA):
    fitur=data.columns
    tmp_min_max=pd.DataFrame()
    for fitur_i in range(len(fitur)):
        minA=data[fitur[fitur_i]].min()
        maxA=data[fitur[fitur_i]].max()
        tmp_result=[]
        for v in data[fitur[fitur_i]]:
            result= ( ((v-minA)/(maxA-minA)) * (newMaxA-newMinA) ) + newMinA
            tmp_result.append(result)
        tmp_min_max.insert(loc=fitur_i, column=fitur[fitur_i], value=tmp_result)
    return tmp_min_max


#=================SGD=================
def addBias(x):
    bias=[1]*len(x)
    tmp_result=np.insert(x,0,bias,axis=1)
    return tmp_result

def predict(row, coefficients):
    yhat = (coefficients * row).sum()
    return 1.0 / (1.0 + exp(-yhat))

def coefficients_sgd(train, target, l_rate, n_epoch):
    train = addBias(train)
    coef = [0.0]*train.shape[1]
    for epoch in range(n_epoch):
        for row in range (len(train)):
            yhat = predict(train[row], coef)
            for col in range(len(train[row])):
                coef[col] += l_rate * (target[row] - yhat) * (1.0 - yhat) * train[row][col]
        # print('>epoch=%d, lrate=%.3f, coef %s' % (epoch, l_rate, coef))
    return coef    

def getThreshold(fitur_y):
        total_class_positif = np.count_nonzero(fitur_y==1)
        total_data = len(fitur_y)
        result = ((total_class_positif/total_data)+0.5)/2
        return result

def fit(test_x,threshold,coef):
    test_prediction_true_value=[]
    test_prediction_with_threshold=[]
    for row in range(len(test_x)):
        yhat = predict(test_x[row], coef)
        test_prediction_true_value.append(yhat)
        if yhat<threshold:
            test_prediction_with_threshold.append(0)
        else:
            test_prediction_with_threshold.append(1)
    test_prediction=pd.DataFrame()
    test_prediction.insert(loc=0, column='prediction_true_value', value=test_prediction_true_value)
    test_prediction.insert(loc=1, column='prediction_with_threshold', value=test_prediction_with_threshold)
    return test_prediction

def confutionMatrix(y_actual,y_prediction):
    matrix={'tp':0, 'tn':0, 'fp':0, 'fn':0}
    for row in range(len(y_actual)):    
        if y_actual[row]==1 and y_prediction[row]==1:#tp
            matrix['tp']=matrix['tp']+1
        elif y_actual[row]==0 and y_prediction[row]==0:#tn
            matrix['tn']=matrix['tn']+1
        elif y_actual[row]==0 and y_prediction[row]==1:#fp
            matrix['fp']=matrix['fp']+1
        elif y_actual[row]==1 and y_prediction[row]==0:#fn
            matrix['fn']=matrix['fn']+1
        # print(y_actual[row],'==',y_prediction[row],':',y_actual[row]==y_prediction[row])
    return matrix




#proses lorens
def generateRandomPartitionInEnsemble(fitur,k_subspace,n_ensemble):
    tmp_partion=pd.DataFrame()
    for i in range((n_ensemble)):
        tmp_emsemble=[]
        for j in range(len(fitur)):
            tmp_emsemble.append(np.random.randint(1,k_subspace+1))
        if i==0:
            tmp_partion.insert(loc=i, column='fitur', value=fitur)
            tmp_partion.insert(loc=i+1, column='ensemble_'+str(i+1), value=tmp_emsemble)
        else:
            tmp_partion.insert(loc=i+1, column='ensemble_'+str(i+1), value=tmp_emsemble)
    return tmp_partion


def getFiturPartition(dataEnsemblePartition,ensemble_i):
    partition = np.unique(dataEnsemblePartition[ensemble_i].values)
    tmp_fitur_partion={} 
    for part in range(len(partition)):
        fitur_partion=[]
        # print('=>fitur partisi :',partition[part])
        for row_fitur in range(len(dataEnsemblePartition[ensemble_i])):
            if partition[part] == dataEnsemblePartition[ensemble_i][row_fitur]:  
                fitur_partion.append(dataEnsemblePartition['fitur'][row_fitur])
                # print(partition[part],dataEnsemblePartition[ensemble_i][row_fitur],':',dataEnsemblePartition['fitur'][row_fitur])
        tmp_fitur_partion[partition[part]]=fitur_partion
    # print(tmp_fitur_partion)
    return(tmp_fitur_partion)

def majorityVoting(data_prediction_with_threshold_partition):
    data_prediction_with_threshold_partition = data_prediction_with_threshold_partition.values
    vote_prediction=[]
    for row in range(len(data_prediction_with_threshold_partition)):
        all_predic_in_row = np.array(data_prediction_with_threshold_partition[row])
        sum_positif  = all_predic_in_row.tolist().count(1)
        sum_negative = all_predic_in_row.tolist().count(0)
        if sum_positif>sum_negative:
            vote_prediction.append(1)
        else:
            vote_prediction.append(0)
    return vote_prediction
 
# #lorens
def lorens(random_partition_in_ensemble, data_fold_i):  
    result={}
    for col_ensemble in random_partition_in_ensemble.columns:
        if col_ensemble != 'fitur':
            fitur_every_partition = getFiturPartition(random_partition_in_ensemble,col_ensemble)
            key_fitur_every_partition = list(fitur_every_partition.keys())
    
            tmp_prediction_true_valu_partition=pd.DataFrame()
            tmp_prediction_with_threshold=pd.DataFrame()
            
            data_train_y = np.array(data_fold_i['Y-traning'])
            data_test_y  = np.array(data_fold_i['Y-testing'])
            threshold    = getThreshold(data_train_y)
            # print('Threshold :', threshold)
            for part in range(len(key_fitur_every_partition)):
                data_train_x = data_fold_i['X-traning'][fitur_every_partition[key_fitur_every_partition[part]]].values
                l_rate = 0.3
                n_epoch = 10
                coef_last = coefficients_sgd(data_train_x,data_train_y, l_rate, n_epoch)
                
                data_test_x = addBias(data_fold_i['X-testing'][fitur_every_partition[key_fitur_every_partition[part]]].values)
                data_test_prediction_y=fit(data_test_x,threshold,coef_last)
                tmp_prediction_true_valu_partition.insert(loc=part, column='partition_'+str(key_fitur_every_partition[part]), value=data_test_prediction_y['prediction_true_value'])
                tmp_prediction_with_threshold.insert(loc=part, column='partition_'+str(key_fitur_every_partition[part]), value=data_test_prediction_y['prediction_with_threshold'])
                if part == len(key_fitur_every_partition)-1:
                    tmp_majority_voting=majorityVoting(tmp_prediction_with_threshold)
                    tmp_prediction_with_threshold.insert(loc=part+1, column='majority_voting', value=tmp_majority_voting)
            confutionM=confutionMatrix(data_test_y,tmp_prediction_with_threshold['majority_voting'])
            akurasi=(confutionM['tp']+confutionM['tn']) / (confutionM['tp']+confutionM['tn']+confutionM['fp']+confutionM['fn'])
            precision=confutionM['tp']/(confutionM['tp']+confutionM['fp'])
            recall=confutionM['tp']/(confutionM['fn']+confutionM['tp'])
            result[col_ensemble]={'threshold':threshold, 'akurasi':akurasi, 'precision':precision, 'recall':recall, 'prediction_with_threshold':tmp_prediction_with_threshold, 'prediction_true_value':tmp_prediction_true_valu_partition, 'fitur_every_partition': fitur_every_partition, 'confution_matrix':confutionM} 
    return result
def crossValidation(data,dataY,fitur,persen,fold):
    total_traning=(persen*len(data)//100)-1
    lompatan=len(data)//fold
    # print(len(data)); print(total_traning); print(lompatan) ;print(fitur)
    bknn=0
    result_fold_index=[]
    for f in range(fold):
        if f==0:
            bkr=0
            bkn=total_traning
        else:
            bkr+=lompatan
            bkn=total_traning+bkr
        if bkn>len(data):
            bkn=len(data)
            bknn=total_traning - (len(data)-bkr)
        # print('--------fold =',f+1); print('bkr =',bkr); print('bkn =',bkn); print('bknn =',bknn); print()
        index_traning=[] ; x_traning=[] ; y_traning=[]
        index_testing=[] ; x_testing=[] ; y_testing=[]
        for r in range(len(data)):
            if bknn == 0:
                if r>=bkr and r<=bkn:
                    index_traning.append(r)
                    x_traning.append(data[r])
                    y_traning.append(dataY[r])
                else:
                    index_testing.append(r)
                    x_testing.append(data[r])
                    y_testing.append(dataY[r])
            else:
                if r<=bknn:
                    index_traning.append(r)
                    x_traning.append(data[r])
                    y_traning.append(dataY[r])
                elif r >= bkr:
                    index_traning.append(r)
                    x_traning.append(data[r])
                    y_traning.append(dataY[r])
                else:
                    index_testing.append(r)
                    x_testing.append(data[r])
                    y_testing.append(dataY[r])
        # print(index_traning,'---',index_testing); print()
        x_traning=pd.DataFrame(x_traning, columns=fitur)
        x_testing=pd.DataFrame(x_testing, columns=fitur)
        result_fold_index.append({'INDEX-traning':index_traning, 'INDEX-testing':index_testing, 
                                  'X-traning'    :x_traning,     'X-testing':x_testing,
                                  'Y-traning'    :y_traning,     'Y-testing':y_testing})
    return result_fold_index
                
# def doCrossValidationLorens(data_fold,random_partition):
#     result = []
#     for fold_i in range(len(data_fold)):
#         print('+++++++++ fold',fold_i+1,'/////////////////////////////////////////////////')
#         tmp_result_lorens = lorens(random_partition,data_fold[fold_i])
#         result.append(tmp_result_lorens)
#     return result    


def doCrossValidationLorens(data_fold,random_partition):
    result = []
    result_lg = []
    for fold_i in range(len(data_fold)):
        print('+++++++++ fold',fold_i+1,'/////////////////////////////////////////////////')
        tmp_result_lorens = lorens(random_partition,data_fold[fold_i])
        result.append(tmp_result_lorens)

        lg_data_train_y = np.array(data_fold[fold_i]['Y-traning'])
        lg_data_test_y  = np.array(data_fold[fold_i]['Y-testing'])
        
        lg_data_train_x = data_fold[fold_i]['X-traning'].values
        lg_l_rate = 0.3
        lg_n_epoch = 10
        lg_threshold    = getThreshold(lg_data_train_y)
        lg_coef_last = coefficients_sgd(lg_data_train_x,lg_data_train_y, lg_l_rate, lg_n_epoch)
        
        lg_data_test_x = addBias(data_fold[fold_i]['X-testing'].values)
        lg_data_test_prediction_y=fit(lg_data_test_x,lg_threshold,lg_coef_last)
        lg_tmp_prediction_logistic_regression=pd.DataFrame()
        lg_tmp_prediction_logistic_regression.insert(loc=0, column='lg_prediction_true_value', value=lg_data_test_prediction_y['prediction_true_value'])
        lg_tmp_prediction_logistic_regression.insert(loc=1, column='lg_prediction_with_threshold', value=lg_data_test_prediction_y['prediction_with_threshold'])
        
        confutionM=confutionMatrix(lg_data_test_y,lg_tmp_prediction_logistic_regression['lg_prediction_with_threshold'])
        akurasi=(confutionM['tp']+confutionM['tn']) / (confutionM['tp']+confutionM['tn']+confutionM['fp']+confutionM['fn'])
        precision=confutionM['tp']/(confutionM['tp']+confutionM['fp'])
        recall=confutionM['tp']/(confutionM['fn']+confutionM['tp'])
        result_lg.append( {"logistic_regresion" : {'threshold':0.5, 'akurasi':akurasi, 'precision':precision, 'recall':recall, 'prediction_logistic_regression':lg_tmp_prediction_logistic_regression, 'confution_matrix_logistic_regression':confutionM}} ) 

    return result, result_lg 




def get_mean(data_performance):
    rata={}
    for i in range(len(data_performance)):
        key_ens= list(data_performance[i].keys())
        if i == 0:
            for j in key_ens:
                rata[j]={'akurasi':0,'precision':0,'recall':0}
                rata[j]['akurasi']=rata[j]['akurasi']+data_performance[i][j]['akurasi']
                rata[j]['precision']=rata[j]['precision']+data_performance[i][j]['precision']
                rata[j]['recall']=rata[j]['recall']+data_performance[i][j]['recall']
        elif i > 0:
            for j in key_ens:
                rata[j]['akurasi']=rata[j]['akurasi']+data_performance[i][j]['akurasi']
                rata[j]['precision']=rata[j]['precision']+data_performance[i][j]['precision']
                rata[j]['recall']=rata[j]['recall']+data_performance[i][j]['recall']
        if i == len(data_performance)-1:
            for j in key_ens:
                rata[j]['akurasi']=rata[j]['akurasi']/len(data_performance)
                rata[j]['precision']=rata[j]['precision']/len(data_performance)
                rata[j]['recall']=rata[j]['recall']/len(data_performance)
    return rata


    
def findBy(dtMeans,ens,sp):#sp adalah spesifik performa
    best={'ens':'---', 'vp':[], 'ens_same':'---', 'vp_same':[]}
    for j in range(len(ens)-1):
        print(dtMeans[ens[j]][sp],'ccccccc')
        if j==0:
            best['ens']=ens[j]
            best['vp']=dtMeans[ens[j]][sp]
            if best['vp'] < dtMeans[ens[j+1]][sp]:
                best['ens']=ens[j+1]
                best['vp']=dtMeans[ens[j+1]][sp]
            elif best['vp'] == dtMeans[ens[j+1]][sp]:
                best['ens']=ens[j+1]
                best['vp']=dtMeans[ens[j+1]][sp]
                best['ens_same']=ens[j]
                best['vp_same']=dtMeans[ens[j]][sp]
        else:
            if best['vp'] < dtMeans[ens[j+1]][sp]:
                best['ens']=ens[j+1]
                best['vp']=dtMeans[ens[j+1]][sp]
            elif best['vp'] == dtMeans[ens[j+1]][sp]:
                best['ens']=ens[j+1]
                best['vp']=dtMeans[ens[j+1]][sp]
                best['ens_same']=ens[j]
                best['vp_same']=dtMeans[ens[j]][sp]
        if j == (len(ens)-1)-1:
            if best['vp_same'] != []:
                if best['vp'] > best['vp_same']:
                    best['vp_same']=[]
                    best['ens_same']='---'
        print(ens[j],best,sp)
    return best

def getBestEnsemble(means_evry_ensemble):    
    tmp_one_best_ensemble=''
    tmp_by=['akurasi', 'precision', 'recall']
    for j in range (len(tmp_by)):
        tmp=findBy(means_evry_ensemble,list(means_evry_ensemble.keys()),tmp_by[j])
        if tmp['vp_same'] == []:

            tmp_one_best_ensemble=tmp['ens']
            break
    return tmp_one_best_ensemble

























