from flask import Flask, render_template, redirect, url_for, jsonify, request, session,Blueprint
from method import *


app3 = Blueprint('app3',__name__,static_folder='static',template_folder='templates')


@app3.route("/input_new_data")
def index():
    # return("wahaisaudaraku")

    return render_template("tam3.html")

    # return render_template("tam2.html", list_data_awal=list_data_awal,list_fitur_awal=list_fitur_awal, 
    # list_dataset_min_max_fitur_nu_ohe=list_dataset_min_max_fitur_nu_ohe,list_dataset_min_max_nu_ohe=list_dataset_min_max_nu_ohe,
    # default_columns_nu_ohe=default_columns_nu_ohe,default_value_nu_ohe=default_value_nu_ohe)


@app3.route('/_do_prediction')
def do_prediction():
    tmp_age = request.args.get('f_age', 0, type=int)
    tmp_trestbps = request.args.get('f_trestbps', 0, type=int)
    tmp_chol = request.args.get('f_chol', 0, type=int)
    tmp_thalach = request.args.get('f_thalach', 0, type=int)
    tmp_oldpeak = request.args.get('f_oldpeak', 0, type=int)
    tmp_ca = request.args.get('f_ca', 0, type=int)

    tmp_sex = request.args.get('f_sex', 0, type=int)
    tmp_cp = request.args.get('f_cp', 0, type=int) 
    tmp_fbs = request.args.get('f_fbs', 0, type=int)
    tmp_restecg = request.args.get('f_restecg', 0, type=int)
    tmp_exang = request.args.get('f_exang', 0, type=int)
    tmp_slope = request.args.get('f_slope', 0, type=int)
    tmp_thal = request.args.get('f_thal', 0, type=int)
    
    df3=pd.read_csv("heart.csv")
    list_fitur_awal3 = df3.columns.values.tolist()
    list_data_awal3  = df3.values.tolist()
    data_y3 = df3.target.values.tolist()
    df3.pop('target')

    default_columns=['fitur', 'ensemble_1', 'ensemble_2', 'ensemble_3', 'ensemble_4','ensemble_5', 'ensemble_6', 'ensemble_7', 'ensemble_8', 'ensemble_9', 'ensemble_10']
    default_value=[['age', 3, 2, 1, 2, 3, 1, 3, 2, 2, 2], ['trestbps', 3, 2, 3, 2, 1, 3, 3, 3, 3, 2], ['chol', 3, 3, 1, 3, 3, 1, 1, 1, 3, 3], ['thalach', 3, 3, 3, 2, 1, 2, 3, 2, 1, 1], ['oldpeak', 1, 3, 2, 1, 3, 3, 1, 2, 3, 3], ['ca', 1, 1, 3, 3, 1, 2, 1, 3, 2, 2], ['sex_0', 2, 2, 3, 3, 3, 1, 1, 1, 1, 2], ['sex_1', 3, 2, 3, 1, 2, 2, 1, 1, 1, 1], ['cp_0', 2, 1, 1, 3, 2, 1, 1, 3, 2, 1], ['cp_1', 2, 2, 1, 2, 1, 3, 1, 1, 3, 2], ['cp_2', 2, 1, 2, 2, 1, 3, 3, 3, 3, 3], ['cp_3', 2, 2, 2, 2, 3, 2, 3, 1, 3, 3], ['fbs_0', 3, 1, 2, 2, 2, 2, 3, 3, 1, 2], ['fbs_1', 2, 2, 2, 2, 1, 1, 3, 2, 2, 1], ['restecg_0', 1, 3, 1, 2, 2, 3, 3, 2, 3, 3], ['restecg_1', 3, 3, 3, 1, 1, 3, 2, 3, 2, 3], ['restecg_2', 3, 2, 1, 1, 3, 2, 2, 1, 1, 3], ['exang_0', 1, 3, 1, 2, 3, 2, 1, 1, 1, 1], ['exang_1', 2, 3, 1, 2, 2, 1, 2, 3, 2, 2], ['slope_0', 1, 1, 1, 3, 3, 3, 2, 3, 3, 2], ['slope_1', 1, 3, 2, 3, 2, 1, 3, 3, 1, 3], ['slope_2', 3, 3, 1, 3, 2, 3, 3, 2, 1, 3], ['thal_0', 3, 2, 2, 1, 1, 3, 2, 2, 2, 3], ['thal_1', 1, 3, 3, 3, 2, 3, 3, 2, 2, 3], ['thal_2', 2, 2, 1, 2, 3, 1, 1, 2, 2, 3], ['thal_3', 2, 2, 2, 2, 3, 1, 2, 2, 3, 3]]
    random_partition_in_ensemble3=pd.DataFrame(default_value, columns=default_columns)

    print(df3)
    print(df3.columns)
    
    # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
    new_data=[tmp_age, tmp_sex, tmp_cp, tmp_trestbps, tmp_chol, tmp_fbs, tmp_restecg, tmp_thalach, tmp_exang, tmp_oldpeak, tmp_slope, tmp_ca, tmp_thal]
    df3.loc[len(df3)] = new_data
    print(df3)
    print(df3.columns)

    print()
    print()

    numeric3 = ['age','trestbps', 'chol','thalach','oldpeak', 'ca']
    nominal3 = ['sex', 'cp',  'fbs', 'restecg','exang', 'slope', 'thal']
    dataset_one_hot_encoding3            = oneHotEncoding(df3,numeric3,nominal3)
    list_dataset_one_hot_encoding_fitur3 = dataset_one_hot_encoding3.columns.values.tolist()
    list_dataset_one_hot_encoding3       = dataset_one_hot_encoding3.values.tolist()
    new_data_one_hot_encoding=(dataset_one_hot_encoding3.iloc[-1]).values

    dataset_min_max3            = minMax(dataset_one_hot_encoding3,0.1,0.9) 
    new_data_min_max = dataset_min_max3.iloc[-1].values
    print(dataset_min_max3)

    dataset_min_max3=dataset_min_max3.drop(len(dataset_min_max3)-1)
    print('------')
    print(dataset_min_max3)
    list_dataset_min_max_fitur3 = dataset_min_max3.columns.values.tolist()
    list_dataset_min_max3       = dataset_min_max3.values.tolist()
    print(len(list_dataset_min_max3), list_dataset_min_max3[-1])
    print(new_data_min_max)
    
    
    df_new_data_min_max=pd.DataFrame([new_data_min_max], columns=list_dataset_min_max_fitur3)
    print(df_new_data_min_max)

    cross_validation3 = crossValidation(list_dataset_min_max3, data_y3, list_dataset_min_max_fitur3, 60,10)

    #lorens
    fitur_every_partition = getFiturPartition(random_partition_in_ensemble3,'ensemble_3')
    key_fitur_every_partition = list(fitur_every_partition.keys())
    
    tmp_prediction_true_valu_partition=pd.DataFrame()
    tmp_prediction_with_threshold=pd.DataFrame()
    
    data_train_y = np.array(cross_validation3[4]['Y-traning'])
    threshold    = getThreshold(data_train_y)
    print(threshold)

    for part in range(len(key_fitur_every_partition)):
        data_train_x = cross_validation3[4]['X-traning'][fitur_every_partition[key_fitur_every_partition[part]]].values
        l_rate = 0.3
        n_epoch = 10
        coef_last = coefficients_sgd(data_train_x,data_train_y, l_rate, n_epoch)
        
        data_test_x = addBias(df_new_data_min_max[fitur_every_partition[key_fitur_every_partition[part]]].values)
        data_test_prediction_y=fit(data_test_x,threshold,coef_last)
        tmp_prediction_true_valu_partition.insert(loc=part, column='partition_'+str(key_fitur_every_partition[part]), value=data_test_prediction_y['prediction_true_value'])
        tmp_prediction_with_threshold.insert(loc=part, column='partition_'+str(key_fitur_every_partition[part]), value=data_test_prediction_y['prediction_with_threshold'])
        if part == len(key_fitur_every_partition)-1:
            tmp_majority_voting=majorityVoting(tmp_prediction_with_threshold)
            tmp_prediction_with_threshold.insert(loc=part+1, column='majority_voting', value=tmp_majority_voting)
    
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(tmp_prediction_with_threshold)

    # cross_validation[0]['X-testing']
    # cross_validation[0].keys()
    # Out[39]: dict_keys(['INDEX-traning', 'INDEX-testing', 'X-traning', 'X-testing', 'Y-traning', 'Y-testing'])


    pred=tmp_prediction_with_threshold['majority_voting'].values[0]
    print('prediksi', pred)





    return jsonify(result=str(pred))