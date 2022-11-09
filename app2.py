from flask import Flask, render_template, redirect, url_for, jsonify, request, session,Blueprint
from method import *


app2 = Blueprint('app2',__name__,static_folder='static',template_folder='templates')


df=pd.read_csv("heart.csv")
list_fitur_awal = df.columns.values.tolist()
list_data_awal  = df.values.tolist()
data_y = df.target.values.tolist()

numeric = ['age','trestbps', 'chol','thalach','oldpeak', 'ca']
nominal = ['sex', 'cp',  'fbs', 'restecg','exang', 'slope', 'thal']
data_x = df[numeric+nominal]

dataset_min_max_nu_ohe            = minMax(data_x,0.1,0.9) 
list_dataset_min_max_fitur_nu_ohe = dataset_min_max_nu_ohe.columns.values.tolist()
list_dataset_min_max_nu_ohe       = dataset_min_max_nu_ohe.values.tolist()

default_columns_nu_ohe=['fitur', 'ensemble_1', 'ensemble_2', 'ensemble_3', 'ensemble_4','ensemble_5', 'ensemble_6', 'ensemble_7', 'ensemble_8', 'ensemble_9', 'ensemble_10']
default_value_nu_ohe= [['age', 3, 1, 1, 1, 1, 1, 2, 1, 2, 2], ['trestbps', 1, 1, 1, 1, 3, 2, 1, 3, 2, 1], ['chol', 3, 3, 1, 2, 1, 1, 2, 2, 3, 3], ['thalach', 2, 2, 2, 2, 1, 3, 1, 2, 3, 3], ['oldpeak', 3, 3, 3, 2, 3, 3, 3, 2, 1, 3], ['ca', 3, 3, 1, 1, 3, 1, 3, 2, 3, 1], ['sex', 1, 3, 2, 2, 3, 3, 3, 3, 2, 2], ['cp', 1, 1, 1, 3, 3, 2, 2, 2, 1, 1], ['fbs', 1, 2, 2, 3, 1, 3, 2, 2, 2, 1], ['restecg', 3, 3, 1, 2, 1, 2, 1, 2, 3, 2], ['exang', 2, 1, 1, 2, 2, 3, 3, 3, 1, 3], ['slope', 2, 1, 2, 1, 3, 3, 3, 3, 2, 2], ['thal', 3, 2, 2, 3, 1, 3, 2, 1, 2, 3]]

random_partition_in_ensemble_nu_ohe=pd.DataFrame(default_value_nu_ohe, columns=default_columns_nu_ohe)
result_cross_validation_lorens_nu_ohe=[]


@app2.route("/_nu_ohe")
def index():
    return render_template("tam2.html", list_data_awal=list_data_awal,list_fitur_awal=list_fitur_awal, 
    list_dataset_min_max_fitur_nu_ohe=list_dataset_min_max_fitur_nu_ohe,list_dataset_min_max_nu_ohe=list_dataset_min_max_nu_ohe,
    default_columns_nu_ohe=default_columns_nu_ohe,default_value_nu_ohe=default_value_nu_ohe)

@app2.route("/profil/<name>")
def profil(name=None):
    return render_template("tam.html",name=name)

@app2.route('/_nu_ohe_do_random_partition')
def nu_ohe_do_random_partition():
    a = request.args.get('k_sub', 0, type=int)
    b = request.args.get('n_ens', 0, type=int)
    # return jsonify(result="<b>ggg %s</b>" %(2))
    global random_partition_in_ensemble_nu_ohe
    random_partition_in_ensemble_nu_ohe=generateRandomPartitionInEnsemble(dataset_min_max_nu_ohe.columns.values, a, b)

    tmp='''
                <div class="card bgCARD">
                    <div class="table-responsive" style="height: 500px; overflow-y: auto;">
                        <table class="table  table-sm table-bordered">
                            <thead>
                                <tr>
    '''
    for x in random_partition_in_ensemble_nu_ohe.columns.values.tolist():
        tmp=tmp+('''<th>'''+x+'''</th>''')
    tmp=tmp+'''
                                </tr>
    '''
    for x in random_partition_in_ensemble_nu_ohe.values.tolist():
        tmp=tmp+'''<tr>'''
        for y in x:
            tmp=tmp+('''<td>'''+str(y) +'''</td>''')
        tmp=tmp+'''</tr>'''
    tmp=tmp+'''
                            </thead>
                        </table>
                    </div>
                </div>
    '''
    pd.DataFrame([str(random_partition_in_ensemble_nu_ohe.values.tolist())]).to_clipboard(index=False,header=False)
    return jsonify(result=tmp)

@app2.route('/_nu_ohe_do_lorens')
def nu_ohe_do_lorens():
    
    
    tmp_training = request.args.get('d_training', 0, type=int)
    tmp_fold = request.args.get('fold', 0, type=int) 

    cross_validation = crossValidation(list_dataset_min_max_nu_ohe, data_y, list_dataset_min_max_fitur_nu_ohe, tmp_training,tmp_fold)
    global result_cross_validation_lorens_nu_ohe
    global result_cross_validation_lg_nu_ohe
    result_cross_validation_lorens_nu_ohe, result_cross_validation_lg_nu_ohe = doCrossValidationLorens(cross_validation, random_partition_in_ensemble_nu_ohe)
    result_mean_nu_ohe=get_mean(result_cross_validation_lorens_nu_ohe)
    result_best_ensemble_nu_ohe=getBestEnsemble(result_mean_nu_ohe)
    result_mean_lg_nu_ohe=get_mean(result_cross_validation_lg_nu_ohe)
    print(result_cross_validation_lorens_nu_ohe[0]['ensemble_'+str(1)]['akurasi'])
    
    tmp=''''''
    for i in range(len(result_cross_validation_lorens_nu_ohe)):
        print()
        tmp+='''<div class="row">'''
        tmp+='''<div class="col-sm-12"><b>fold '''+str(i+1)+'''</b></div>'''
        tmp+='''<div class="row ml-4">'''
        for ens_j in random_partition_in_ensemble_nu_ohe.columns:
            if ens_j != 'fitur':
                tmp+='''<div class="col-sm-4 mb-1">'''
                tmp+='''<b>'''+str(ens_j)+'''</b>'''
                tmp+='''<div>&nbsp akurasi : '''+str(result_cross_validation_lorens_nu_ohe[i][ens_j]['akurasi'])+'''</div>'''
                tmp+='''<div>&nbsp precision : '''+str(result_cross_validation_lorens_nu_ohe[i][ens_j]['precision'])+'''</div>'''
                tmp+='''<div>&nbsp recall : '''+str(result_cross_validation_lorens_nu_ohe[i][ens_j]['recall'])+'''</div>'''
                tmp+='''<div>&nbsp <a class="btn btn-info btn-sm" role="button" id=s style="width: 60%; margi:0px" onclick='cekDetail("'''+str(i)+'''-'''+str(ens_j)+'''")'>detail...</a></div>'''
                tmp+='''</div>'''
        
        tmp+='''<div class="col-sm-4 mb-1">'''
        tmp+='''<b>logistic regression</b>'''
        tmp+='''<div>&nbsp akurasi : '''+str(result_cross_validation_lg_nu_ohe[i]["logistic_regresion"]['akurasi'])+'''</div>'''
        tmp+='''<div>&nbsp precision : '''+str(result_cross_validation_lg_nu_ohe[i]["logistic_regresion"]['precision'])+'''</div>'''
        tmp+='''<div>&nbsp recall : '''+str(result_cross_validation_lg_nu_ohe[i]["logistic_regresion"]['recall'])+'''</div>'''
        tmp+='''<div>&nbsp <a class="btn btn-info btn-sm" role="button" id=s style="width: 60%; margi:0px" onclick='cekDetailLogistic("logistic-'''+str(i)+'''")'>detail...</a></div>'''
        tmp+='''</div>'''

        tmp+='''</div>'''
        tmp+='''</div>'''


    tmp+='''<div class="row">'''    
    tmp+='''<h5 class="mt-4 mb-2">Mean</h5> '''
    tmp+='''<div class="row ml-4">'''
    for ens_m in (result_mean_nu_ohe):
        tmp+='''<div class="col-sm-4 mb-1">'''
        tmp+='''<b>'''+str(ens_m)+'''</b>'''
        tmp+='''<div>&nbsp akurasi : '''+str(result_mean_nu_ohe[ens_m]['akurasi'])+'''</div>'''
        tmp+='''<div>&nbsp precision : '''+str(result_mean_nu_ohe[ens_m]['precision'])+'''</div>'''
        tmp+='''<div>&nbsp recall : '''+str(result_mean_nu_ohe[ens_m]['recall'])+'''</div>'''
        tmp+='''</div>'''
    tmp+='''<div class="col-sm-4 mb-1">'''
    tmp+='''<b>logistic regression</b>'''
    tmp+='''<div>&nbsp akurasi : '''+str(result_mean_lg_nu_ohe['logistic_regresion']['akurasi'])+'''</div>'''
    tmp+='''<div>&nbsp precision : '''+str(result_mean_lg_nu_ohe['logistic_regresion']['precision'])+'''</div>'''
    tmp+='''<div>&nbsp recall : '''+str(result_mean_lg_nu_ohe['logistic_regresion']['recall'])+'''</div>'''
    tmp+='''</div>'''
    tmp+='''</div>'''
    tmp+='''</div>'''


    tmp+='''<div class="row">'''
    tmp+='''<div class="col-sm-12">'''
    tmp+='''<h5 class="mt-4 mb-2">Best ensemble ===> '''+str(result_best_ensemble_nu_ohe)+'''</h5> '''
    tmp+='''</div>'''
    tmp+='''</div>'''
    return jsonify(result=tmp)



@app2.route('/_nu_ohe_check_detail')
def nu_ohe_check_detail():
    p = request.args.get('tmp_position')
    p=p.split('-')

    tmp=''''''
    tmp+='''<div class="table-responsive" style="overflow-y: auto;">
                <table class="table  table-sm table-borderless">'''
    tmp+='''        <tr><td>accuracy</td> <td>:</td> <td>'''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['akurasi'])+'''</td></tr>'''
    tmp+='''        <tr><td>precision</td> <td>:</td> <td>'''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['precision'])+'''</td></tr>'''
    tmp+='''        <tr><td>recall</td> <td>:</td> <td>'''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['recall'])+'''</td></tr>'''
    tmp+='''    </table>
            </div> '''

    tmp+='''        <div class="ml-1">confusion matrix</div>'''
    tmp+='''<div class="">
                <div class="table-responsive" style="overflow-y: auto;">
                    <table class="table  table-sm table-bordered">'''
    tmp+='''            <tr><td rowspan="2" class="align-middle">Kelas Aktual</td> <td colspan="2" class="text-center">Kelas Prediksi</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>Negatif</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>TP ('''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['confution_matrix']['tp'])+''')</td> <td>FN ('''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['confution_matrix']['fn'])+''')</td></tr>'''
    tmp+='''            <tr><td>Negatif</td> <td>FP ('''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['confution_matrix']['fp'])+''')</td> <td>TN ('''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['confution_matrix']['tn'])+''')</td></tr>'''
    tmp+='''        </table>
                </div>
            </div>'''

    fiturEveryPartition=result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['fitur_every_partition']
    tmp+='''<div class="table-responsive" style="overflow-y: auto;">
                <table class="table  table-sm table-borderless">'''
    for i in (fiturEveryPartition.keys()):
        tmp+='''        <tr>'''
        tmp+='''            <td>partition_'''+str(i)+'''</td>'''
        tmp+='''            <td>:</td>'''
        tmp+='''            <td>'''+str(fiturEveryPartition[i])+'''</td>'''
        tmp+='''        </tr>'''
        
    tmp+='''    </table>
            </div> '''

    tmp+='''<div class="ml-1 mb-3">threshold : '''+str(result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['threshold'])+'''</div>'''


    tmp+='''<div class="row">'''
    tmp+='''    <div class="col-sm-6">'''
    tmp+='''        <div class="ml-1">prediction before threshold</div>'''
    tmp+='''        <div class="table-responsive" style="height: 250px;overflow-y: auto;">
                        <table class="table  table-sm">'''
    for h in result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['prediction_true_value']:
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['prediction_true_value'].values:
        tmp+='''            <tr>'''
        for col in row:
            tmp+='''            <td>'''+str(col)+'''</td>'''        
        tmp+='''            </tr>'''
    tmp+='''            </table>
                    </div> '''
    tmp+='''    </div>'''

    tmp+='''    <div class="col-sm-6">'''
    tmp+='''        <div class="ml-1">prediction after threshol and voting</div>'''
    tmp+='''        <div class="table-responsive" style="height: 250px;overflow-y: auto;">
                        <table class="table  table-sm">'''
    for h in result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['prediction_with_threshold']:
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lorens_nu_ohe[int(p[0])][p[1]]['prediction_with_threshold'].values:
        tmp+='''            <tr>'''
        for col in row:
            tmp+='''            <td>'''+str(col)+'''</td>'''        
        tmp+='''            </tr>'''
    tmp+='''            </table>
                    </div> '''
    tmp+='''    </div>'''
    tmp+='''</div> '''
    return jsonify(result=tmp)
    




@app2.route('/_nu_ohe_check_detail_logistic')
def check_detail_logistic():
    p = request.args.get('tmp_position')
    p=p.split('-')

    tmp=''''''
    tmp+='''<div class="table-responsive" style="overflow-y: auto;">
                <table class="table  table-sm table-borderless">'''
    tmp+='''        <tr><td>accuracy</td> <td>:</td> <td>'''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['akurasi'])+'''</td></tr>'''
    tmp+='''        <tr><td>precision</td> <td>:</td> <td>'''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['precision'])+'''</td></tr>'''
    tmp+='''        <tr><td>recall</td> <td>:</td> <td>'''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['recall'])+'''</td></tr>'''
    tmp+='''    </table>
            </div> '''

    tmp+='''        <div class="ml-1">confusion matrix</div>'''
    tmp+='''<div class="">
                <div class="table-responsive" style="overflow-y: auto;">
                    <table class="table  table-sm table-bordered">'''
    tmp+='''            <tr><td rowspan="2" class="align-middle">Kelas Aktual</td> <td colspan="2" class="text-center">Kelas Prediksi</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>Negatif</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>TP ('''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['tp'])+''')</td> <td>FN ('''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['fn'])+''')</td></tr>'''
    tmp+='''            <tr><td>Negatif</td> <td>FP ('''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['fp'])+''')</td> <td>TN ('''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['tn'])+''')</td></tr>'''
    tmp+='''        </table>
                </div>
            </div>'''

    tmp+='''<div class="ml-1 mb-3">threshold : '''+str(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['threshold'])+'''</div>'''

    tmp+='''<div class="row">'''
    tmp+='''    <div class="col-sm-12">'''
    tmp+='''        <div class="ml-1">prediction before threshold</div>'''
    tmp+='''        <div class="table-responsive" style="height: 250px;overflow-y: auto;">
                        <table class="table  table-sm">'''
    for h in list(result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['prediction_logistic_regression'].columns):
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lg_nu_ohe[int(p[1])]["logistic_regresion"]['prediction_logistic_regression'].values:
        tmp+='''            <tr>'''
        for col in row:
            tmp+='''            <td>'''+str(col)+'''</td>'''        
        tmp+='''            </tr>'''
    tmp+='''            </table>
                    </div> '''
    tmp+='''    </div>'''

    
    tmp+='''    <div class="col-sm-6">'''
    tmp+='''    </div>'''


    tmp+='''</div> '''
    return jsonify(result=tmp)