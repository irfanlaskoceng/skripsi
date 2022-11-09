from flask import Flask, render_template, redirect, url_for, jsonify, request, session
from app2 import app2
from app3 import app3
from method import *

app = Flask(__name__)
app.secret_key = '5917a4a0a9961a2dd5a808e5311fd4ca5a5b2255acce0b9befc85a7afa54cd82'

app.register_blueprint(app2,url_prefix='')
app.register_blueprint(app3,url_prefix='')

df=pd.read_csv("heart.csv")
list_fitur_awal = df.columns.values.tolist()
list_data_awal  = df.values.tolist()
data_y = df.target.values.tolist()

numeric = ['age','trestbps', 'chol','thalach','oldpeak', 'ca']
nominal = ['sex', 'cp',  'fbs', 'restecg','exang', 'slope', 'thal']
dataset_one_hot_encoding            = oneHotEncoding(df,numeric,nominal)
list_dataset_one_hot_encoding_fitur = dataset_one_hot_encoding.columns.values.tolist()
list_dataset_one_hot_encoding       = dataset_one_hot_encoding.values.tolist()

dataset_min_max            = minMax(dataset_one_hot_encoding,0.1,0.9) 
list_dataset_min_max_fitur = dataset_min_max.columns.values.tolist()
list_dataset_min_max       = dataset_min_max.values.tolist()

default_columns=['fitur', 'ensemble_1', 'ensemble_2', 'ensemble_3', 'ensemble_4','ensemble_5', 'ensemble_6', 'ensemble_7', 'ensemble_8', 'ensemble_9', 'ensemble_10']
default_value=[['age', 3, 2, 1, 2, 3, 1, 3, 2, 2, 2], ['trestbps', 3, 2, 3, 2, 1, 3, 3, 3, 3, 2], ['chol', 3, 3, 1, 3, 3, 1, 1, 1, 3, 3], ['thalach', 3, 3, 3, 2, 1, 2, 3, 2, 1, 1], ['oldpeak', 1, 3, 2, 1, 3, 3, 1, 2, 3, 3], ['ca', 1, 1, 3, 3, 1, 2, 1, 3, 2, 2], ['sex_0', 2, 2, 3, 3, 3, 1, 1, 1, 1, 2], ['sex_1', 3, 2, 3, 1, 2, 2, 1, 1, 1, 1], ['cp_0', 2, 1, 1, 3, 2, 1, 1, 3, 2, 1], ['cp_1', 2, 2, 1, 2, 1, 3, 1, 1, 3, 2], ['cp_2', 2, 1, 2, 2, 1, 3, 3, 3, 3, 3], ['cp_3', 2, 2, 2, 2, 3, 2, 3, 1, 3, 3], ['fbs_0', 3, 1, 2, 2, 2, 2, 3, 3, 1, 2], ['fbs_1', 2, 2, 2, 2, 1, 1, 3, 2, 2, 1], ['restecg_0', 1, 3, 1, 2, 2, 3, 3, 2, 3, 3], ['restecg_1', 3, 3, 3, 1, 1, 3, 2, 3, 2, 3], ['restecg_2', 3, 2, 1, 1, 3, 2, 2, 1, 1, 3], ['exang_0', 1, 3, 1, 2, 3, 2, 1, 1, 1, 1], ['exang_1', 2, 3, 1, 2, 2, 1, 2, 3, 2, 2], ['slope_0', 1, 1, 1, 3, 3, 3, 2, 3, 3, 2], ['slope_1', 1, 3, 2, 3, 2, 1, 3, 3, 1, 3], ['slope_2', 3, 3, 1, 3, 2, 3, 3, 2, 1, 3], ['thal_0', 3, 2, 2, 1, 1, 3, 2, 2, 2, 3], ['thal_1', 1, 3, 3, 3, 2, 3, 3, 2, 2, 3], ['thal_2', 2, 2, 1, 2, 3, 1, 1, 2, 2, 3], ['thal_3', 2, 2, 2, 2, 3, 1, 2, 2, 3, 3]]

random_partition_in_ensemble=pd.DataFrame(default_value, columns=default_columns)
result_cross_validation_lorens=[]



@app.route("/")
def index():
    return render_template("tam.html", list_data_awal=list_data_awal,list_fitur_awal=list_fitur_awal, 
    list_dataset_one_hot_encoding_fitur=list_dataset_one_hot_encoding_fitur,list_dataset_one_hot_encoding=list_dataset_one_hot_encoding,
    list_dataset_min_max_fitur=list_dataset_min_max_fitur,list_dataset_min_max=list_dataset_min_max,
    default_columns=default_columns,default_value=default_value)

@app.route("/profil/<name>")
def profil(name=None):
    return render_template("tam.html",name=name)

@app.route('/_do_random_partition')
def do_random_partition():
    a = request.args.get('k_sub', 0, type=int)
    b = request.args.get('n_ens', 0, type=int)
    # return jsonify(result="<b>ggg %s</b>" %(2))
    global random_partition_in_ensemble
    random_partition_in_ensemble=generateRandomPartitionInEnsemble(dataset_min_max.columns.values, a, b)

    tmp='''
                <div class="card bgCARD">
                    <div class="table-responsive" style="height: 500px; overflow-y: auto;">
                        <table class="table  table-sm table-bordered">
                            <thead>
                                <tr>
    '''
    for x in random_partition_in_ensemble.columns.values.tolist():
        tmp=tmp+('''<th>'''+x+'''</th>''')
    tmp=tmp+'''
                                </tr>
    '''
    for x in random_partition_in_ensemble.values.tolist():
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

    pd.DataFrame([str(random_partition_in_ensemble.values.tolist())]).to_clipboard(index=False,header=False)

    print('clippppp')
    return jsonify(result=tmp)

@app.route('/_do_lorens')
def do_lorens():
    tmp_training = request.args.get('d_training', 0, type=int)
    tmp_fold = request.args.get('fold', 0, type=int) 
    cross_validation = crossValidation(list_dataset_min_max, data_y, list_dataset_min_max_fitur, tmp_training,tmp_fold)
    global result_cross_validation_lorens
    global result_cross_validation_lg
    result_cross_validation_lorens, result_cross_validation_lg = doCrossValidationLorens(cross_validation, random_partition_in_ensemble)
    result_mean=get_mean(result_cross_validation_lorens)
    result_best_ensemble=getBestEnsemble(result_mean)
    result_mean_lg=get_mean(result_cross_validation_lg)
    
    
    print(result_cross_validation_lorens[0]['ensemble_'+str(1)]['akurasi'])

    print('lorens=======>',len(result_cross_validation_lorens))
    print('lg=======>',len(result_cross_validation_lg))
    # print('lg=======>',result_cross_validation_lg)
    
    
    
    tmp=''''''
    for i in range(len(result_cross_validation_lorens)):
        print()
        tmp+='''<div class="row">'''
        tmp+='''<div class="col-sm-12"><b>fold '''+str(i+1)+'''</b></div>'''
        tmp+='''<div class="row ml-4">'''
        for ens_j in (random_partition_in_ensemble.columns):
            if ens_j != 'fitur':
                tmp+='''<div class="col-sm-4 mb-1">'''
                tmp+='''<b>'''+str(ens_j)+'''</b>'''
                tmp+='''<div>&nbsp akurasi : '''+str(result_cross_validation_lorens[i][ens_j]['akurasi'])+'''</div>'''
                tmp+='''<div>&nbsp precision : '''+str(result_cross_validation_lorens[i][ens_j]['precision'])+'''</div>'''
                tmp+='''<div>&nbsp recall : '''+str(result_cross_validation_lorens[i][ens_j]['recall'])+'''</div>'''
                tmp+='''<div>&nbsp <a class="btn btn-info btn-sm" role="button" id=s style="width: 60%; margi:0px" onclick='cekDetail("'''+str(i)+'''-'''+str(ens_j)+'''")'>detail...</a></div>'''
                tmp+='''</div>'''

        tmp+='''<div class="col-sm-4 mb-1">'''
        tmp+='''<b>logistic regression</b>'''
        tmp+='''<div>&nbsp akurasi : '''+str(result_cross_validation_lg[i]["logistic_regresion"]['akurasi'])+'''</div>'''
        tmp+='''<div>&nbsp precision : '''+str(result_cross_validation_lg[i]["logistic_regresion"]['precision'])+'''</div>'''
        tmp+='''<div>&nbsp recall : '''+str(result_cross_validation_lg[i]["logistic_regresion"]['recall'])+'''</div>'''
        tmp+='''<div>&nbsp <a class="btn btn-info btn-sm" role="button" id=s style="width: 60%; margi:0px" onclick='cekDetailLogistic("logistic-'''+str(i)+'''")'>detail...</a></div>'''
        tmp+='''</div>'''

        tmp+='''</div>'''
        tmp+='''</div>'''
    

    tmp+='''<div class="row">'''    
    tmp+='''<h5 class="mt-4 mb-2">Mean</h5> '''
    tmp+='''<div class="row ml-4">'''
    for ens_m in (result_mean):
        tmp+='''<div class="col-sm-4 mb-1">'''
        tmp+='''<b>'''+str(ens_m)+'''</b>'''
        tmp+='''<div>&nbsp akurasi : '''+str(result_mean[ens_m]['akurasi'])+'''</div>'''
        tmp+='''<div>&nbsp precision : '''+str(result_mean[ens_m]['precision'])+'''</div>'''
        tmp+='''<div>&nbsp recall : '''+str(result_mean[ens_m]['recall'])+'''</div>'''
        tmp+='''</div>'''
    tmp+='''<div class="col-sm-4 mb-1">'''
    tmp+='''<b>logistic regression</b>'''
    tmp+='''<div>&nbsp akurasi : '''+str(result_mean_lg['logistic_regresion']['akurasi'])+'''</div>'''
    tmp+='''<div>&nbsp precision : '''+str(result_mean_lg['logistic_regresion']['precision'])+'''</div>'''
    tmp+='''<div>&nbsp recall : '''+str(result_mean_lg['logistic_regresion']['recall'])+'''</div>'''
    tmp+='''</div>'''
    tmp+='''</div>'''
    tmp+='''</div>'''


    tmp+='''<div class="row">'''
    tmp+='''<div class="col-sm-12">'''
    tmp+='''<h5 class="mt-4 mb-2">Best ensemble ===> '''+str(result_best_ensemble)+'''</h5> '''
    
    
    tmp+='''</div>'''
    tmp+='''</div>'''
    return jsonify(result=tmp)


@app.route('/_check_detail')
def check_detail():
    p = request.args.get('tmp_position')
    p=p.split('-')

    tmp=''''''
    tmp+='''<div class="">
                <div class="table-responsive" style="overflow-y: auto;">
                    <table class="table  table-sm table-borderless">'''
    tmp+='''            <tr><td>accuracy</td> <td>:</td> <td>'''+str(result_cross_validation_lorens[int(p[0])][p[1]]['akurasi'])+'''</td></tr>'''
    tmp+='''            <tr><td>precision</td> <td>:</td> <td>'''+str(result_cross_validation_lorens[int(p[0])][p[1]]['precision'])+'''</td></tr>'''
    tmp+='''            <tr><td>recall</td> <td>:</td> <td>'''+str(result_cross_validation_lorens[int(p[0])][p[1]]['recall'])+'''</td></tr>'''
    tmp+='''        </table>
                </div>
            </div>'''

    tmp+='''        <div class="ml-1">confusion matrix</div>'''
    tmp+='''<div class="">
                <div class="table-responsive" style="overflow-y: auto;">
                    <table class="table  table-sm table-bordered">'''
    tmp+='''            <tr><td rowspan="2" class="align-middle">Kelas Aktual</td> <td colspan="2" class="text-center">Kelas Prediksi</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>Negatif</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>TP ('''+str(result_cross_validation_lorens[int(p[0])][p[1]]['confution_matrix']['tp'])+''')</td> <td>FN ('''+str(result_cross_validation_lorens[int(p[0])][p[1]]['confution_matrix']['fn'])+''')</td></tr>'''
    tmp+='''            <tr><td>Negatif</td> <td>FP ('''+str(result_cross_validation_lorens[int(p[0])][p[1]]['confution_matrix']['fp'])+''')</td> <td>TN ('''+str(result_cross_validation_lorens[int(p[0])][p[1]]['confution_matrix']['tn'])+''')</td></tr>'''
    tmp+='''        </table>
                </div>
            </div>'''

    fiturEveryPartition=result_cross_validation_lorens[int(p[0])][p[1]]['fitur_every_partition']
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

    tmp+='''<div class="row">'''
    tmp+='''    <div class="col-sm-6">'''
    tmp+='''        <div class="ml-1">prediction before threshold</div>'''
    tmp+='''        <div class="table-responsive" style="height: 250px;overflow-y: auto;">
                        <table class="table  table-sm">'''
    for h in result_cross_validation_lorens[int(p[0])][p[1]]['prediction_true_value']:
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lorens[int(p[0])][p[1]]['prediction_true_value'].values:
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
    for h in result_cross_validation_lorens[int(p[0])][p[1]]['prediction_with_threshold']:
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lorens[int(p[0])][p[1]]['prediction_with_threshold'].values:
        tmp+='''            <tr>'''
        for col in row:
            tmp+='''            <td>'''+str(col)+'''</td>'''        
        tmp+='''            </tr>'''
    tmp+='''            </table>
                    </div> '''
    tmp+='''    </div>'''
    tmp+='''</div> '''
    print('===============xsxsxs')
    print(result_cross_validation_lorens[1]['ensemble_5']['confution_matrix'])
    print('===============xsxsxs')
    
    return jsonify(result=tmp)



@app.route('/_check_detail_logistic')
def check_detail_logistic():
    p = request.args.get('tmp_position')
    p=p.split('-')

    tmp=''''''
    tmp+='''<div class="table-responsive" style="overflow-y: auto;">
                <table class="table  table-sm table-borderless">'''
    tmp+='''        <tr><td>accuracy</td> <td>:</td> <td>'''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['akurasi'])+'''</td></tr>'''
    tmp+='''        <tr><td>precision</td> <td>:</td> <td>'''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['precision'])+'''</td></tr>'''
    tmp+='''        <tr><td>recall</td> <td>:</td> <td>'''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['recall'])+'''</td></tr>'''
    tmp+='''    </table>
            </div> '''
    
    tmp+='''        <div class="ml-1">confusion matrix</div>'''
    tmp+='''<div class="">
                <div class="table-responsive" style="overflow-y: auto;">
                    <table class="table  table-sm table-bordered">'''
    tmp+='''            <tr><td rowspan="2" class="align-middle">Kelas Aktual</td> <td colspan="2" class="text-center">Kelas Prediksi</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>Negatif</td></tr>'''
    tmp+='''            <tr><td>Positif</td> <td>TP ('''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['tp'])+''')</td> <td>FN ('''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['fn'])+''')</td></tr>'''
    tmp+='''            <tr><td>Negatif</td> <td>FP ('''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['fp'])+''')</td> <td>TN ('''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['confution_matrix_logistic_regression']['tn'])+''')</td></tr>'''
    tmp+='''        </table>
                </div>
            </div>'''


    tmp+='''<div class="ml-1 mb-3">threshold : '''+str(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['threshold'])+'''</div>'''

    tmp+='''<div class="row">'''
    tmp+='''    <div class="col-sm-12">'''
    tmp+='''        <div class="ml-1">prediction before threshold</div>'''
    tmp+='''        <div class="table-responsive" style="height: 250px;overflow-y: auto;">
                        <table class="table  table-sm">'''
    for h in list(result_cross_validation_lg[int(p[1])]["logistic_regresion"]['prediction_logistic_regression'].columns):
        tmp+='''            <th>'''+str(h)+'''</th>'''        
    for row in result_cross_validation_lg[int(p[1])]["logistic_regresion"]['prediction_logistic_regression'].values:
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
    print(p)
    print(result_cross_validation_lg[int(p[1])]['logistic_regresion']['confution_matrix_logistic_regression'],'+++++++++')
    return jsonify(result=tmp)

















'''
==> cara install : pertama buat folder baruu kemudian masuk pada directori folder yg sudah di buat menggunakan cmd kemudian jalankan tiap baris code di bawah
py -3 -m venv venv
venv\Scripts\activate
pip install Flask

==> cara run di server local, masuk ke directori folder projek yg sudah di buat dengan cmd kemudian jalankan tiap baris code di bawah
venv\Scripts\activate
set FLASK_APP=app1.py
set FLASK_ENV=development
flask run 


jika mau run di port lain
==> flask run -h localhost -p 3000
'''