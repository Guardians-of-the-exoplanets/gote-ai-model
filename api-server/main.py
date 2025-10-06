from datetime import datetime, timedelta
import functions_framework
import tempfile
from flask import Response, stream_with_context, request
import time
import json
import io
import pandas as pd
import xgboost as xgb
from google.cloud import storage
from functions import *
import json, time
import base64
import io
import zipfile
import time
import xgboost as xgb
from uuid import uuid4


# Baixa o modelo do Cloud Storage
def download_xgb_classifier_original(dataset_name: str,complete: bool) -> xgb.XGBClassifier:
    bucket_name = "vitor_ml"
    if complete:
        file_name = f"original_models/{dataset_name}_model_complete.model"
    else:
        file_name = f"original_models/{dataset_name}_model_lite.model"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    model_bytes = blob.download_as_bytes()

    # salva temporariamente no disco
    with tempfile.NamedTemporaryFile(delete=False,suffix=".model") as tmp:
        tmp.write(model_bytes)
        tmp.close()  # <- importante fechar antes de passar para load_model
        
        model = xgb.XGBClassifier()
        model.load_model(tmp.name)  # agora aceita o caminho
    return model

def generate_prediction_xgbclassifier_upload(dataset, user_data, hyperparameters):
    # Leitura do CSV
    cont=1
    yield json.dumps({"step": cont, "status": "Reading your CSV"}) + "\n"
    cont+=1
    # Decode Base64 into bytes

    zip_bytes = base64.b64decode(user_data)

    # Load as in-memory ZIP
    zip_file_like = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(zip_file_like, "r") as zip_ref:
        # Assuming there's only one CSV in the ZIP
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as csv_file:
            # Read directly into Pandas
            df = pd.read_csv(csv_file)
    time.sleep(1)

    # Pré-processamento
    yield json.dumps({"step": cont, "status": "Preprocessing your data"}) + "\n"
    cont+=1
    if(dataset=="k2"):
        ids = df['pl_name'].tolist()
        pubdate = df['pl_pubdate'].tolist()
        df_processed = k2_preprocess_complete(df)
    elif(dataset=="kepler"):
        ids = df['kepid'].tolist()
        kepoi_name = df['kepoi_name'].to_list()
        df_processed = kepler_preprocess_complete(df)
    elif (dataset == "tess"):
        toi = df['toi'].tolist()
        tid = df['tid'].tolist()
        df_processed = tess_preprocess_complete(df)

    else:
        return json.dumps({"status":"ainda em construção"})

    new_model = None
    if hyperparameters:
        yield json.dumps({"step": cont, "status": "Training a new model with you hyperparameters"}) + "\n"
        cont+=1
        container = {"cont": cont}
        mode = "complete"
        for step in train_new_hyperparameters(dataset, hyperparameters, container, cont, mode):
            yield step
        new_model = container["model"]
        cont = container['cont']
    else:
        # Download do Modelo
        yield json.dumps({"step": cont, "status": "Downloading the Machine Learning model"}) + "\n"
        cont+=1
        
    model = download_xgb_classifier_original(dataset, True)
    time.sleep(1)

    # Predict
    yield json.dumps({"step": cont, "status": "Predicting your data"}) + "\n"
    cont+=1
    
    # Mapenando labels

    preds = model.predict(df_processed)
    probas = model.predict_proba(df_processed)

    if(new_model):
        new_preds = new_model.predict(df_processed)
        new_probas = new_model.predict_proba(df_processed)

    result = []
    comparison = []
    if new_model:
        loop_iterable = zip(preds, new_preds)  # loop over both arrays
    else:
        loop_iterable = zip(preds,)    

    for i, items in enumerate(loop_iterable):
        if new_model:
            p, np_ = items
        else:
            (p,) = items
        old_class_name = class_mapping[int(p)]  # converte para int do Python
        old_prob = float(probas[i][int(p)] * 100)  # força float do Python
        if new_model:
            new_class_name = class_mapping[int(np_)]
            new_prob = float(new_probas[i][int(np_)] * 100)  # força 
            if(dataset == "k2"):
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "pubdate": str(pubdate[i]),
                    "old_classification": old_class_name,
                    "old_probability": round(old_prob, 2)
                })
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "pubdate": str(pubdate[i]),
                    "new_classification": new_class_name,
                    "new_probability": round(new_prob, 2)
                })
            elif(dataset == "kepler"):
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "kepoi_name": str(kepoi_name[i]),
                    "old_classification": old_class_name,
                    "old_probability": round(old_prob, 2)
                })
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "kepoi_name": str(kepoi_name[i]),
                    "new_classification": new_class_name,
                    "new_probability": round(new_prob, 2)
                })
            elif (dataset == "tess"):
                comparison.append({
                    "toi": str(toi[i]),  # garante que seja string
                    "tid": str(tid[i]),
                    "old_classification": old_class_name,
                    "old_probability": round(old_prob, 2)
                })
                comparison.append({
                    "toi": str(toi[i]),  # garante que seja string
                    "tid": str(tid[i]),
                    "new_classification": new_class_name,
                    "new_probability": round(new_prob, 2)
                })
        else:
            if(dataset == "k2"):
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "pubdate": str(pubdate[i]),
                    "classification": old_class_name,
                    "probability": round(old_prob, 2)
                })
            elif(dataset == "kepler"):
                comparison.append({
                    "id": str(ids[i]),  # garante que seja string
                    "kepoi_name": str(kepoi_name[i]),
                    "classification": old_class_name,
                    "probability": round(old_prob, 2)
                })
            elif(dataset == "tess"):
                comparison.append({
                    "toi": str(toi[i]),  # garante que seja string
                    "tid": str(tid[i]),
                    "classification": old_class_name,
                    "probability": round(old_prob, 2)
                })
        result.append(comparison.copy())
        comparison.clear()

    time.sleep(1)

    # Resultado Final
    yield json.dumps({"step": cont, "status": "Your prediction is done!", "predictions": result}) + "\n"

def generate_prediction_xgbclassifier_manual(dataset, user_data, hyperparameters):
    cont=1
    ## Validação do user data
    if(dataset=="k2"):
        overwrite_selected_features = ['sy_pnum','soltype','pl_orbper','sy_vmag','sy_kmag','sy_gaiamag','st_rad','sy_dist']
        # Verifica se todas as colunas estão presentes
        all_present = all(col in user_data for col in overwrite_selected_features) and (isinstance(user_data['sy_pnum'],int) and isinstance(user_data['soltype'],str) and isinstance(user_data['pl_orbper'],float))
        
        if user_data['soltype'] not in ['Published Confirmed', 'Published Candidate', 'TESS Project Candidate']:
            yield json.dumps({"status": "error", "details":"Looks like your soltype field have a invalid value, take a look on it and send to us again!"}) + "\n"
            return
    elif(dataset == "kepler"):
        all_present = all(col in user_data for col in features_kepler_lite)
    elif(dataset == "tess"):
        all_present = all(col in user_data for col in features_tess_lite)
    if not all_present:
            yield json.dumps({"status": "error", "details":"Looks like your data is missing one or more mandatory fields, take a look on it and send to us again!"}) + "\n"
            return



    # if(hyper):
    yield json.dumps({"step": cont, "status": "Reading your manual data"}) + "\n"
    cont+=1
    df = pd.DataFrame([user_data])
    time.sleep(1)

    # Pré-processamento
    yield json.dumps({"step": cont, "status": "Preprocessing your data"}) + "\n"
    cont+=1
    if(dataset=="k2"):
        df_processed = k2_preprocess_manual(df)
    elif(dataset=="kepler"):
        df_processed = kepler_preprocess_manual(df)
    elif(dataset == "tess"):
        df_processed = tess_preprocess_manual(df)

    new_model = None
    if hyperparameters:
        yield json.dumps({"step": cont, "status": "Training a new model with you hyperparameters"}) + "\n"
        cont+=1
        container = {"cont": cont}
        mode = "lite"
        for step in train_new_hyperparameters(dataset, hyperparameters, container, cont, mode):
            yield step
        new_model = container["model"]
        cont = container['cont']
    else:
        # Download do Modelo
        yield json.dumps({"step": cont, "status": "Downloading the Machine Learning model"}) + "\n"
        cont+=1

    model = download_xgb_classifier_original(dataset, False)
    time.sleep(1)

    # Predict
    yield json.dumps({"step": cont, "status": "Predicting your data"}) + "\n"
    cont+=1

    preds = model.predict(df_processed)
    probas = model.predict_proba(df_processed)

    if(new_model):
        new_preds = new_model.predict(df_processed)
        new_probas = new_model.predict_proba(df_processed)

    result = []
    comparison = []
    if new_model:
        loop_iterable = zip(preds, new_preds)  # loop over both arrays
    else:
        loop_iterable = zip(preds,)    

    for i, items in enumerate(loop_iterable):
        if new_model:
            p, np_ = items
        else:
            (p,) = items
        old_class_name = class_mapping[int(p)]  # converte para int do Python
        old_prob = float(probas[i][int(p)] * 100)  # força float do Python
        if new_model:
            new_class_name = class_mapping[int(np_)]
            new_prob = float(new_probas[i][int(np_)] * 100)  # força 
            if(dataset == "k2" or dataset == "kepler" or dataset == "tess"):
                comparison.append({
                    "old_classification": old_class_name,
                    "old_probability": round(old_prob, 2)
                })
                comparison.append({
                    "new_classification": new_class_name,
                    "new_probability": round(new_prob, 2)
                })
        else:
            if(dataset == "k2" or dataset == "kepler" or dataset == "tess"):
                comparison.append({
                    "classification": old_class_name,
                    "probability": round(old_prob, 2)
                })
        result.append(comparison.copy())
        comparison.clear()

    time.sleep(1)

    # Resultado Final
    yield json.dumps({"step": cont, "status": "Your prediction is done!", "predictions": result}) + "\n"

def retrain_model(dataset, user_data, hyperparameters):
    start = time.time()
    cont=0
    yield json.dumps({"step": cont, "status": "Downloading original dataset"}) + "\n"
    cont+=1
    old_model = download_xgb_classifier_original(dataset, True)

    X_train_full = pd.read_csv(f'./data/{dataset}/X_train_full.csv')
    X_test = pd.read_csv(f'./data/{dataset}/X_test.csv')
    X_blind = pd.read_csv(f'./data/{dataset}/X_blind.csv')
    y_train_full = pd.read_csv(f'./data/{dataset}/y_train_full.csv')
    y_test = pd.read_csv(f'./data/{dataset}/y_test.csv')
    y_blind = pd.read_csv(f'./data/{dataset}/y_blind.csv')

    yield json.dumps({"step": cont, "status": "Processing new user data"}) + "\n"
    cont += 1
    # Decoding Base64 and zip
    user_bytes = base64.b64decode(user_data)
    zip_file = io.BytesIO(user_bytes)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        csv_name = zip_ref.namelist()[0]
        with zip_ref.open(csv_name) as f:
            df = pd.read_csv(f)

    reverse_mapping = {v: k for k, v in class_mapping.items()}

    if dataset == "k2":
        y_new = df[["disposition"]].rename(columns={"disposition": "label"})
        y_new["label"] = y_new["label"].map(reverse_mapping)
        selected_features = features_k2_complete
        df_processed = k2_preprocess_complete(df)
    elif dataset == "kepler":
        y_new = df[["koi_disposition"]].rename(columns={"koi_disposition": "label"})
        y_new["label"] = y_new["label"].map(reverse_mapping)
        selected_features = features_kepler_complete 
        df_processed = kepler_preprocess_complete(df)
    elif dataset == "tess":
        y_new = df[["tfopwg_disp"]].rename(columns={"tfopwg_disp": "label"})
        y_new["label"] = y_new["label"].map(reverse_mapping)
        selected_features = features_tess_complete 
        df_processed = tess_preprocess_complete(df)

    X_train_sel = X_train_full[selected_features]
    X_test_sel = X_test[selected_features]
    X_blind_sel = X_blind[selected_features]
    X_new = df_processed[selected_features]
    print(X_train_sel.shape,X_test_sel.shape,X_blind_sel.shape, X_new.shape)

    X_train_combined = pd.concat([X_train_sel, X_new], ignore_index=True)
    y_train_combined = pd.concat([y_train_full, y_new], ignore_index=True)
    print("X:",X_train_sel.shape, X_new.shape, X_train_combined.shape)
    print("Y:",y_train_full.shape,y_new.shape,y_train_combined.shape)
    print(y_train_combined['label'].unique())

    y_pred_old = old_model.predict(X_test_sel)
    old_acc_test = accuracy_score(y_test, y_pred_old)
    old_f1_test = f1_score(y_test, y_pred_old, average="weighted")

    y_pred_old_blind = old_model.predict(X_blind_sel)
    old_acc_blind = accuracy_score(y_blind, y_pred_old_blind)
    old_f1_blind = f1_score(y_blind, y_pred_old_blind, average="weighted")

    yield json.dumps(
        {
            "step": cont,
            "status": f"Old model - Metrics",
            "metrics": {
                "test": {
                    "accuracy": f"{old_acc_test:.4f}",
                    "f1_score": f"{old_f1_test:.4f}"
                },
                "blind_test": {
                    "accuracy": f"{old_acc_blind:.4f}",
                    "f1_score": f"{old_f1_blind:.4f}"
                },
            }
        }
    ) + "\n"
    cont += 1

    yield json.dumps({"step": cont, "status": "Training new model with Stratified K-Fold"}) + "\n"
    cont += 1

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    yield json.dumps({"step": cont, "status": f"Starting StratifiedKFold ({n_splits} folds)"}) + "\n"
    cont += 1

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train_combined), 1):
        X_train, X_val = X_train_combined.iloc[train_idx], X_train_combined.iloc[val_idx]
        y_train, y_val = y_train_combined.iloc[train_idx], y_train_combined.iloc[val_idx]

        model = xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train.values, y_train.values)

        y_pred_val = model.predict(X_val.values)
        acc = accuracy_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, average='weighted')
        precision = precision_score(y_val, y_pred_val, average='weighted')
        recall = recall_score(y_val, y_pred_val, average='weighted')

        fold_results.append({
            "fold": fold, "accuracy": acc, "f1": f1,
            "precision": precision, "recall": recall
        })
        yield json.dumps(
            {
                "step": cont,
                "status": f"Fold {fold}",
                "metrics": {
                    "accuracy": f"{acc:.4f}",
                    "f1_score": f"{f1:.4f}",
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}"
                }
            }
        ) + "\n"
        cont += 1

        acc_mean = np.mean([r["accuracy"] for r in fold_results])
        f1_mean = np.mean([r["f1"] for r in fold_results])
        yield json.dumps(
            {
                "step": cont,
                "status": f"KFold score after fold {fold}",
                "metrics": {
                    "mean_accuracy": f"{acc_mean:.4f}",
                    "mean_f1_score": f"{f1_mean:.4f}"
                }
            }
        ) + "\n"
        cont += 1

    new_model = xgb.XGBClassifier(**hyperparameters)
    new_model.fit(X_train_combined, y_train_combined)

    # Evaluate on test + blind
    y_pred_new_test = new_model.predict(X_test_sel)
    y_pred_new_blind = new_model.predict(X_blind_sel)

    new_acc_test = accuracy_score(y_test, y_pred_new_test)
    new_f1_test = f1_score(y_test, y_pred_new_test, average='weighted')
    new_acc_blind = accuracy_score(y_blind, y_pred_new_blind)
    new_f1_blind = f1_score(y_blind, y_pred_new_blind, average='weighted')

    yield json.dumps(
        {
            "step": cont,
            "status": f"New model - Test",
            "metrics": {
                "test": {
                    "accuracy": f"{new_acc_test:.4f}",
                    "f1_score": f"{new_f1_test:.4f}"
                },
                "blind_test": {
                    "accuracy": f"{new_acc_blind:.4f}",
                    "f1_score": f"{new_f1_blind:.4f}"
                },
            }
        }
    ) + "\n"
    cont += 1

    # if improved_test and blind_not_worse:
    yield json.dumps({"step": cont, "status": "New model approved - sending to Google Cloud Storage (in memory)"}) + "\n"
    cont += 1

    # Cria arquivo temporário em /tmp
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as tmp_file:
        temp_path = tmp_file.name
        new_model.save_model(temp_path)  # salva modelo XGBoost em .model binário

    # Upload para GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket("vitor_ml")
    file_id = uuid4()
    blob_path = f"custom_models/{dataset}/{file_id}_xgb_model_new.model"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(temp_path, content_type="application/octet-stream")
    print("UPOU")

    url = f"https://storage.googleapis.com/vitor_ml/{blob_path}"

    yield json.dumps({
        "step": cont,
        "status": "Upload completed successfully",
        "url_for_download": url,
        "training_time_in_seconds": f"{time.time() - start:.2f}"
    }) + "\n"

def health():
    yield json.dumps({"status":"ok"})  + "\n"
    time.sleep(2)
    yield json.dumps({"status":"encerrado"})
    
@functions_framework.http
def main(request):
    data = request.get_json(silent=True) or {}
    endpoint = data.get("endpoint") 
    dataset = data.get("dataset") # espera {"endpoint": "predict"} ou {"endpoint": "health"}
    user_data = data.get("data")  # deve ser uma lista de registros ou dict
    hyperparameters = data.get("hyperparameters")  # deve ser uma lista de registros ou dict
    print("ENDPOINT", endpoint)
    print("HYPER",hyperparameters)
    print("User data:", user_data)
    if(dataset == "k2"):
        if(isinstance(user_data,dict)):
            for key in ["pl_orbper", "sy_vmag","sy_kmag","sy_gaiamag","st_rad","sy_dist"]:
                if key in user_data and user_data[key] is not None:
                    user_data[key] = float(user_data[key])
            print("User data after conversion to float:", user_data)
    elif(dataset == "kepler"):
        if(isinstance(user_data,dict)):
            for key in ["koi_model_snr","koi_prad","koi_duration_err1","koi_steff_err1","koi_steff_err2"]:
                if key in user_data and user_data[key] is not None:
                    user_data[key] = float(user_data[key])
            print("User data after conversion to float:", user_data)
    elif(dataset == "tess"):
        if(isinstance(user_data,dict)):
            for key in user_data.keys():
                if key in user_data and user_data[key] is not None:
                    user_data[key] = float(user_data[key])
            print("User data after conversion to float:", user_data)
    else:
        return json.dumps({"error": "dataset must be one of kepler, tess, k2"}), 400

    if not user_data:
        return json.dumps({"error": "user data is required"}), 400
    
    # Rota 2 - Upload Manual de arquivo CSV
    if endpoint == "predict_upload":
        if hyperparameters:
            return Response(stream_with_context(generate_prediction_xgbclassifier_upload(dataset, user_data, hyperparameters)), mimetype="application/json")
        return Response(stream_with_context(generate_prediction_xgbclassifier_upload(dataset, user_data, None)), mimetype="application/json")
    elif endpoint == "predict_manual":
        if hyperparameters:
            return Response(stream_with_context(generate_prediction_xgbclassifier_manual(dataset, user_data, hyperparameters)), mimetype="application/json")
        return Response(stream_with_context(generate_prediction_xgbclassifier_manual(dataset, user_data, None)), mimetype="application/json")
    elif endpoint == "retrain":
        if hyperparameters:
            return Response(stream_with_context(retrain_model(dataset, user_data, hyperparameters)), mimetype="application/json")
        return Response(stream_with_context(generate_prediction_xgbclassifier_manual(dataset, user_data, None)), mimetype="application/json")
    elif endpoint == "health":
        return Response(stream_with_context(health()), mimetype="application/json")
    return "Not Found", 404