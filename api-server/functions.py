from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
from time import time
import json
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import io
from google.cloud import storage
import time
# Não enviar a coluna Disposition, se for enviada, será desconsiderada
features_k2_lite = ['sy_pnum', 'soltype_encoded', 'pl_orbper', 'sy_vmag', 'sy_kmag', 'sy_gaiamag', 'st_rad']
features_k2_complete = ['sy_pnum', 'pl_radelim', 'pl_radjerr1', 'default_flag', 'st_meterr1', 'st_met', 'sy_gaiamagerr1', 'soltype_encoded', 'pl_radeerr1', 'st_loggerr2', 'dec', 'pl_orbperlim', 'pl_rade_st_ratio', 'st_mass', 'pl_orbpererr2', 'pl_rade_uncertainty', 'st_teff', 'ra', 'pl_radeerr2', 'st_logg', 'st_rad', 'vmag_minus_kmag']

features_kepler_lite = ['koi_fpflag_co', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_model_snr', 'koi_prad', 'koi_duration_err1', 'koi_steff_err1', 'koi_steff_err2']
features_kepler_complete = ['koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ss', 'koi_fpflag_ec', 'planet_star_volume_ratio', 'koi_model_snr', 'koi_duration_err2', 'koi_steff_err2', 'koi_prad', 'koi_steff_err1', 'koi_period_err2', 'koi_duration_err1', 'koi_time0bk_err2', 'koi_prad_err2', 'planet_star_radius_ratio', 'koi_depth_err2', 'koi_tce_plnt_num', 'koi_insol', 'koi_period', 'temp_ratio_planet_star', 'koi_prad_err1', 'koi_insol_err1', 'koi_teq', 'koi_period_err1']

features_tess_complete = ['st_dist', 'st_tmag', 'pl_eqt', 'pl_insol', 'st_disterr2', 'st_disterr1', 'pl_rade', 'pl_tranmid', 'pl_tranmiderr2', 'st_raderr2', 'pl_radeerr2', 'st_loggerr2', 'pl_radeerr1', 'pl_orbper', 'pl_tranmiderr1', 'st_loggerr1', 'pl_orbpererr2', 'pl_trandep', 'pl_trandeperr1', 'pl_trandurh', 'pl_orbpererr1', 'st_tefferr2', 'st_logg', 'pl_trandurherr1', 'dec', 'pl_trandurherr2', 'st_rad', 'st_pmraerr1', 'st_teff', 'ra', 'st_tefferr1', 'st_raderr1', 'st_tmagerr1', 'pl_trandeperr2', 'st_pmdecerr1']
features_tess_lite = ["st_dist","st_tmag", "pl_eqt","pl_insol","st_disterr2", "st_disterr1","pl_rade","pl_tranmid","pl_tranmiderr2","st_raderr2"]

class_mapping = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE POSITIVE"}

def k2_preprocess_complete(df):
    if 'disposition' in df:
        df.drop(['disposition'],axis=1,inplace=True)
    df.drop(['loc_rowid','pl_name','hostname','disp_refname','disc_year',
    'pl_refname','st_refname','sy_refname','st_spectype','pl_bmassprov','rastr','decstr','rowupdate','pl_pubdate','releasedate'],axis=1,inplace=True)

    encoder = LabelEncoder()
    # Aplicar no label
    df["discoverymethod_encoded"] = encoder.fit_transform(df["discoverymethod"])
    df["soltype_encoded"] = encoder.fit_transform(df["soltype"])
    df["disc_facility_encoded"] = encoder.fit_transform(df["disc_facility"])

    # Fill NaN with placeholder
    df["st_metratio"] = df["st_metratio"].fillna("missing")
    df["st_metratio_encoded"] = encoder.fit_transform(df["st_metratio"])
    df.drop(['st_metratio','discoverymethod','soltype','disc_facility'],axis=1,inplace=True)
    
    for col in ["discoverymethod_encoded", "disc_facility_encoded", "soltype_encoded"]:
        if df[col].isnull().any():
            df.fillna({col: df[col].mode()[0]}, inplace=True)
    df = df.fillna(df.mean())
    
    
    # 1. Relações estrela–planeta
    # Razão raio do planeta pelo raio da estrela
    df["pl_rade_st_ratio"] = df["pl_rade"] / df["st_rad"]

    # Densidade aproximada (massa/raio³) para estrela e planeta
    # (valores em unidades relativas, já que dados vêm em raio solar/jupiteriano/terrestre)
    df["st_density_est"] = df["st_mass"] / (df["st_rad"] ** 3)
    df["pl_density_est"] = np.where(df["pl_rade"] > 0, df["pl_radj"] / (df["pl_rade"] ** 3), np.nan)

    # Temperatura de equilíbrio planetária aproximada
    # Usando 3ª Lei de Kepler: a ≈ ( (P² * M*)^(1/3) )
    # P em dias → converter para segundos
    G = 6.67430e-11  # constante gravitacional
    M_sun = 1.98847e30  # massa do Sol em kg
    R_sun = 6.9634e8    # raio do Sol em m

    # Transformar unidades aproximadas (massa e raio em solares, período em dias)
    P_sec = df["pl_orbper"] * 24 * 3600
    M_star_kg = df["st_mass"] * M_sun
    a_m = ((G * M_star_kg * (P_sec ** 2)) / (4 * (np.pi ** 2))) ** (1/3)

    R_star_m = df["st_rad"] * R_sun
    Teff = df["st_teff"]

    df["pl_eq_temp"] = Teff * np.sqrt(R_star_m / (2 * a_m))

    # 2. Magnitudes combinadas (cores estelares aproximadas)
    df["vmag_minus_kmag"] = df["sy_vmag"] - df["sy_kmag"]
    df["vmag_minus_gaiamag"] = df["sy_vmag"] - df["sy_gaiamag"]

    # 3. Features de incerteza
    df["pl_orbper_uncertainty"] = df["pl_orbpererr1"].abs() + df["pl_orbpererr2"].abs()
    df["pl_rade_uncertainty"] = df["pl_radeerr1"].abs() + df["pl_radeerr2"].abs()
    df["st_teff_uncertainty"] = df["st_tefferr1"].abs() + df["st_tefferr2"].abs()
    df["st_rad_uncertainty"] = df["st_raderr1"].abs() + df["st_raderr2"].abs()
    df["st_mass_uncertainty"] = df["st_masserr1"].abs() + df["st_masserr2"].abs()
    df["sy_dist_uncertainty"] = df["sy_disterr1"].abs() + df["sy_disterr2"].abs()
    df["sy_vmag_uncertainty"] = df["sy_vmagerr1"].abs() + df["sy_vmagerr2"].abs()
    df["sy_kmag_uncertainty"] = df["sy_kmagerr1"].abs() + df["sy_kmagerr2"].abs()
    df["sy_gaiamag_uncertainty"] = df["sy_gaiamagerr1"].abs() + df["sy_gaiamagerr2"].abs()
    
    df = df[features_k2_complete]

    return df

def k2_preprocess_manual(df):    
    encoder = LabelEncoder()
    df["soltype_encoded"] = encoder.fit_transform(df["soltype"])
    df.drop(['soltype'],axis=1,inplace=True)
    df = df[features_k2_lite]
    return df

def kepler_preprocess_complete(df):
    df.drop(['koi_teq_err1','koi_teq_err2','kepler_name',
         'loc_rowid','kepid','kepoi_name','koi_pdisposition','koi_score'],axis=1,inplace=True)
    encoder = LabelEncoder()

    # Aplicar no label
    df["label"] = encoder.fit_transform(df["koi_disposition"])
    encoder = LabelEncoder()
    # Fill NaN with placeholder
    df["koi_tce_delivname"] = df["koi_tce_delivname"].fillna("missing")
    df["tce_name_encoded"] = encoder.fit_transform(df["koi_tce_delivname"])
    df.drop(['koi_tce_delivname','koi_disposition','tce_name_encoded'],axis=1,inplace=True)
    df = df.fillna(df.mean())

    df['planet_star_radius_ratio'] = df['koi_prad'] / df['koi_srad']
    df['planet_star_volume_ratio'] = (df['koi_prad'] / df['koi_srad']) ** 3
    df['transit_density_proxy'] = df['koi_depth'] / df['koi_duration']
    df['temp_ratio_planet_star'] = df['koi_teq'] / df['koi_steff']

    scaler = StandardScaler()
    new_features = ['planet_star_radius_ratio', 'planet_star_volume_ratio', 'transit_density_proxy', 'temp_ratio_planet_star']
    df[new_features] = scaler.fit_transform(df[new_features])

    df = df[features_kepler_complete]
    return df

def kepler_preprocess_manual(df):
    df = df[features_kepler_lite]
    return df

def tess_preprocess_complete(df):
    if 'tfopwg_disp' in df:
        df.drop(['tfopwg_disp'],axis=1,inplace=True)
    
    df.drop(['loc_rowid','tid','toi','rastr','decstr','pl_insolerr1','pl_insolerr2',
         'pl_insollim','pl_eqterr1','pl_eqterr2','pl_eqtlim','toi_created',
         'rowupdate'
    ],axis=1,inplace=True)
    df = df.fillna(df.mean())
    df = df[features_tess_complete]
    return df

def tess_preprocess_manual(df):
    return df[features_tess_lite]
    
def train_new_hyperparameters(dataset, hyperparameters, container,cont, mode):
    cont = container["cont"]
    yield json.dumps({"step": cont, "status": "Starting training with new Hyperparameters"}) + "\n"
    cont+=1
    
    start = time.time()
    X_train_full = pd.read_csv(f'./data/{dataset}/X_train_full.csv')
    X_test = pd.read_csv(f'./data/{dataset}/X_test.csv')
    X_blind = pd.read_csv(f'./data/{dataset}/X_blind.csv')

    y_train_full = pd.read_csv(f'./data/{dataset}/y_train_full.csv')
    y_test = pd.read_csv(f'./data/{dataset}/y_test.csv')
    y_blind = pd.read_csv(f'./data/{dataset}/y_blind.csv')

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        print(f"\nFold {fold}")
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        model = xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train.values, y_train.values)

        y_pred_val = model.predict(X_val.values)
        acc = accuracy_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, average='weighted')
        precision = precision_score(y_val, y_pred_val, average='weighted')
        recall = recall_score(y_val, y_pred_val, average='weighted')
        print(f"Validation Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'f1_score': f1,
            'recall': recall,
            'precision': precision
        })

    acc_mean = np.mean([r['accuracy'] for r in fold_results])
    f1_mean = np.mean([r['f1_score'] for r in fold_results])
    print(f"\nMédia Validation Accuracy: {acc_mean:.4f}")
    print(f"Média Validation F1-score: {f1_mean:.4f}")

    final_model = xgb.XGBClassifier(**hyperparameters)
    final_model.fit(X_train_full.values, y_train_full.values)

    feat_imp = pd.Series(final_model.feature_importances_, index=X_train_full.columns).sort_values(ascending=False)
    print("\nTop 10 Feature importances:")
    print(feat_imp.head(10))

    # Avaliação no teste
    y_pred_test = final_model.predict(X_test.values)
    print("\n--- Base Model Test Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_test, average='weighted'):.4f}")

    # Avaliação na blind validation
    y_pred_blind = final_model.predict(X_blind.values)
    print("\n--- Base Model Blind Results ---")
    print(f"Accuracy: {accuracy_score(y_blind, y_pred_blind):.4f}")
    print(f"F1-score: {f1_score(y_blind, y_pred_blind, average='weighted'):.4f}")

    results_summary = []

    if (dataset=="k2"):
        if(mode == "lite"):
            selected_features = features_k2_lite
        else:
            selected_features = features_k2_complete
    elif(dataset == "kepler"):
        if(mode == "lite"):
            selected_features = features_kepler_lite
        else:
           selected_features = features_kepler_complete 
    elif(dataset == "tess"):
        if(mode == "lite"):
            selected_features = features_tess_lite
        else:
           selected_features = features_tess_complete 

    X_train_sel = X_train_full[selected_features]
    X_test_sel = X_test[selected_features]
    X_blind_sel = X_blind[selected_features]

    model_sel = xgb.XGBClassifier(**hyperparameters)
    model_sel.fit(X_train_sel.values, y_train_full.values)

    # Test set
    y_pred_test_sel = model_sel.predict(X_test_sel.values)
    acc_test = accuracy_score(y_test, y_pred_test_sel)
    f1_test = f1_score(y_test, y_pred_test_sel, average='weighted')
    precision_test = precision_score(y_test, y_pred_test_sel, average='weighted')
    recall_test = recall_score(y_test, y_pred_test_sel, average='weighted')

    # Suponha que você já tenha y_test e y_pred_test_sel
    
    cm_test = confusion_matrix(y_test, y_pred_test_sel)
    print(cm_test)
    y_test_values = y_test.values.flatten() 
    classes_test = sorted(list(set(y_test_values)))  # lista de classes únicas
    cm_test_dict = {}
    for i, true_class in enumerate(classes_test):
        true_name = class_mapping[true_class]
        cm_test_dict[true_name] = {}
        for j, pred_class in enumerate(classes_test):
            pred_name = class_mapping[pred_class]
            cm_test_dict[true_name][pred_name] = int(cm_test[i, j])

    # Blind set
    y_pred_blind_sel = model_sel.predict(X_blind_sel.values)
    acc_blind = accuracy_score(y_blind, y_pred_blind_sel)
    f1_blind = f1_score(y_blind, y_pred_blind_sel, average='weighted')
    precision_blind = precision_score(y_blind, y_pred_blind_sel, average='weighted')
    recall_blind = recall_score(y_blind, y_pred_blind_sel, average='weighted')

    cm_blind = confusion_matrix(y_blind, y_pred_blind_sel)
    print(cm_blind)
    y_blind_values = y_blind.values.flatten()
    classes_blind = sorted(list(set(y_blind_values)))  # lista de classes únicas
    cm_blind_dict = {}
    for i, true_class in enumerate(classes_blind):
        true_name = class_mapping[true_class]
        cm_blind_dict[true_name] = {}
        for j, pred_class in enumerate(classes_blind):
            pred_name = class_mapping[pred_class]
            cm_blind_dict[true_name][pred_name] = int(cm_blind[i, j])

    print(f"Test Accuracy: {acc_test:.4f} | F1: {f1_test:.4f}")
    print(f"Blind Accuracy: {acc_blind:.4f} | F1: {f1_blind:.4f}")
    
    end = time.time()
    end_time = round(end - start, 2)

    print(f"\nTotal runtime: {end_time} seconds")

    results_summary.append({
        'n_features': len(selected_features),
        'fold_metrics': fold_results,
        'test_metrics':{
            "accuracy": acc_test,
            "f1_score": f1_test,
            "precision": precision_test,
            "recall": recall_test,
            'confusion_matrix': cm_test_dict,
        },
        'blind_metrics': {
            'accuracy': acc_blind,
            'f1_score': f1_blind,
            "precision": precision_blind,
            "recall": recall_blind,
            "confusion_matrix": cm_blind_dict,
        },
        'Training_Test_Total_Time': end_time,
    })

    container["model"] = model_sel 
    yield json.dumps({"step": cont, "status": "Training with new Hyperparameters finished without errors","details":results_summary}) + "\n"
    cont+=1
    container["cont"] = cont
