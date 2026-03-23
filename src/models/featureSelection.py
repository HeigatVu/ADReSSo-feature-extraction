import numpy as np
import pandas as pd

from scipy.stats import f_oneway
import operator

from sklearn.decomposition import PCA

# Using MRMR build based on https://github.com/elliotweissberg/Algorithms/blob/master/MRMr_Explained.ipynb
def get_idxs_by_class(df:pd.DataFrame, 
                    target_name:str) -> list[np.ndarray]:
    """ Group indices by class
    """
    result = []
    for value in df[target_name].unique():
        result.append(df[df[target_name] == value].index)

    return result

def get_features(df:pd.DataFrame, 
                target_name:str) -> list[str]:
    """ Get features from DataFrame
    """
    feature_names = []
    for col in df.columns:
        if col != target_name:
            feature_names.append(col)
    return feature_names

def calc_feature_relevance(df:pd.DataFrame, 
                            feature_name:str, 
                            idxs_by_class:list[np.ndarray]) -> float:
    """ Calculate feature relevance between two classes
    """
    
    result = []
    for class_idxs in idxs_by_class:
        result.append(df[feature_name][class_idxs].values)

    return f_oneway(*result).statistic

def calculate_all_feature_relevances(df:pd.DataFrame, 
                                    features:list[str], 
                                    idxs_by_class:list[np.ndarray]) -> dict[str, float]:
    """ Calculate feature relevance for all features
    """
    
    relevance = dict()
    for feat in features:
        relevance[feat] = calc_feature_relevance(df, feat, idxs_by_class)
    return relevance


def calculate_featrure_redundancy(df:pd.DataFrame,
                                feature:str,
                                ranked_features:list,
                                calculated_correlations:dict) -> int:
    """ Calculate feature redundancy between two classes
    """
    redundacy = 0
    for feat in ranked_features:
        key = (feat, feature)
        if key not in calculated_correlations:
            corr_value = abs(np.corrcoef(df[feature], df[feat])[1, 0])
            calculated_correlations[(feat, feature)] = corr_value
            calculated_correlations[(feature, feat)] = corr_value
        redundacy += calculated_correlations[key]
    return redundacy

def rank_features(df:pd.DataFrame,
                    target_name:str,
                    method="difference") -> list[str]:
    """ Rank features by relevance and redundancy 
    """
    use_differences = method == "difference"
    idxs_by_class = get_idxs_by_class(df, target_name)
    features = get_features(df, target_name)
    feature_relevance = calculate_all_feature_relevances(df, features, idxs_by_class)
    calculated_correlations = dict()
    ranked_features = []
    
    most_important_feature = max(feature_relevance.items(), key=operator.itemgetter(1))[0]
    ranked_features.append(most_important_feature)
    
    while len(ranked_features) != len(features):
        top_importance = float('-inf')
        most_important_feature = None
        for feat in features:
            if feat in ranked_features:
                continue

            redundancy = calculate_featrure_redundancy(df, feat, ranked_features, calculated_correlations)
            relevance = feature_relevance[feat]
            if use_differences:
                importance = relevance - redundancy
            else:
                importance = relevance / redundancy

            if importance > top_importance:
                top_importance = importance
                most_important_feature = feat

        ranked_features.append(most_important_feature)

    return ranked_features


# Using PCA
def pca_selection(X_train_scaled:np.ndarray, 
                    X_test_scaled:np.ndarray, 
                    n_components:float=0.95) -> tuple[np.ndarray, np.ndarray]:
    """ Select features using PCA
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_comp = pca.n_components_
    var = pca.explained_variance_ratio_.sum()
    print(f"\nPCA — {n_comp} components explain {var * 100:.1f}% of variance")
    print(f"  Reduced: {X_train_scaled.shape[1]} → {n_comp} dimensions")
    return X_train_pca, X_test_pca