import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
import pandas as pd

# Anova feature selection
def select_anova(X_train_scaled: np.ndarray, 
                y_train:np.ndarray,
                X_test_scaled: np.ndarray,
                k:int=20,
                feature_names: list=None) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Keep the top-k features ranked by ANOVA F-score
    """

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    if feature_names is not None:
        selected = np.array(feature_names)[selector.get_support()]
        scores = selector.scores_[selector.get_support()]
        ranking = pd.DataFrame({
            "feature": selected,
            "F_score": scores,
        })
        ranking = ranking.sort_values("F_score", ascending=False)
        print(f"\nANOVA - top {k} features:")
        print(ranking.head(k).to_string(index=False))
    return X_train_sel, X_test_sel, selector


