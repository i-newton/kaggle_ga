from utils import get_fold
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
MAX_TOWERS = 6
from pipelines import get_cont_cols, get_coord_cols


class Ensembler:

    def __init__(self):
        ridge = Ridge(random_state=17, alpha=18)
        rtree = RandomForestRegressor(criterion="mae",
                                      n_jobs=-1, random_state=19)
        xgboost = xgb.XGBRegressor(random_state=23, n_jobs=-1)
        lasso = Lasso(random_state=29, tol=0.001, alpha=0.55)

        self.cont_predictors = {"lasso": lasso, "ridge": ridge}
        self.cat_predictors = {"rtree": rtree, "xgboost": xgboost}
        self.stack_predictor = xgb.XGBRegressor(random_state=37, n_jobs=-1)

    @staticmethod
    def _get_preds(X, y, preds):
        predicts = []
        for cl in preds.values():
            predicts.append(
                cross_val_predict(cl, X, y, n_jobs=-1, cv=get_fold()))
        return pd.DataFrame(np.vstack(predicts).transpose(), index=y.index,
                            columns=preds.keys())

    def fit(self, X_train, y_train):
        #first get predictions
        cont_cols = get_cont_cols()
        X_cont = X_train[cont_cols]
        cat_cols = list(set(X_train.columns) - set(cont_cols))
        X_cat = X_train[cat_cols]
        cont_predicts = self._get_preds(X_cont, y_train,
                                        self.cont_predictors)
        for rgr in self.cont_predictors.values():
            rgr.fit(X_cont, y_train)

        cat_predicts = self._get_preds(X_cat, y_train,
                                       self.cat_predictors)
        for rgr in self.cat_predictors.values():
            rgr.fit(X_cat, y_train)
        X_meta = pd.concat([cont_predicts, cat_predicts], axis=1)
        #fit stack regressor
        self.stack_predictor.fit(X_meta, y_train, eval_metric="mae")

    def predict(self, X_test):
        test_predicts_cont = []
        cont_cols = get_cont_cols()
        X_cont = X_test[cont_cols]
        cat_cols = list(set(X_test.columns) - set(cont_cols))
        X_cat = X_test[cat_cols]
        for rgr in self.cont_predictors.values():
            pr = rgr.predict(X_cont)
            test_predicts_cont.append(pr)

        test_predictions_cont = pd.DataFrame(np.vstack(test_predicts_cont).transpose(),
                                        index=X_test.index,
                                        columns=self.cont_predictors.keys())

        test_predicts_cat = []
        for rgr in self.cat_predictors.values():
            pr = rgr.predict(X_cat)
            test_predicts_cat.append(pr)
        test_predictions_cat = pd.DataFrame(
            np.vstack(test_predicts_cat).transpose(),
            index=X_test.index,
            columns=self.cat_predictors.keys())

        X_meta = pd.concat([test_predictions_cont, test_predictions_cat], axis=1)

        return self.stack_predictor.predict(X_meta)


