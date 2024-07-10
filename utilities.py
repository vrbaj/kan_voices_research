from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.base import BaseSampler


class CustomSMOTE(BaseSampler):
    """
    Try 10 times KmeansSMOTE, if failed everytime, SMOTE is applied to data.
    Sometimes, KmeansSMOTE fails due to badly chosen clusters.
    """
    _sampling_type = "over-sampling"

    def __init__(self, kmeans_args=None, smote_args=None):
        super().__init__()
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)

    def _fit_resample(self, X, y):
        resample_try = 0
        while resample_try < 10:
            try:
                X_res, y_res = self.kmeans_smote.fit_resample(X, y)
                return X_res, y_res
            except Exception as e:
                resample_try += 1
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res
