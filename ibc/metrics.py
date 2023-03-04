from torchmetrics.classification import (
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryConfusionMatrix,
    BinaryAveragePrecision
)
# # to be removed
# from torchmetrics.utilities.data import dim_zero_cat
# from torchmetrics.functional.classification.auroc import _binary_auroc_arg_validation, _binary_auroc_compute

# class BinaryAUROC(BinaryPrecisionRecallCurve):

#     is_differentiable: bool = False
#     higher_is_better: Optional[bool] = None
#     full_state_update: bool = False

#     def __init__(
#         self,
#         max_fpr: Optional[float] = None,
#         thresholds: Optional[Union[int, List[float], Tensor]] = None,
#         ignore_index: Optional[int] = None,
#         validate_args: bool = True,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(thresholds=thresholds,
#                          ignore_index=ignore_index, validate_args=False, **kwargs)
#         if validate_args:
#             _binary_auroc_arg_validation(max_fpr, thresholds, ignore_index)
#         self.max_fpr = max_fpr

#     def compute(self) -> Tensor:
#         if self.thresholds is None:
#             state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
#         else:
#             state = self.confmat
#         return _binary_auroc_compute(state, self.thresholds, self.max_fpr)


class Metrics:
    def __init__(self, device):
        self.accuracy = BinaryAccuracy().to(device)
        self.conf_matrix = BinaryConfusionMatrix().to(device)
        self.recall = BinaryRecall().to(device)
        self.precision = BinaryPrecision().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.auroc = BinaryAUROC().to(device)
        self.ap = BinaryAveragePrecision().to(device)
        self.conf_matrix = BinaryConfusionMatrix().to(device)
        # TODO: add G-mean and AUC-PR

    def __call__(self, y_hat, y):
        acc = self.accuracy(y_hat, y).item()
        recall = self.recall(y_hat, y).item()
        precision = self.precision(y_hat, y).item()
        f1 = self.f1(y_hat, y).item()
        (tn, fp), (fn, tp) = self.conf_matrix(y_hat, y)
        gmean = ((tp/(tp+fn) * tn/(tn+fp))**0.5).item()
        auc = self.auroc(y_hat, y).item()
        ap = self.ap(y_hat, y).item()
        return acc, recall, precision, f1, gmean, auc, ap

    # def compute(self, y_hat, y):
    #     auc = self.auroc(y_hat, y)
    #     f1 = self.f1(y_hat, y)
    #     recall = self.recall(y_hat, y)
    #     acc = self.accuracy(y_hat, y)
    #     return auc, f1, recall, acc
