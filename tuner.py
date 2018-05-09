import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from hyperopt import fmin, tpe, hp, Trials, space_eval


class Tuner:

    def __init__(
        self,
        train, ytrain,
        model, maximize,
        base_params, tunable_params, integer_params=[],
        n_splits=5, stratify=False, split_seed=0,
        metric=None, return_labels=False
    ):
        """
        Tuner searches for best hyperparameters on train data using
        hyperopt module and cross-validation. Can be used for both
        classification and regression problems.
        Main method is self.tune.

        Tuning is done in epochs, on each epoch CV folds are fixed.
        Let's say we have list of trials number per epoch: [100, 40, 15, 5].
        Then on first epoch hyperopt is used to determine 100 possible sets
        of parameters. We select top 40 sets and run them on new CV folds.
        Then we average perfomance of these parameter sets on two epochs
        and choose top 15 out of 40 and move with them into epoch 3.
        Process is repeated until there is no epochs remained. Hyperopt is
        only used on the first epoch.

        Input data:
        train, ytrain (pd.DataFrame/numpy.array/scipy.csr_matrix) - train data
        model ('lgb' or 'xgb' or sklearn non-initialized model) - model
            to tune.
        maximize (bool) - whether ot maximize the metric. If boosting model is
            being tuned - applied to metric in base_params. If sklearn model -
            to self.metric.

        base_params (dict) - base non-tunable parameters for model.
        tunable_params (dict of hyperopt.hp objects) - parameters to tune.
        integer_params (list of strings) - parameters' names in tunable_params,
            that must be transformed into integers so model won't break.

        n_splits (int) - number of splits for CV.
        stratify (bool) - whether use StratifiedKFold of common KFold
        split_seed (int) - seed for CV
        metric - function, that accepts (true_values, predicted_values)
            and returns float. Only needed for sklearn models.
        return_labels (bool) - if False - return probabilities, otherwise -
            predicted labels. Is only used for classification and sklearn.
        """
        self.train = train
        self.ytrain = ytrain

        self.model = model
        self.maximize = maximize

        self.base_params = base_params
        self.tunable_params = tunable_params
        self.integer_params = integer_params

        self.n_splits = n_splits
        self.split_seed = split_seed
        self.stratify = stratify

        self.metric = metric
        self.return_labels = return_labels

        self.results = pd.DataFrame()
        self.trials = Trials()
        self.loss_history = []
        self.num_iters = []
        self.t_start = None
        self.folds = None
        self.base_scores = {}

        if model == 'xgb':
            self.dtrain = xgb.DMatrix(train, ytrain)
        elif model == 'lgb':
            self.dtrain = lgb.Dataset(train, ytrain)

    def tune(self, trials_to_to):
        """
        Main method
        trials_to_to (list of ints) - number of trials per epoch.
        """
        self.tune_first_epoch(trials_to_to[0])
        for i, max_trials in enumerate(trials_to_to[1:]):
            epoch = i + 2
            self.tune_not_first_epoch(epoch, max_trials)

    def tune_first_epoch(self, max_trials):
        print('Started 1 epoch of tuning')
        self.set_folds(epoch=1)
        self.get_base_score(epoch=1)
        self.reset_histories()

        # Train hyperopt.
        self.trials = Trials()
        search_space = {**self.tunable_params, **self.base_params}
        fmin(
            fn=self.get_score,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=self.trials,
            return_argmin=False)
        self.fill_in_results_df()
        print('\n')

    def set_folds(self, epoch):
        if self.stratify:
            self.folds = list(StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.split_seed+epoch
                ).split(self.train, self.ytrain))
        else:
            self.folds = list(KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.split_seed+epoch
                ).split(self.train))

    def get_base_score(self, epoch=0):
        self.reset_histories()
        base_score = self.get_score(self.base_params)
        if self.maximize:
            base_score *= -1
        print('Default parameters score for current folds:',
              round(base_score, 4))
        self.base_scores[epoch] = base_score

    def reset_histories(self):
        self.loss_history = []
        self.num_iters = []
        self.t_start = time.time()

    def fill_in_results_df(self):

        def process_dict(d):
            # Parses trials object correctly.
            for k in d.keys():
                d[k] = d[k][0]
            d = space_eval(self.tunable_params, d)
            for p in self.integer_params:
                if p in d:
                    d[p] = int(d[p])
            return d

        sorted_trials = sorted(
            self.trials.trials,
            key=lambda t: t['result']['loss']).copy()
        sorted_indexes = sorted(
            range(len(self.trials.trials)),
            key=lambda t: self.trials.trials[t]['result']['loss']
        )
        self.results['params'] = [
            {**process_dict(t['misc']['vals']),
             **self.base_params} for t in sorted_trials]
        self.results['num_iters'] = np.array(self.num_iters)[sorted_indexes]
        self.results['checks'] = 1
        self.results['total_loss'] = np.array(
            self.loss_history)[sorted_indexes]
        self.results['1_epoch_loss'] = np.array(
            self.loss_history)[sorted_indexes]

    def tune_not_first_epoch(self, epoch, max_trials):
        print('Started {} epoch of tuning'.format(epoch))
        self.set_folds(epoch)
        self.get_base_score(epoch)
        self.reset_histories()
        colname = str(epoch)+'_epoch_loss'
        self.results[colname] = 0.

        for params in self.results.params.iloc[:max_trials]:
            self.get_score(params)

        # Update and resort results.
        self.results.loc[:max_trials-1, 'num_iters'] = (
            (self.results.loc[:max_trials-1, 'num_iters']
             * self.results.loc[:max_trials-1, 'checks']
             + self.num_iters
             ) / (self.results.loc[:max_trials-1, 'checks'] + 1)
            ).astype(int)

        self.results.loc[:max_trials-1, colname] = self.loss_history
        self.results.loc[:max_trials-1, 'checks'] += 1
        epoch_losses = [
            col for col in self.results.columns if '_epoch_loss' in col]
        self.results['total_loss'] = self.results[epoch_losses].sum(axis=1)

        # In top will be values with max checks and min loss.
        self.results = self.results.sort_values(
            ['checks', 'total_loss'], ascending=[False, not self.maximize]
            ).reset_index(drop=True)
        print('\n')

    def get_score(self, params):

        for p in self.integer_params:
            if p in params:
                params[p] = int(params[p])

        if self.model == 'xgb':
            cvmodel = xgb.cv(
                params.copy(),
                self.dtrain,
                num_boost_round=10000,
                folds=self.folds,
                early_stopping_rounds=10,
                verbose_eval=False)
            score = cvmodel.iloc[-1, 0]
            best_iter = len(cvmodel)
        elif self.model == 'lgb':
            cvmodel = lgb.cv(
                params.copy(),
                self.dtrain,
                num_boost_round=10000,
                folds=self.folds,
                early_stopping_rounds=10,
                verbose_eval=False)
            k = [k for k in cvmodel.keys() if '-mean' in k][0]
            score = cvmodel[k][-1]
            best_iter = len(cvmodel[k])
        elif isinstance(self.model(), BaseEstimator):
            model = self.model(**params)
            if isinstance(model, RegressorMixin) or self.return_labels:
                method = 'predict'
            else:
                method = 'predict_proba'
            oofs = cross_val_predict(
                model, self.train, self.ytrain,
                cv=self.folds, method=method)
            score = self.metric(self.ytrain, oofs)
            best_iter = 0
        else:
            raise NotImplementedError

        self.loss_history.append(score)
        self.num_iters.append(best_iter)

        if len(self.loss_history) > 1:
            print('{} trials are done in {} minutes. Best metric achieved is {}'.
                  format(
                    len(self.loss_history),
                    round((time.time() - self.t_start) / 60, 1),
                    round(sorted(
                        self.loss_history, reverse=self.maximize)[0], 4)
                    ), end='\r')
        if self.maximize:
            score *= -1
        return score

    def plot_1_epoch_results(self):
        import matplotlib.pylab as plt
        import seaborn as sns
        plt.style.use('ggplot')
        plt.style.use('seaborn-poster')
        sns.set_palette('Set1', 10, desat=0.75)

        plt.figure(figsize=(10, 5))
        xs = [t['misc']['tid'] for t in self.trials.trials]
        ys = [t['result']['loss'] for t in self.trials.trials]
        if self.maximize:
            ys = [-y for y in ys]

        baseline = self.base_scores[1]
        delta = baseline - np.min(ys)
        if delta > 0:
            y_lower = np.min(ys) - 0.1*delta
        else:
            y_lower = baseline - 0.1*abs(delta)
        y_higher = baseline + 5*abs(delta)

        plt.scatter(xs, ys, s=30, label='hyperopt trials')
        plt.axhline(
            y=self.base_scores[1],
            linestyle='--',
            label='default parameters')
        plt.legend(frameon=True, framealpha=1)
        plt.title('Trials loss progression')
        plt.xlabel('Trial number')
        plt.ylabel('CV loss')
        plt.ylim(y_lower, y_higher)
        plt.show()
