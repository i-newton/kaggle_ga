from collections import Iterable
import pandas as pd


class DataCleaner:
    def __init__(self, common_pipelines=None, column_pipelines=None,
                 target_hooks=None, train_hooks=None, test_hooks=None):
        if common_pipelines and not isinstance(common_pipelines, Iterable):
            common_pipelines = [common_pipelines]
        if column_pipelines and not isinstance(column_pipelines, Iterable):
            column_pipelines = [column_pipelines]
        if target_hooks and not isinstance(target_hooks, Iterable):
            target_hooks = [target_hooks]
        if train_hooks and not isinstance(train_hooks, Iterable):
            train_hooks = [train_hooks]
        if test_hooks and not isinstance(test_hooks, Iterable):
            test_hooks = [test_hooks]

        self.common_pipeline = common_pipelines or []
        self.column_pipeline = column_pipelines or []
        self.target_pipeline = target_hooks or []
        self.train_hooks = train_hooks or []
        self.test_hooks = test_hooks or []

    @staticmethod
    def launch_single_tf(df, tfs):
        for tf in tfs:
            df = tf(df)
        return df

    @staticmethod
    def launch_multi_tf(train, test, tfs):
        for tf in tfs:
            train, test = tf(train, test)
        return train, test

    @staticmethod
    def log_shape(text, train, test):
        print(text)
        print(train.shape)
        print(test.shape)

    @staticmethod
    def log_nan(text, train, test):
        print(text)
        print(train.isnull().values.any() or test.isnull().values.any())

    def get_clean_data(self, train, test, group_col=None):
        # starting with hooks
        train_after_hook = self.launch_single_tf(train, self.train_hooks)
        test_after_hook = self.launch_single_tf(test, self.test_hooks)
        self.log_shape("after hooks", train_after_hook, test_after_hook)
        # extract target
        train_wo_target, target = self.launch_single_tf(train_after_hook, self.target_pipeline)
        # launch common pipelines
        train_common, test_common = self.launch_multi_tf(train_wo_target, test_after_hook, self.common_pipeline)

        self.log_shape("After common_pipeline", train_common, test_common)
        train_common_ppl = []
        test_common_ppl = []
        # launch column_pipelines
        for ppl in self.column_pipeline:
            traincppl, testcppl = ppl(train_common, test_common)
            train_common_ppl.append(traincppl)
            test_common_ppl.append(testcppl)

        train_after_ppl = pd.concat(train_common_ppl, axis=1)
        test_after_ppl = pd.concat(test_common_ppl, axis=1)
        self.log_nan("nans in column pipelines", train_after_ppl, test_after_ppl)
        self.log_shape("shape_after_column_pipelines", train_after_ppl,
                       test_after_ppl)
        return train_after_ppl, test_after_ppl
