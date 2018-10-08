import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold


def show_uniq_test_train(train, test):
    # check all values that have zero ans nan only
    for c in sorted(train.columns):
        un = train[c].unique()
        if len(un)<100:
            tun = test[c].unique()
            print("%s ;train: %s; test:%s"%(c, un, tun))


def drop_duplicates_by_key(df, key):
    keys = df[key]
    non_dupe_ind = keys.drop_duplicates().index
    dupe_num = len(df) - len(non_dupe_ind)
    print("%s duplicates" % dupe_num)
    return df.loc[non_dupe_ind]


def get_train():
    train_main = pd.read_csv("../data/task1/train_1.8.csv", encoding="cp1251")
    train_main.drop_duplicates()
    print("Main shape %s" % str(train_main.shape))

    train_aux_frac = pd.read_csv("../data/task1_additional/frac_train_1.csv", encoding="cp1251")
    train_aux_frac = drop_duplicates_by_key(train_aux_frac, "Скважина")
    train_frac_main = pd.merge(train_main, train_aux_frac, how="left",
                               left_on="Скважина", right_on="Скважина")

    train_aux_gdis = pd.read_csv("../data/task1_additional/gdis_train1.2.csv", encoding="cp1251")
    train_aux_gdis = drop_duplicates_by_key(train_aux_gdis, "Скважина")
    train_main_frac_gdis = pd.merge(train_frac_main, train_aux_gdis, how="left",
                                    left_on="Скважина",
                                    right_on="Скважина")

    train_aux_coords = pd.read_csv(
        "../data/task1_additional/coords_train_1.1.csv", encoding="cp1251")
    train_aux_coords = drop_duplicates_by_key(train_aux_coords, "well_hash")
    all_recs = pd.merge(train_main_frac_gdis, train_aux_coords, how="left",
                        left_on="Скважина", right_on="well_hash")
    final_recs = all_recs.drop(["well_hash"], axis=1)
    print("final shape %s" % str(final_recs.shape))
    return final_recs.drop_duplicates()


def get_test():
    test_main = pd.read_csv("../data/task1/test_1.9.csv", encoding="cp1251")
    print("Main shape %s" % str(test_main.shape))
    test_aux_frac = pd.read_csv("../data/task1_additional/frac_test_1.csv", encoding="cp1251")
    test_aux_frac = drop_duplicates_by_key(test_aux_frac, "Скважина")

    test_frac_main = pd.merge(test_main, test_aux_frac, how="left",
                              left_on="Скважина", right_on="Скважина")

    test_aux_gdis = pd.read_csv("../data/task1_additional/gdis_test1.2.csv", encoding="cp1251")
    test_aux_gdis = drop_duplicates_by_key(test_aux_gdis, "Скважина")
    test_main_frac_gdis = pd.merge(test_frac_main, test_aux_gdis, how="left", left_on="Скважина", right_on="Скважина")

    test_aux_coords = pd.read_csv(
        "../data/task1_additional/coords_test_1.1.csv", encoding="cp1251")

    test_aux_coords = drop_duplicates_by_key(test_aux_coords, "well_hash")
    all_recs = pd.merge(test_main_frac_gdis, test_aux_coords, how="left", left_on="Скважина", right_on="well_hash")
    final_recs = all_recs.drop(["well_hash"], axis=1)
    print("final shape %s" % str(final_recs.shape))
    return final_recs


def get_existed(columns, df):
    return list(set(columns)&set(df.columns))


def split_continious_date_categorical_text(df):
    group_id = ["Скважина"]

    exclude_cont = []
    """'Ток номинальный', 'Приемистость, м3/сут',
       'Глубина верхних дыр перфорации', 'Пластовое давление начальное', 'Низ',
       'I X/X', 'Обводненность (вес), %', 'ГП - Забойное давление',
       'ТП - Забойное давление', 'FCD', 'Простой, ч', 'М пр', 'JD',
       'Буферное давление', 'Мощность ПЭД', 'Обводненность',
       'Пластовое давление', 'М бр', 'Глубина спуска',
       'Производительность ЭЦН', 'JD факт', 'Рпл Хорнер', 'К пр от стимуляции',
       'Xf', 'Закачка, м3', 'Давление в линии', 'Диаметр НКТ',
       'ГП(ГРП) Дебит жидкости', 'ГП(ГРП) Дебит жидкости скорр-ый', 'Эфф',
       'Напор', 'Верх', 'Азимут', 'Диаметр экспл.колонны', 'Ток рабочий',
       'Затрубное давление', 'Hf', 'Wf',
                   "Дата ввода в эксплуатацию",
                   "Дата запуска после КРС" ,
                   "Диаметр плунжера",
                   "Природный газ, м3",
                   "Конденсат, т",
                   "Длина хода плунжера ШГН",
                   "Коэффициент подачи насоса",
                   "Дебит конденсата",
                   "Вязкость воды в пластовых условиях",
                   "Газ из газовой шапки, м3",
                   "Число качаний ШГН",
                   "Группа фонда",
                   "Фонтан через насос",
                   "Неустановившийся режим",
                   "Закачка, м3",
                   "ГП(ИДН) Прирост дефита нефти",
                   "Вязкость нефти в пластовых условия",
                   "Закачка, м3",
                   "ГП(ИДН) Дебит жидкости скорр-ый",
                   ]
                   """

    continious = list(set(df.columns) - set(dates) - set(categorical)
                      - set(text) - set(group_id) - set(coords) - set(exclude_cont))
    return (df[group_id], df[continious], df[get_existed(dates, df)], df[get_existed(categorical, df)],
            df[get_existed(text, df)], df[get_existed(coords, df)])


def get_fold():
    return KFold(n_splits = 5,shuffle=True, random_state = 17)


def clean_non_targeted(train_array, y_train, dates_ord=None):
    clean_array = []
    train_array.append(y_train)
    #clear nans in target
    indexes_to_delete = y_train[(y_train.isnull())|(y_train == 0)].index
    if dates_ord is not None:
        dates_index = dates_ord[dates_ord > 6].index
        indexes_to_delete = indexes_to_delete.union(dates_index)
    for df in train_array:
        item = df.drop(index=indexes_to_delete)
        clean_array.append(item)
        print(item.shape)
    return clean_array


def check_passed(to_test, df):
    for c in df.columns:
        if df[c].corr(to_test) > 0.8:
            return False
    return True


def square_cont(train, test, y):
    columns = []
    train_squared = []
    print("started squaring")
    for c1 in train.columns:
        for c2 in train.columns:
            name = str(c1) + "1_" + str(c2) + "2"
            sq_item = train[c1].multiply(train[c2])
            sq_item.rename(name, inplace=True)
            corr = sq_item.corr(y)
            if (corr > 0.1 or corr < -0.1) and check_passed(sq_item, train):
                columns.append((c1, c2))
                train_squared.append(sq_item)
    print("finish squaring")
    test_squared = []
    for c1, c2 in columns:
        name = str(c1) + "1_" + str(c2) + "2"
        sq_item = test[c1].multiply(test[c2])
        sq_item.rename(name, inplace=True)
        test_squared.append(sq_item)
    return pd.concat(train_squared, axis=1), pd.concat(test_squared, axis=1)


def  sqrt(x):
    if np.all(x>0):
        return np.sqrt(x)
    return 0
def reverse(x):
    if np.all(x!=0):
        return 1/x
    return 0

def log(x):
    if np.all(x>0):
        return np.log(x)
    return 0

transformations = {"log":log,
                   "exp":np.exp,
                   "sqrt":sqrt,
                   "sq":lambda x: x**2,
                   "cube":lambda x:x**3,
                   "reverse":reverse,
                   "orig":lambda x:x}


def get_max_correlation(x,y):
    max_corr = 0
    max_corr_fn = "orig"
    for n,tf in transformations.items():
        x_tf = x.apply(tf)
        corr = abs(y.corr(x_tf))
        if corr>max_corr:
            max_corr = corr
            max_corr_fn  = n
    return max_corr_fn


def transform_with_max_corr(train, test, y):
    for c in train.columns[:150]:
        fn = get_max_correlation(train[c], y)
        if fn != "orig":
            train.loc[:,c] = train[c].apply(transformations[fn])
            test.loc[:,c] = test[c].apply(transformations[fn])
    return train, test


from sklearn.metrics import make_scorer

def my_loss(y_true, y_pred,**kwargs):
    loss = np.abs(np.exp(y_true) - np.exp(y_pred))
    return np.average(loss)


my_score = make_scorer(my_loss, greater_is_better=False)


def get_prediction(X_train, y_train, X_groups, X_test, constant = 701.4750):
    X_step, y_step = getXY_for_step(X_train, y_train, X_groups)
    preds = get_preds(X_step, y_step, X_test, X_train, y_train,groups)
    return preds+(constant- np.mean(preds))