from collections import Iterable
import pandas as pd

from sklearn.preprocessing import StandardScaler


class PipelineStep:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, train, test, *args, **kwargs):
        if isinstance(self.tf, Iterable):
            for step in self.tf:
                train, test = step(train, test)
            return train, test
        return self.tf(train, test)


class Hook:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, item, *args, **kwargs):
        if isinstance(self.tf, Iterable):
            for step in self.tf:
                item = step(item)
            return item
        return self.tf(item)


class ColumnPipelineStep(PipelineStep):

    def __init__(self, columns, tf):
        super(ColumnPipelineStep, self).__init__(tf)
        self.cols = columns

    def __call__(self, train, test, *args, **kwargs):
        train = train[self.cols]
        test = test[self.cols]
        return super(ColumnPipelineStep, self).__call__(train, test)


# sort by converted date and group
def convert_and_sort(df):
    df["Дата"] = df["Дата"].apply(pd.to_datetime)
    return df.sort_values(by=["Скважина", "Дата"])


train_hook = Hook(tf=convert_and_sort)


def get_non_useful(df):
    non_useful_columns = []
    for c in df.columns:
        null_columns = df[df[c].isnull()]
        if len(null_columns) == len(df):
            non_useful_columns.append(c)
    return non_useful_columns


def drop_non_useful(train, test):
    non_useful = set(get_non_useful(train)) | set(get_non_useful(test))
    print("%s dropped"% non_useful)
    return train.drop(list(non_useful), axis=1), test.drop(list(non_useful), axis=1)


drop_constants = PipelineStep(tf=drop_non_useful)


# drop non present columns in test
def drop_not_present(train, test):
    absent_columns = list(set(train.columns) - set(test.columns))
    print("%s dropped" % absent_columns)
    return train.drop(absent_columns, axis=1), test


drop_not_present_test_train = PipelineStep(tf=drop_not_present)

common_pipeline = PipelineStep(tf=[drop_constants, drop_not_present_test_train])


def get_float(v):
    v = str(v)
    if v != "NaN":
        new = v.replace(",",".")
        return float(new)
    return v


def get_target(df, column="Нефть, т"):
    target = df[column]
    print("%s dropped"% column)
    return df.drop([column], axis=1), target.apply(get_float)


target_hook = Hook(tf=get_target)


def get_text_cols():
    text = ["Причина простоя",
            "Причина простоя.1",]
    return text


def get_cat_cols():
    categorical = ["Тип испытания",
                   "Тип скважины",
                   "Неустановившийся режим",
                   "ГТМ",
                   "Метод",
                   "Характер работы",
                   "Состояние",
                   "Пласт МЭР",
                   "Способ эксплуатации",
                   #"Тип насоса",
                   "Состояние на конец месяца",
                   "Номер бригады",
                   "Фонтан через насос",
                   "Нерентабельная",
                   "Назначение по проекту",
                   "Группа фонда",
                   "Тип дополнительного оборудования",
                   #"Марка ПЭД",
                   "Тип ГЗУ",
                   "ДНС",
                   "КНС",
                   # "Агент закачки",
                   # text converted
                   "Мероприятия",
                   #"Проппант",
                   #"Куст",
                   'ПЛАСТ'
                   ]
    return categorical


def get_date_cols():
    dates = ["Скважина", "Дата"]
    return dates


def get_coord_cols():
    coords = ["УСТЬЕ_X", "УСТЬЕ_Y", "ПЛАСТ_X", "ПЛАСТ_Y"]
    return coords


def get_cont_cols():
    all_cols = [
        # ,
        "Дата",
        "ГТМ",
        "Метод",
        "Характер работы",
        "Состояние",
        "Время работы, ч",
        "Время накопления",
        "Попутный газ, м3",
        #"Закачка, м3",
        "Природный газ, м3",
        "Газ из газовой шапки, м3",
        "Конденсат, т",
        "Простой, ч",
        "Причина простоя",
        #"Приемистость, м3/сут",
        #"Обводненность (вес), %",
        "Дебит конденсата",
        "Добыча растворенного газа, м3",
        "Дебит попутного газа, м3/сут",
        "Пласт МЭР",
        #"Куст",
        "Тип скважины",
        "Диаметр экспл.колонны",
        #"Диаметр НКТ",
        "Диаметр штуцера",
        "Глубина верхних дыр перфорации",
        "Удлинение",
        "Способ эксплуатации",
        #"Тип насоса",
        "Производительность ЭЦН",
        #"Напор",
        "Частота",
        "Коэффициент сепарации",
        "Глубина спуска",
        #"Буферное давление",
        #"Давление в линии",
        "Пластовое давление",
        "Динамическая высота",
        #"Затрубное давление",
        "Давление на приеме",
        "Забойное давление",
        "Обводненность",
        "Состояние на конец месяца",
        "Давление наcыщения",
        "Газовый фактор",
        "Температура пласта",
        "SKIN",
        "JD факт",
        "Дата ГРП",
        "Вязкость нефти в пластовых условиях",
        "Вязкость воды в пластовых условиях",
        "Вязкость жидкости в пласт. условиях",
        "объемный коэффициент",
        "Плотность нефти",
        "Плотность воды",
        "Высота перфорации",
        "Удельный коэффициент",
        "Коэффициент продуктивности",
        #"ТП - Забойное давление",
        "ТП - JD опт.",
        "ТП - SKIN",
        "К пр от стимуляции",
        "Глубина спуска.1",
        "КВЧ",
        "Время до псевдоуст-ся режима",
        "Причина простоя.1",
        "Дата запуска после КРС",
        "Дата пуска",
        "Дата останова",
        "Радиус контура питания",
        "Мероприятия",
        "Номер бригады",
        "Фонтан через насос",
        "Нерентабельная",
        "Неустановившийся режим",
        "Дата ввода в эксплуатацию",
        "Назначение по проекту",
        "Замерное забойное давление",
        "Группа фонда",
        "Нефтенасыщенная толщина",
        "Плотность раствора глушения",
        "Глубина текущего забоя",
        "Тип дополнительного оборудования",
        "Диаметр дополнительного оборудования",
        "Глубина спуска доп. оборудования",
        #"Марка ПЭД",
        #"Мощность ПЭД",
        #"I X/X",
        #"Ток номинальный",
        #"Ток рабочий",
        "Число качаний ШГН",
        "Длина хода плунжера ШГН",
        "Диаметр плунжера",
        "Коэффициент подачи насоса",
        "Тип ГЗУ",
        "ДНС",
        "КНС",
        "КН закрепленный",
        "Пластовое давление начальное",
        "Характеристический дебит жидкости",
        "Время в работе",
        "Время в накоплении",
        #"ГП - Забойное давление",
        "ГП(ИДН) Дебит жидкости",
        "ГП(ИДН) Дебит жидкости скорр-ый",
        "ГП(ИДН) Прирост дефита нефти",
        #ƒ"ГП(ГРП) Дебит жидкости",
        #"ГП(ГРП) Дебит жидкости скорр-ый",
        "Наклон",
        "Азимут",
        "k",
        #"Ноб",
        "Нэф",
        "Pпл",
        "Верх",
        "Низ",
        "Xf",
        #"Hf",
        #"Wf",
        #"JD",
        "FCD",
        #"М пр",
        #"Проппант",
        #"Рпл Хорнер",
        #"Эфф",
        "Конц",
        "Гель",
        #"М бр",
        "V под",
        "V гель",
        "Давление пластовое",
        "Тип испытания",
        "ПЛАСТ",
        "УСТЬЕ_X",
        "УСТЬЕ_Y",
        "ПЛАСТ_X",
        "ПЛАСТ_Y",
        "Альтитуда",
        "Дата ГРП",
        "Время до псевдоуст-ся режима",
        "Дата запуска после КРС",
        "Дата пуска",
        "Дата останова",
        "Дата ввода в эксплуатацию"
    ]
    continious = list(set(all_cols) - set(get_date_cols()) - set(get_cat_cols())
                      - set(get_text_cols()) - set(get_coord_cols()))
    return continious


def get_object_columns(df):
    objects = []
    for c in df.columns:
        if df[c].dtype != pd.np.float:
            objects.append(c)
    return objects


def convert_locale_to_float(df):
    loc_float = get_object_columns(df)
    converted = df.copy()
    for c in loc_float:
        converted.loc[:, c] = df[c].apply(get_float)
    return converted


def convert_train_test(train, test):
    return convert_locale_to_float(train), convert_locale_to_float(test)


to_float_step = PipelineStep(tf=convert_train_test)


def fill_with_median(train, test):
    means = train.median()
    norm_train = train.fillna(means)
    norm_test = test.fillna(means)
    return norm_train, norm_test


median_step = PipelineStep(tf=fill_with_median)


# now we have clear non-normalized data, let's normalize first
def normalize(train, test):
    scaler = StandardScaler()
    norm_train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index = train.index)
    norm_test = pd.DataFrame(scaler.transform(test), columns=test.columns, index = test.index)
    return norm_train, norm_test


normalize_step = PipelineStep(tf=normalize)


def null_cat(train, test):
    train = train.isnull().astype(int).add_suffix('_indicator')
    test = test.isnull().astype(int).add_suffix('_indicator')

    return train, test


null_cat_step = ColumnPipelineStep(columns=get_cont_cols(),
                                   tf=null_cat)


cont_pipeline = ColumnPipelineStep(columns=get_cont_cols(),
                                   tf=[to_float_step, median_step,
                                       normalize_step])

from functools import partial

def get_one_hot(train, test, drop_first=True):
    if isinstance(train, pd.Series):
        train = train.astype(str)
        test = test.astype(str)
    else:
        for c in train.columns:
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)
    train_oh = pd.get_dummies(train, drop_first=drop_first)
    test_oh = pd.get_dummies(test, drop_first=drop_first)
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)
    return train_oh, test_oh


one_hot_no_drop = partial(get_one_hot, drop_first=False)


cat_pipeline = ColumnPipelineStep(columns=get_cat_cols(), tf=get_one_hot)


def text_trasnsform_pipeline(train, test):
    for c in train.columns:
        train[c] = train[c].str.lower()
        test[c] = test[c].str.lower()
    return train, test


text_pipeline = ColumnPipelineStep(columns=get_text_cols(),
                                   tf=[text_trasnsform_pipeline, get_one_hot])


def transform_dates_into_order(dates, group):
    grouped = pd.concat([dates, group], axis=1)
    idx = []
    orders = []
    for name, group in grouped.groupby(["Скважина"]):
        index = group.index
        for i in range(len(index)):
            idx.append(index[i])
            orders.append(i)
    ord_index = pd.Index(idx)
    ordered_fr = pd.Series(orders, index=ord_index, dtype="int32",
                           name="pos_number")
    return ordered_fr


def get_time_of_year(dates):
    def time_of_year(date):
        month = date.month
        if month >= 3 or month < 6:
            return 1
        elif month >= 6 or month < 9:
            return 2
        elif month >= 9 or month < 12:
            return 3
        else:
            return 4
    return dates.apply(time_of_year)


def date_transform(train, test):
    train_date = train["Дата"].apply(pd.to_datetime)
    train_group = train["Скважина"]
    train_cat_ord = transform_dates_into_order(train_date, train_group)
    train_cat_toy = get_time_of_year(train_date)
    train = pd.concat([train_cat_ord, train_cat_toy], axis=1)

    test_date = test["Дата"].apply(pd.to_datetime)
    test_cat_ord = pd.Series(0, index=test_date.index, dtype="int32",
                             name="pos_number")
    test_cat_toy = get_time_of_year(test_date)
    test_f = pd.concat([test_cat_ord, test_cat_toy], axis=1)
    return train, test_f


date_pipeline = ColumnPipelineStep(columns=get_date_cols(),
                                   tf=[date_transform, one_hot_no_drop])
