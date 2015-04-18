# coding=utf-8

import win32com.client as win32
import os
from pprint import pprint
import time


def init_unique_emotion_classes():
    """
    :return: labels of unique emotion classes
    """
    classes = {
        u"улыбка",
        u"закрыл глаза",
        u"пренебрежение",
        u"отвращение",
        u"ярость",
        u"боль",
        u"ужас",
        u"затаить злобу",
        u"озадаченность",
        u"удивление",
        u"надутые губы",
        u"плакса",
        u"так себе"
    }
    return classes


def get_authors():
    authors = {
        u"volodymyr",
        u"oleksandr",
        u"alexandr"
    }
    return authors


def init_container(set_of_labels):
    """
    :param set_of_labels: set of emotions/authors
    :return: an empty dic to gather example file names
    """
    container = {}
    for label in set_of_labels:
        container[label] = []
    return container


def find_valid_label(cell_val):
    """
    :param cell_val: cell value, read from xlsx file
    :return: its unique class label
    """
    class_labels = init_unique_emotion_classes()
    valid_label = None
    for label in class_labels:
        if label in cell_val:
            valid_label = label
    return valid_label


def upd_column(col_name, values):
    """
     Update cell_name column in missed_data.xlsx
    :param col_name: excel column name
    :param values: info list
    """
    time.sleep(1)   # waiting to close prev events
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    path = os.path.join(os.getcwd(), r"missed_data.xlsx")
    wb = excel.Workbooks.Open(path)
    ws = wb.Worksheets("missed")
    ws.Range(col_name + ":" + col_name).ClearContents()
    for i, info in enumerate(values):
        pointer = col_name + str(i+2)
        ws.Range(pointer).Value = str(info)
    wb.Save()
    wb.Close()


def verify_excel_file():
    """
     Checks for not overlapping cell values in xlsx file.
    """
    # excel = win32.GetActiveObject('Excel.Application')
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    wb = excel.Workbooks.Open(r"D:\GesturesDataset\Emotion\description.xls")
    ws = wb.Worksheets(u"границы сегментов")
    col = "H"
    row = 3
    cell_pointer = col + str(row)
    valid_labels = init_unique_emotion_classes()

    while ws.Range(cell_pointer).Value is not None:
        this_val = ws.Range(cell_pointer).Value
        unique = 0
        for label in valid_labels:
            if label in this_val:
                unique += 1
        assert unique <= 1, "xlsx file has overlapping cell values"
        row += 1
        cell_pointer = col + str(row)
    wb.Close()
    print("verify_excel_file: \tOKAY. Ready to parse xls.")


def read_my_comments(cell_val):
    """
     Prints out lower and upper bound for some csv/blend files to be fit in.
     :param cell_val: K's column cell value
    """
    if cell_val is not None:
        a_comment = str(cell_val)
        if "begin from " in a_comment:
            a_comment = a_comment.strip("begin from ")
            begin = int(a_comment.split(' ')[0])
        else:
            begin = 0
        if "stop on " in a_comment:
            a_comment = a_comment.split("stop on ")[-1]
            end = int(a_comment)
        else:
            end = "--"
        print("comment: %s; begin: %s, end: %s" % (str(cell_val), begin, end))


def parse_xls(only_interest):
    """
    :returns:
        (1) a collection of file names for each emotion class
        (2) a collection of authors for each emotion class
        (3) a collection of (begin, end) for some emotion file names
    """
    verify_excel_file()

    excel = win32.gencache.EnsureDispatch('Excel.Application')
    wb = excel.Workbooks.Open(r"D:\GesturesDataset\Emotion\description.xls")
    ws = wb.Worksheets(u"границы сегментов")
    my_labels_col = "H"
    firsname_col = "I"
    secondname_col = "J"
    author_col = "X"
    row = 3
    cell_pointer = my_labels_col + str(row)

    emotions_basket = init_container(init_unique_emotion_classes())
    authors_basket = init_container(get_authors())
    boundaries_basket = {}

    while ws.Range(cell_pointer).Value is not None:
        cell_val = str(ws.Range(cell_pointer).Value)
        valid_label = find_valid_label(cell_val)

        firsname_val = str(int(ws.Range(firsname_col + str(row)).Value))
        secondname_val = str(ws.Range(secondname_col + str(row)).Value)
        secondname_val = secondname_val.replace("s", "-")
        secondname_val = secondname_val.replace("e", "-")
        joined_fname = firsname_val + secondname_val
        author = str(ws.Range(author_col + str(row)))
        # read_my_comments(ws.Range("K" + str(row)).Value)

        row += 1
        cell_pointer = my_labels_col + str(row)

        if only_interest and valid_label is None:
            # we do not account for labels we are not interested in
            continue
        elif valid_label not in emotions_basket:
            # it is not a valid label anymore, actually
            emotions_basket[valid_label] = []

        emotions_basket[valid_label].append(joined_fname)
        authors_basket[author].append(joined_fname)
        boundaries_basket[joined_fname] = ws.Range("K" + str(row)).Value
    wb.Close()

    return emotions_basket, authors_basket, boundaries_basket


def how_many_examples_we_have():
    """
     Prints out how many emotion examples we have per one class.
    """
    emotions_basket, _, _ = parse_xls(False)
    for key in emotions_basket:
        print(key, len(emotions_basket[key]))


if __name__ == "__main__":
    # verify_excel_file()
    emotions_basket, authors_basket, bound = parse_xls(False)
    pprint(emotions_basket)
    # how_many_examples_we_have()
