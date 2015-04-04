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
    :param col_name: excel cell name
    :param values: info list
    """
    time.sleep(1)   # waiting to close prev events
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    path = os.path.join(os.getcwd(), r"missed_data.xlsx")
    wb = excel.Workbooks.Open(path)
    ws = wb.Worksheets("missed")
    ws.Range(col_name + ":" + col_name).Delete()
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
    print("verify_excel_file: \tOkay. Ready for parsing xls.")


def parse_xls(only_interest):
    """
    :return: a basket of collections of file names for each gesture class
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

    while ws.Range(cell_pointer).Value is not None:
        cell_val = str(ws.Range(cell_pointer).Value)
        valid_label = find_valid_label(cell_val)

        firsname_val = str(int(ws.Range(firsname_col + str(row)).Value))
        secondname_val = str(ws.Range(secondname_col + str(row)).Value)
        secondname_val = secondname_val.replace("s", "-")
        secondname_val = secondname_val.replace("e", "-")
        joined_fname = firsname_val + secondname_val
        author = str(ws.Range(author_col + str(row)))

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
    wb.Close()

    return emotions_basket, authors_basket


if __name__ == "__main__":
    # verify_excel_file()
    emotions_basket, authors_basket = parse_xls(False)
    pprint(emotions_basket)
