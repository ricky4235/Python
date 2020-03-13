# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:06:43 2019

@author: 11004076
"""

import pymysql
# xlrd 為 python 中讀取 excel 的庫，支援.xls 和 .xlsx 檔案
# import xlrd
 
# openpyxl 庫支援 .xlsx 檔案的讀寫
from openpyxl.reader.excel import load_workbook
from builtins import int

#cur 是資料庫的遊標連結，path 是 excel 檔案的路徑
def importExcelToMysql(cur, path):
 
    ### xlrd版本
    # 讀取excel檔案
    # workbook = xlrd.open_workbook(path)
    # sheets = workbook.sheet_names()
    # worksheet = workbook.sheet_by_name(sheets[0])
    ###

    ### openpyxl版本
    # 讀取excel檔案
    workbook = load_workbook(path)
    # 獲得所有工作表的名字
    sheets = workbook.get_sheet_names()
    # 獲得第一張表
    worksheet = workbook.get_sheet_by_name(sheets[0])
    ###
 
    ### xlrd版本
    # 將表中資料讀到 sqlstr 陣列中
    # for i in range(1, worksheet.nrows):
    #     row = worksheet.row(i)
    #
    #     sqlstr = []
    #
    #     for j in range(0, worksheet.ncols):
    #         sqlstr.append(worksheet.cell_value(i, j))
    ###
 
    ### openpyxl版本
    # 將表中每一行資料讀到 sqlstr 陣列中
    for row in worksheet.rows:
 
        sqlstr = []
 
        for cell in row:
            sqlstr.append(cell.value)
###
 
        valuestr = [str(sqlstr[0]), int(sqlstr[1]), int(sqlstr[2]), int(sqlstr[3]), int(sqlstr[4])]
 
        # 將每行資料存到資料庫中(sql語法)
        cur.execute("insert into student(姓名, 語文, 數學, 英語, 物理) values(%s, %s, %s, %s, %s)", valuestr)
 
# 輸出資料庫中內容
def readTable(cursor):
    # 選擇全部
    cursor.execute("select * from student")
    # 獲得返回值，返回多條記錄，若沒有結果則返回()
    results = cursor.fetchall()
 
    for i in range(0, results.__len__()):
        for j in range(0, 5):
            print(results[i][j], end='\t')
 
        print('\n')
 
if __name__ == '__main__':
    # 和資料庫建立連線
    conn = pymysql.connect('localhost', 'root', 'admin', charset='utf8')
    # 建立遊標連結
    cur = conn.cursor()
 
    # 新建一個database
    cur.execute("drop database if exists students")
    cur.execute("create database students")
    # 選擇 students 這個資料庫
    cur.execute("use students")
 
    # sql中的內容為建立一個名為student的表
    sql = """CREATE TABLE IF NOT EXISTS `student` (
                `姓名` VARCHAR (20),
                `語文` INT,
                `數學` INT,
                `英語` INT,
                `物理` INT
              )"""
    # 如果存在student這個表則刪除
    cur.execute("drop table if exists student")
    # 建立表
    cur.execute(sql)
 
    # 將 excel 中的資料匯入 資料庫中
    importExcelToMysql(cur, "C:/Users/11004076/Desktop/student.xlsx")
    readTable(cur)
 
    # 關閉遊標連結
    cur.close()
    conn.commit()
    # 關閉資料庫伺服器連線，釋放記憶體
    conn.close()