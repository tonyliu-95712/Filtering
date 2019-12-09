import numpy as np
import pandas as pd

import time
#文件读取
import os
#进度条
from tqdm import tqdm
#数学工具包: for多元线性回归
import statsmodels.formula.api as sm


# 导入原始数据并处理
# input是Excel的文件路径
# Return处理后的DataFrame
def input(filepath):
    dframe = pd.read_excel(filepath, sheet_name='Sheet1')
    print('已获取原始数据')    
    print('日期/股票代码规范化：')
    for i in tqdm(range(len(dframe))):
        timeStruct = time.strptime(str(dframe.iloc[i,0]), '%Y-%m-%d %H:%M:%S')
        strTime = time.strftime('%Y%m',timeStruct)
        dframe.iloc[i,0] = strTime

        code = dframe.iloc[i,1]
        if code < 600000:
            code = ''+ "{0:06d}".format(code) + '.SZ'
        else:
            code = ''+ "{0:06d}".format(code) + '.SH'
        dframe.iloc[i,1] = code

    #把数据按日期和股票代码排列
    result = dframe.set_index(['END_DATE','STOCK_CODE']) 

    #缺失数据填充（先填前一行，再填后一行）
    # （此处需修改）
    nan_result = result.fillna(method='pad') 
    nan_result = nan_result.fillna(method='backfill')


    print('正在读取收益率csv：')
    file_name = list()
    quote_csv = list()
    for file in tqdm(os.listdir('./quote')):
        csv = pd.read_csv('./quote/'+file)
        file_name.append(file)
        quote_csv.append(csv)

 
    print('匹配收益率数据：')
    stock_return = list()
    dropped = 0  
    for i in tqdm(range(len(nan_result))):
        stockCode = nan_result.index[i-dropped][1]
        date = nan_result.index[i-dropped][0]

        if (int(date) >= 200901) and (int(date) <= 201907):
            csv_path = 'quote_'+ date + '.csv'
            ret_dat = quote_csv[file_name.index(csv_path)]
            stock_ret_dat = ret_dat.loc[ret_dat['Code'] == stockCode]

            cal_ret = 1

            for i in range(len(stock_ret_dat)):
                cal_ret = cal_ret * (1 + stock_ret_dat.iloc[i,4])
            
            stock_return.append(cal_ret)

        else:
            nan_result.drop(index[i-dropped])
            dropped = dropped+1


    nan_result.insert(0, 'RETURN', stock_return)
    return nan_result



# 多元线性回归-残差
# Input     单个因子的值向量和因变量向量y
# Return    该因子的残差
def LinearReg(NewBackup, y):

    myFormula = y.columns[0] + '~ ' + NewBackup.columns[0]

    BackupResid = list()

    myList = [y,NewBackup]
    myFactor = pd.concat(myList,axis=1)

    #多元线性回归（公式，数据集）
    Model=sm.ols(formula = myFormula, data=myFactor).fit()
    #模型概述
    #print(Model.summary())

    #残差
    myResid = Model.resid

    return myResid

# FamaMacBeth
# input     残差向量，因变量向量y，收益向量
# return    该残差向量的显著性和R2
def FamaMacBeth(myResid, y, myReturn):
    myFormula = myReturn.columns[0] + '~ ' + y.columns[0] + '+' + str(myResid.columns[0])

    myList = [myReturn,y,myResid]
    myFactor = pd.concat(myList,axis=1)
    Model=sm.ols(formula = myFormula, data = myFactor).fit()
    signi = Model.pvalues[1]
    adjustR2 = Model.rsquared_adj
    return signi,adjustR2

# 横截面回归
# input     单个Factor的值向量和收益向量
# return    该向量的R2
def CrossSectionReg(oneFactor, myReturn):
    myFormula = myReturn.columns[0] + ' ~ ' + oneFactor.columns[0]
    myList = [oneFactor,myReturn]
    myFactor = pd.concat(myList,axis=1)

    Model=sm.ols(formula = myFormula, data=myFactor).fit()
    adjustR2 = Model.rsquared_adj
    return adjustR2

# 返回一个因子库里，拥有最大R2值的因子的位置
# Input     因子库DataFrame
# Return    最大因子的编号
def maxR2(myBackup):
    # 计算每个备选因子的横截面回归R2
    adjustR2 = list()
    for i in range(1, len(myBackup.columns)):
        myReturn = myBackup.iloc[:,[0]]
        oneFactor = myBackup.iloc[:,[i]]

        myR2 = CrossSectionReg(oneFactor,myReturn)
        adjustR2.append(myR2)
    maxR2 = max(adjustR2)           #最大值
    maxPos = adjustR2.index(maxR2)      #最大值的位置
    return maxPos

#主函数部分
if __name__ == '__main__':

    #-----------------------------------------------------------------------Variables
    #备选因子库，格式DataFrame
    #（目前是test.xlsx，只取了原始数据的3441行）
    BackupFactor = input('test.xlsx') 
    #（目前只测试了日期为200903的部分）
    BackupFactor = BackupFactor.loc[('200903',slice(None)),:]
    #已选因子库，格式list
    PickedFactor = list()
    #因变量向量y，格式DataFrame
    y = pd.DataFrame()
    #收益向量，格式DataFrame
    stockFrame = pd.DataFrame(BackupFactor.iloc[:,[0]])

    step = 1


    #-----------------------------------------------------------------------Filtering
    print('正在筛选因子：')
    pbar = tqdm(total=len(BackupFactor.columns)-1)


    #直到备选因子库里没有因子为止
    while len(BackupFactor.columns) > 1:

        # 若已选因子库为空，则选取R2最大的那个因子作为第一个因子
        if len(PickedFactor) == 0:
            maxPos = maxR2(BackupFactor)
            y = BackupFactor.iloc[:,[maxPos+1]]
            PickedFactor.append(BackupFactor.iloc[:,[maxPos+1]])   #添加到已选因子
            y.columns = ['y_sum']
            #print ("-----------------------------------------")
            #print ("第"+str(step)+"个Factor")
            #print (BackupFactor.iloc[:,[maxWhPos+1]])
            step = step + 1
            del BackupFactor[BackupFactor.columns[maxPos+1]]               #从备选因子库删除
            pbar.update(1)


        # 若已选因子库非空
        else:
            # 每个备选因子，对所有已选因子，做多元回归
            BackupResid = list()  #把回归后的残差项记录到list
            for i in range(1,len(BackupFactor.columns)):
                BackupResid.append(LinearReg(BackupFactor.iloc[:,[i]],y))

            # 每个残差项，和所有已选因子，做FamaMacbeth
            BackupSigni = list()  #把显著性记录到list
            BackupR2 = list()
            Resid_Frame = pd.concat(BackupResid,axis=1)
            Resid_Frame.columns = BackupFactor.columns[1:]
            
            for i in range(len(Resid_Frame.columns)):
                mySigni, myRsquare= FamaMacBeth(Resid_Frame.iloc[:,[i]],y,stockFrame)
                BackupSigni.append(mySigni)
                BackupR2.append(myRsquare)

                
            # 对于不显著的（假设标准是<0.05),将因子剔除
            for i in range(len(BackupSigni)):
                t = len(BackupSigni) - 1 - i
            if BackupSigni[t] > 0.05:
                del BackupResid[t]
                del BackupR2[t]
                del BackupFactor[BackupFactor.columns[t+1]]
                pbar.update(1)
            
            # 对剩下的（显著的），计算每个备选因子的横截面回归R2
            if len(BackupFactor) > 0:          
                maxPos = BackupR2.index(max(BackupR2))#最大值位置

                this_res = pd.DataFrame(BackupResid[maxPos])
                this_res.columns = ['y_sum']
                y = y + this_res
                PickedFactor.append(BackupFactor.iloc[:,[maxPos+1]])   #添加到已选因子
                
                #print ("-----------------------------------------")
                #print ("第"+str(step)+"个Factor")
                #print (BackupFactor.iloc[:,[maxPos+1]])
                step = step + 1
                del BackupFactor[BackupFactor.columns[maxPos+1]]           #从备选因子库删除
                pbar.update(1)
    pbar.close()


    print ("-----------------------------------------")
    print ("共"+str(step)+"个Factor")
    print (pd.concat(PickedFactor,axis=1))
