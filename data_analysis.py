import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_columns",101)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.display.max_rows = 15 #最多显示15行
import warnings
warnings.filterwarnings('ignore') #为了整洁，去除弹出的warnings

df = pd.read_csv('data/cs-training.csv')

#删除第一列Unnamed
df = df.drop(df.columns[0],axis=1)

#使用cut函数，将连续变量转换成分类变量
def binning(col, cut_points, labels=None,isright=True):
    minval = col.min()
    maxval = col.max()
    break_points = [minval] + cut_points + [maxval]
     
    if not labels:
        labels = range(len(cut_points)+1)
    else:
        labels=[str(i+1)+":"+labels[i] for i in range(len(cut_points)+1)]  
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True,right=isright)
    return colBin

def get_frequency(df,col_x,col_y, cut_points, labels,isright=True):
    df_tmp=df[[col_x,col_y]]
    df_tmp['columns_Bin']=binning(df_tmp[col_x], cut_points, labels,isright=isright)
    total_size=df_tmp.shape[0] 
    per_table=pd.pivot_table(df_tmp,index=['columns_Bin'], aggfunc={col_x:[len, lambda x:len(x)/total_size*100],col_y:[np.sum] },values=[col_x,col_y])
    if(per_table.columns[0][0]!=col_x): #假如col_x不在第一列，说明是在第2、3列，就把它们往前挪
        per_table=per_table.reindex_axis((per_table.columns[1],per_table.columns[2],per_table.columns[0]),axis=1)
    per_table[col_y,'percent']=per_table[col_y,'sum']/per_table[col_x,'len']*100
    per_table=per_table.rename(columns={'<lambda>':'percent','len': 'number','sum':'number'})
    per_table=per_table.reindex_axis((per_table.columns[1],per_table.columns[0],per_table.columns[2],per_table.columns[3]),axis=1)
    return per_table

#画出类别分布统计图
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='SeriousDlqin2yrs',data=df,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
df['SeriousDlqin2yrs'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()


#Age
cut_points=[25,35,45,55,65]
labels=['below25', '26-35', '36-45','46-55','56-65','above65']
feq_age=get_frequency(df,'age','SeriousDlqin2yrs', cut_points, labels)

print(feq_age)
y1 = [2.01867, 12.30533, 19.87933, 24.46000, 22.27067, 19.06600]
y2 = [11.16248, 11.12255, 8.81317, 7.59335, 4.58301, 2.41267]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('age',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#RevolvingUtilizationOfUnsecuredLines
cut_points = [0.25,0.5,0.75,1,2]
labels = ["below0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-2.0","above2"]
feq_ratio=get_frequency(df,'RevolvingUtilizationOfUnsecuredLines','SeriousDlqin2yrs', cut_points, labels)

print(feq_ratio)
y1 = [58.43800 , 14.03667, 9.17600, 16.13533, 1.96667, 0.24733]
y2 = [2.13674, 5.29090, 10.12787, 18.21262, 40.10169, 14.55526]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('RevolvingUtilizationOfUnsecuredLines',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#DeptRatio
cut_points = [0.25,0.5,0.75,1,2]
labels = ["below0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-2.0","above2"]
feq_ratio=get_frequency(df,'DebtRatio','SeriousDlqin2yrs', cut_points, labels)

print(feq_ratio)
y1 = [34.90733, 27.56467, 10.48533, 3.61800, 2.72800, 20.69667]
y2 = [5.97009, 6.11653, 9.43540, 10.98213, 13.17204, 5.64342]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('DeptRatio',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#NumberRealEstateLoansOrLines
cut_points=[5,10,15,20]
labels=['below 5', '6-10', '11-15','16-20','above 20']
feq_RealEstate=get_frequency(df,'NumberRealEstateLoansOrLines','SeriousDlqin2yrs', cut_points, labels)

print(feq_RealEstate)
y1 = [99.47133, 0.46600, 0.04667, 0.04667, 0.00667]
y2 = [6.62435, 17.31044, 22.85714, 21.42857, 20.00000]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('NumberRealEstateLoansOrLines',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#NumberOfTime30-59DaysPastDueNotWorse
cut_points=[1,2,3,4,5,6,7]
labels=['0','1','2','3','4','5','6','7 and above',]
feq_30days=get_frequency(df,'NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs', cut_points, labels,isright=False)

print(feq_30days)
y1 = [84.01200, 10.68867, 3.06533, 1.16933, 0.49800, 0.22800,0.09333, 0.06933]
y2 = [4.00022, 15.02526, 26.51153, 35.23375, 42.57028, 45.02924, 52.85714, 48.07692]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('NumberOfTime30-59DaysPastDueNotWorse',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#MonthlyIncome
cut_points=[5000,10000,15000]
labels=['below 5000', '5000-10000','1000-15000','above 15000']
feq_Income=get_frequency(df,'MonthlyIncome','SeriousDlqin2yrs', cut_points, labels)

print(feq_Income)
y1 = [37.23933, 30.72733, 8.69000, 3.52267]
y2 = [8.61634, 5.97080, 4.19639, 4.63664]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('MonthlyIncome',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#NumberOfDependents
cut_points = [1,2,3,4,5]
labels = ["0","1","2","3","4","5 and more"]
feq_dependent=get_frequency(df,'NumberOfDependents','SeriousDlqin2yrs', cut_points, labels,isright=False)

print(feq_dependent)
y1 = [57.93467, 17.54400, 13.01467, 6.32200, 1.90800, 0.66000]
y2 = [5.86293, 7.35294, 8.11392, 8.82632, 10.37736, 10.00000]
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(labels, y1,'o-', c='orangered',label='Proportion of the number of people', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
plt.legend(loc=2)
ax2 = ax1.twinx() # 创建第二个坐标轴
ax2.plot(labels, y2, 'o-', c='blue',label='Loan default rate', linewidth = 1) #同上
plt.legend(loc=1)
ax1.set_xlabel('NumberOfDependents',size=12)
ax1.set_ylabel('Proportion of the number of people',size=12)
ax2.set_ylabel('Loan default rate', size=12)
plt.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
plt.show()


#NumberOfOpenCreditLinesAndLoans
cut_points=[5,10,15,20,25,30]
labels=['below 5', '6-10', '11-15','16-20','21-25','26-30','above 30']
feq_OpenCredit=get_frequency(df,'NumberOfOpenCreditLinesAndLoans','SeriousDlqin2yrs', cut_points, labels)

print(feq_OpenCredit)
y = [8.41812, 5.53808, 6.18147, 6.86573, 6.72298, 7.89809, 7.34463]
plt.ylim((0, 20))
plt.yticks([0, 10.0, 20.0])
plt.plot(labels, y, 'o-', c='blue', label='Loan default rate', linewidth = 1)
plt.xlabel('NumberOfOpenCreditLinesAndLoans', size=12)
plt.ylabel('Loan default rate', size=12)
plt.show()