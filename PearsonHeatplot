#热力图
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xlwt
import xlrd
import mrcfile
import seaborn as sb ###绘制相关系数的热力图
import openpyxl ###读取excel

stdsc = StandardScaler()

df=pd.read_excel(r'F:\\KNN\\数据\\压电性0919\\d33_0925.xls',sheet_name=2)  #"SHEETNAME"
#df.columns = ['d33 (pC/N)','iB','iA*iB','kp','Tc(℃)','容差因子','P-A(10-23cm3)','P-B','Rb(pm)',
          # 'Ra*Rb','Ra/Rb','Amass','Bmass','Amass*Bmass','Ea(pauling)','Eb(p)','Ea*Eb(p)',
           #'Ea-绝对标度','Eb(a)','Ea*Eb(a)','Za(电离能(eV))','A-aff 电子亲和力','B-aff','ρ-A(g/cm3)','ρ-B']
features = pd.DataFrame(df, columns=df.columns[1:])#提取特征集
print('-------------------------')
print(features)
print('-------------------------')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df_coor=df.corr(method='pearson') #求相关性method='spearman'
print(np.around(df_coor,2))
Data = np.array(np.around(df_coor,2))
filename = xlwt.Workbook()
sheet1 = filename.add_sheet(u'sheet1',cell_overwrite_ok=True)
[h,l]=Data.shape
for i in range (h):
    for j in range (l):
        sheet1.write(i,j,Data[i,j])
filename.save('data_hot_d33.xls')
#特征间相互关系
#sns.set(style='whitegrid',context='notebook')
#cols = ['d33 (pC/N)','iB','iA*iB','kp','Tc(℃)','容差因子']
#sns.pairplot(df[cols], height=2.5)
#sns.set(font="simhei")
#plt.rcParams['axes.unicode_minus']=False
#plt.show()
#热力图
############只绘制左下角的热力图####################
#mask为 和相关系数矩阵xcorr一样大的 全0(False)矩阵
mask55 = np.zeros_like(df_coor, dtype=bool)##显示左下角
# 将mask右上三角(列号》=行号)设置为True
#print(mask55)
mask55[np.triu_indices_from(mask55)] = True

sb.heatmap(data=df_coor,mask=mask55,annot=True,vmax=1,square=True,annot_kws={'size':7})#sb.heatmap(data = r_pearson,cmap="YlGnBu")
ax = plt.gca()###正方形
ax.set_aspect(1)###正方形
plt.tick_params(length=1.5,width=0.3,pad=1.5,labelsize=8,bottom=True,left=True)######刻显示度线
# plt.show()
####将矩阵写入表格#######
# pearson_matrix = round(df.corr(method='pearson',min_periods=1),2)###皮尔逊相关系数矩阵
# print(pearson_matrix)
# #pearson_matrix = np.tril(pearson_matrix,0)###上三角矩阵
pm_np_tril = np.triu(Data,0)###下三角矩阵
pm_np = pd.DataFrame(pm_np_tril)
#print(pm_np)###皮尔逊相关系数（数组）
pm_np.to_excel(r'皮尔逊相关系数矩阵_d33.xlsx')#####矩阵写入表格

# ######画全图#########
# sns.set(font_scale=2)
# sns.set(font="simhei")
# plt.rcParams['axes.unicode_minus']=False
# plt.subplots(figsize=(50,50))
# h = sns.heatmap(df_coor,annot=True,vmax=1,square=True, cmap="Blues") #linewidths = 0.05,linecolor= 'white')
# #cb = h.figure.colorbar(h.collections[0])
# #cb.ax.tick_params(labelsize=1)
# plt.rcParams['axes.unicode_minus'] = False #坐标轴显示正负号
# plt.rcParams['font.sans-serif'] = ['SimHei']
# #h.set_yticklabels(h.get_yticklabels(),rotation=45)  #设置坐标轴刻度角度
# #h.set_xticklabels(h.get_xticklabels(),rotation=45)
# plt.show()
# ###########
# h.get_figure().savefig('df_corr.png',bbox_inches='tight',transparent=True)
#bbox_inches让图片显示完整，transparent=True让图片背景透明

############准备进行相关性筛选##########
def diNhan(lie, rmax):
    '''筛选出高度相关的特征，用于去除元素性质中高度相关的特征'''
    # 打开excel文件
    wb = openpyxl.load_workbook('皮尔逊相关系数矩阵_d33.xlsx')
    # 获取活跃表对象
    sheet = wb.active
    row_num = sheet.max_row  # 获取当前表中最大的行数
    i = 0
    for row in range(1, row_num + 1):
        cell = sheet.cell(row + 1, lie + 1)
        try:
            if abs(float(cell.value)) >= rmax:
                i += 1
                # print(cell.value, '重复次数：', i)
        except:
            pass
    if i > 1:####输出可被其他性质替代的性质（高线性相关）
        return sheet.cell(1, lie + 1).value
#####################
feature_highly_correlated = []
wb = openpyxl.load_workbook('皮尔逊相关系数矩阵_d33.xlsx')
# 获取活跃表对象
sheet = wb.active
row_num = sheet.max_row
for iii in range(1, row_num+1):
    ###第二个参数为相关性系数的极值
    feature_highly_correlated += [diNhan(iii,0.9)]####将高度相关的特征存入列表，第一个为0！
    #print(diNhan(iii,0.85))

print(feature_highly_correlated)
#将None去掉，留下高度相关的序号
feature_highly_correlated_list = []
for i in feature_highly_correlated:
    if i != None:
        feature_highly_correlated_list += [i+1]####两个Excel的差一行一列
print('-------高度相关的特征---------')
print(feature_highly_correlated_list)
#print(features.columns[feature_highly_correlated_list])####差1
print('------------------------')
#########
feature_highly_correlated_list_55 = list(range(1,len(features.columns)+1))
#########全部特征去掉高相关的性质
feature_low_correlated_list_55_last = list(set(feature_highly_correlated_list_55)-set(feature_highly_correlated_list))
print('----------低相关的特征--------------')
print(feature_low_correlated_list_55_last)
a_feature_low_correlated_list_55_last=[ i-1 for i in feature_low_correlated_list_55_last]
print(features.columns[a_feature_low_correlated_list_55_last])
print('------------------------')
features_low_correlation = pd.DataFrame(df, columns=df.columns[feature_low_correlated_list_55_last])
print(features_low_correlation)

label_low_correlation = pd.DataFrame(df, columns=[])#提取标签列
boston_low_correlation = pd.concat([features_low_correlation, label_low_correlation], axis=1)#合并特征集和标签列
####
r_pearson_low_correlation = boston_low_correlation.corr()
sb.set_theme(style='whitegrid', font='Times New Roman', font_scale=0.5)#####配置风格、字体、字号
plt.figure(dpi=600, figsize=(50, 50)) # width and height in inches
########导出筛选后的元素性质
# feature_low_correlated_list_55_last
###feature_low_correlated_list_55_last为筛选后的特征序号，差一个元素符号[0]
filter_label = [0]+feature_low_correlated_list_55_last
print(filter_label)
df=pd.read_excel(r'F:\\KNN\\数据\\透明性0506\\AB-Nanoenergy.xlsx',sheet_name=5, usecols=filter_label)####留下低相关性的特征（目前为28个）
df.to_excel(r'筛选后的元素性质_d33.xlsx', sheet_name='filter', index=False)
