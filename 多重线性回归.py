import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression

'''
回归分析步骤

    根据预测目标，确定自变量和因变量

    绘制散点图，确定回归模型类型

    估计模型参数，建立回归模型

    对回归模型进行检验

    利用回归模型进行预测
'''

#获取数据
data = pd.read_csv('test2.csv',encoding='gbk')
print(data)

#绘制散点图，确定回归模型类型
font = {
    'family':'SimHei'
}
plt.rc('font',**font)

scatter_matrix(
    data[["百分比利率","抽取用户佣金","金融产品销售额"]],
    figsize =(10,10),diagonal = 'kid'
)
#plt.show()

print(data[["百分比利率","抽取用户佣金","金融产品销售额"]].corr())

#确定自变量与因变量
x = data[["百分比利率","抽取用户佣金"]]  #自变量
y = data[["金融产品销售额"]]   #因变量


#绘制散点图，确定回归模型类型
lrModel = LinearRegression()

#训练模型
lrModel.fit(x,y)

#查看模型训练后的评分结果
print(lrModel.score(x,y))

#预测
print(lrModel.predict([[11, 50]]))

