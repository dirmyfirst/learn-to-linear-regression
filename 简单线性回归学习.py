import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#导入数据
data = pd.read_csv('test.csv',encoding='gbk')

#画出散点图，求x和y的相关系数
plt.scatter(data.活动推广费,data.销售额)

print(data.corr())

#估计模型参数，建立回归模型
'''
(1) 首先导入简单线性回归的求解类LinearRegression
(2) 然后使用该类进行建模，得到lrModel的模型变量
'''

lrModel = LinearRegression()
#(3) 接着，我们把自变量和因变量选择出来
x = data[['活动推广费']]    #自变量
y = data[['销售额']]   #因变量

#模型训练
'''
调用模型的fit方法，对模型进行训练
这个训练过程就是参数求解的过程
并对模型进行拟合
'''
lrModel.fit(x,y)

#查看模型训练后的评分结果
print(lrModel.score(x,y))


#利用回归模型进行预测
#预测投入80,90万的推广费，能得到多少的销售额
print(lrModel.predict([[80],[90]]))


'''
    sklearn建模流程

        建立模型
            lrModel = sklearn.linear_model.LinearRegression()

        训练模型
            lrModel.fit(x,y)

        模型评估
            lrModel.score(x,y)

        模型预测
            lrModel.predict(x)
'''
