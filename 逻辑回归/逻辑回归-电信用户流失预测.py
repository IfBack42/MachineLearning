"""
one-hot热编码：为分类变量的每一个可能的类别 创建一个新的二进制特征列（虚拟变量, Dummy Variable）
    比如，一列水果中，1代表苹果，2代表香蕉，3代表梨子，处理后三个水果一人一列，分类为该水果的话这列则为1否则为0
    优点：消除了类别之间可能存在的错误数值关系。易于理解和实现。适合处理名义数据（Nominal Data），即类别之间没有内在顺序关系的数据
    缺点：维度灾难；多重共线性
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report #最后一个，分类评估报告
#                                                           分类评估报告不适用于回归模型，多标签分类模型，无监督学习模型


#1. 数据预处理
def preprocessing():
    churn_data = pd.read_csv('./data/churn.csv')
    churn_data.info()
    # 把object类型的性别数据 使用one-hot热编码 转为bool类型
    churn_df = pd.get_dummies(churn_data,columns=['Churn','gender'])
    print(churn_df)
    churn_df.info()
    churn_df.dropna(inplace=True)
    # 删掉多余的列，比如Male Female，留一个就好
    churn_df.drop(['Churn_No','gender_Female'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'label'},inplace=True)
    print(churn_df.label.value_counts())
    print(churn_df.columns) # ['Partner_att', 'Dependents_att', 'landline', 'internet_att','internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month','Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic','MonthlyCharges', 'TotalCharges', 'label', 'gender_Male']
    return churn_df

#2. 可视化
def visual(churn_df):
    #                                 x轴列名👇    分组字段👇
    sns.countplot(data=churn_df,x='Contract_Month',hue='label')
    plt.show()

#3. 模型训练
def model(churn_df):
    #3.1 特征选取 月度会员，是否有互联网服务，是否是电子支付
    x = churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    x_charge = churn_df['TotalCharges'] # 偷偷加了一个收费特征，玩
    x_charge = churn_df['TotalCharges'].values.reshape(-1,1) #注意series对象是1维数组，转为2维才能标准化
    x_charge = churn_df[['TotalCharges']] # 或者这样操作
    y = churn_df['label'] # 0 流失 ，1 不流失
    #3.2 收费列标准化
    transfer = StandardScaler()
    x_charge = transfer.fit_transform(x_charge)
    x_charge = pd.DataFrame(x_charge,columns=['TotalCharges'])
    x = pd.concat([x,x_charge],axis=1)
    #3.3 划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    #3.4 模型训练
    #    LogisticRegressor参数：penalty正则化类型（"l1"，"l2"，"elasticnet",None）;C正则化强度；
    #    solver优化算法（'newton-cg','lbfgs','liblinear','sag','saga');max_iter最大迭代次数。
    estimator = LogisticRegression(penalty='l2',solver='lbfgs')
    estimator.fit(x_train,y_train)
    pre_result = estimator.predict(x_test)
    #3.5 模型评估
    print(f"准确率：{accuracy_score(pre_result,y_test)}")
    print(f"精确率：{precision_score(y_test,pre_result,pos_label=1)}")
    print(f"召回率：{recall_score(y_test,pre_result,pos_label=1)}")
    print(f"F1值：{f1_score(y_test,pre_result,pos_label=1)}")
    # macro avg宏平均，不考虑权重直接求平均，适用于数据均衡
    #weight avg权重平均，考虑样本权重求平均，适用于数据不均衡
    print(f'分类评估报告:\n{classification_report(y_test,pre_result)}')


if __name__ == '__main__':
    churn_df = preprocessing()
    # visual(churn_df)
    model(churn_df)