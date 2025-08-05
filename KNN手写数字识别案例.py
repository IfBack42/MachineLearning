"""
KNN识别手写数字灰度图
每张图都是由 28*28 像素组成的，表示为 一行数据有784个像素点，每个像素点的值为该像素的颜色

"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score              # 模型评估参数
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # 用于保存训练好的模型，记录参数
from collections import Counter


def data_load():
    px_pd = pd.read_csv("./data/手写数字识别.csv")
    return px_pd

# 定义图像显示函数，把像素数值显示为灰度图
def pic_visual(px_pd,idx):
    y = px_pd.iloc[:, 0]
    x = px_pd.iloc[:, 1:]
    print(f"所有标签分布情况：{y.value_counts()}")   # 为series对象，使用效果同Counter（y）（字典），不用导格外的包
    print(f"{idx}索引对应数字是：{y.iloc[idx]}")                  # 索引对应数字
    print(f"{idx}索引所在行形状为：{x.iloc[idx,:].shape}")        # 索引所在行形状为：(784,)
    reshaped_x = x.iloc[idx,:].values.reshape(28,28)           # 重塑型后为：(28, 28)
    print(f"重塑型后为：{reshaped_x.shape}")
    plt.imshow(reshaped_x,cmap='gray')   # cmap -> 将数值转化为的图像类别，‘gray’展示灰度图
    plt.axis('off')                      # 关闭显示灰度图的坐标
    plt.show()

def model_train(px_pd):
    y = px_pd.iloc[:, 0].to_numpy()  # to_numpy -> 拿DF和SR对象的数组
    x = px_pd.iloc[:, 1:].to_numpy()
    # 因为数据比较干净所以使用归一化
    transfer = MinMaxScaler()
    x = transfer.fit_transform(x)
    # 划分数据集 👇划分顺序不能乱,报错几次了
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,stratify=y) # stratify参考标签进行抽取，防止数据集极端分布
    # 创建模型，丢给网格搜索和交叉验证
    estimator = KNeighborsClassifier(n_neighbors=3)
    # # 定义可能的参数组合
    # param_dict = {'n_neighbors':[i for i in range(1,11)]}
    # # 定义网格搜索交叉验证模型
    # search_model = GridSearchCV(estimator=estimator,param_grid=param_dict,cv=4)
    # # 训练模型，找到最佳参数
    # search_model.fit(x_train,y_train)
    # # 拿到最佳参数和模型
    # print(f"最优评分：{search_model.best_score_}")
    # print(f"最优超参组合：{search_model.best_params_}")
    # print(f"最优模型对象：{search_model.best_estimator_}")
    # print(f"具体交叉验证结果：{search_model.cv_results_}")
    # estimator = search_model.best_estimator_
    # 最后训练最优模型
    estimator.fit(x_train,y_train)
    result = estimator.predict(x_test)
    # 模型评分
    print(f"模型准确率：{accuracy_score(y_test,result)}")
    print(f"模型准确率：{estimator.score(x_test,y_test)}")
    # 保存模型
    joblib.dump(estimator,"./手写数字识别模型.pkl") #pkl -> pickle文件

def use_model():  # test开头的函数会调用专门的包
    # 加载数据
    im_data = plt.imread('./data0/demo.png') # 其实就是特征二维数据, x
    print(im_data.shape)
    x = im_data.reshape(1,-1) # -1 表示能转为多少列转为多少列
    print(x) # ⭐这里的图片像素范围是(0,1) 而不是csv数据集的 (0,255)
    # 不要 进行数据归一化
    # transfer = MinMaxScaler()
    # x = transfer.fit_transform(x)
    ## 显示图片
    # plt.imshow(im_data,cmap='gray')
    # plt.axis('off')
    # plt.show()
    # 加载模型
    estimator = joblib.load('./手写数字识别模型.pkl')
    # 模型预测
    pre_result = estimator.predict(x)
    print(pre_result)



if __name__ == '__main__':
    px_pd = data_load()
    # idx = int(input("?"))
    # pic_visual(px_pd,idx)
    # model_train(px_pd)
    use_model()