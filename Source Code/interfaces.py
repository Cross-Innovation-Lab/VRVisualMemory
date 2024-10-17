import pandas as pd
from pandas import DataFrame,Series
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score,accuracy_score,recall_score
from pathlib import Path

from sklearn.metrics import f1_score, r2_score,mean_squared_error,mean_absolute_error,accuracy_score,precision_score, recall_score

class subject():
    def __init__(self):
        self.l_e=DataFrame()#脸部眼部数据
        self.lip=DataFrame()#脸部数据
        self.eye=DataFrame()#眼部数据
        self.image=DataFrame() #对应的帧数据
        self.emo_score=dict()
        self.prea_score=-1#对应的前测正向情绪得分数据
        self.pred_score=-1#对应的前测负向情绪得分数据
        self.afta_score = -1  # 对应的后测正向情绪得分数据
        self.aftd_score = -1  # 对应的后测负向情绪得分数据
        self.r_time=dict()#对应的反应时数据
class periods(ABC):
    pass

class dataConstruct(ABC):
    def __init__(self):
        self.datasets=dict()
        self.subjects=dict() #存储每个被试的数据。被试编号为键，值为对应的subject对象
    @abstractmethod
    def load_data(self):
        '''
        遍历被试文件夹，读取每个被试对应的脸部和眼部数据并，合并为一个dataframe
         将对应被试的准确率数据存入subjects字典，键为被试编号，值为对应的数据
        对于image_datas，每个被试有两个列表，对应了所处的实验状态和实验时间，实验时间和脸，眼部数据的时间戳对应
        '''
        pass

    @abstractmethod
    def feature_clean(self,sample_rate=5):
        '''
        清洗lip_eye_datas中的数据，提取特征
        去除lip_eye_feature中各行数据的离群点，有离群的数据就把一行都删了
        对于每一个被试，对数据进行采样,采样率为sample_rate,默认为每5帧采样一次
        采样后更新result中的数据，保证最终能匹配采样数据和result字典中对象的数据
        根据image.csv中的数据所处状态，去除等待状态中的实验数据
        最终将lip_datas更新一个新的采样后被试dataframe的列表，保留时间戳

        '''
        pass

    @abstractmethod
    def feature_extract(self):
        '''
        对每个被试的脸部和眼部数据的每个特征，计算均值和方差，形成对应的新的特征
        对于在feature_clean中生成的每一个dataframe，形成一行新的数据作为预测的特征，并将result中对应的正确率情况加入该行
        最终结果存入dataset中
        '''
        pass

    @abstractmethod
    def data_divide(self):
        '''
        基于image_datas对数据进行划分和清洗，划分的依据有两类，首先是依据场景进行划分，如白天黑夜，墙壁颜色，然后在对应场景下分出
        - 记忆状态
        - 回忆状态
        - 整体实验
        三类，因此，每个被试的数据可以划分为
        '''
        pass

class ML_record():
    score_dict={}#记录每次预测的评分，并按照属性进行分组划分
    attributes=[]
    def __init__(self,attribute_name:list):
        self.score_dict = {}  # 记录每次预测的评分，并按照属性进行分组划分
        self.res_dict = {}  # 记录每次预测的结果
        self.score_dict['MAE'] = []
        self.score_dict['MSE'] = []
        self.score_dict['R2'] = []
        self.score_dict["dataset"]=[]
        self.attributes = []
        self.attributes = attribute_name
        for atr in attribute_name:
            self.score_dict[atr] = []
    # 传入每次预测得到的结果以及对应属性的字典
    def rec_predict(self,y,pre_y,dataset,atr):
        self.score_dict['MAE'].append(mean_absolute_error(y_true=y,y_pred=pre_y))
        self.score_dict['MSE'].append(mean_squared_error(y_true=y, y_pred=pre_y))
        self.score_dict['R2'].append(r2_score(y_true=y, y_pred=pre_y))
        self.score_dict['dataset'].append(dataset)
        for attribute in self.attributes:
            self.score_dict[attribute].append(atr[attribute])

    #         保存最终结果到目标文件夹
    def save_csv_r_time(self,name):
        path1=Path("predict_result/r_time",name+"score.csv")
        for i,i2 in self.score_dict.items():
            print(i,":",len(i2))
        res_score = pd.DataFrame(self.score_dict)
        res_score.to_csv(path1)


