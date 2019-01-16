## README
### GoogLeNet 提取的特征数据
* traincat.mat 利用Matlab r2017b自带的 GoogLeNet包提取的训练数据特征，猫矩阵，包含1981张猫图片对应的每张图片1024个特征
* traindog.mat 同上，狗矩阵，1024特征 x 1930
* test25000.mat 利用训练出的GoogLeNet模型提取的待分类的25000张猫狗图片特征，1024特征 x 25000 其中前12500列为猫图片，后12500列为狗特征

### Classifiers
* catordogadboost.py: AdaBoost
* catordoggbdt.py: GBDT
* catordog.py: XGBoost

Cross validation在注释中的cv part， 把其他部分注释，cv part取消注释就可以跑cv.
