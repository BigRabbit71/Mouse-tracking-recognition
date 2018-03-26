# Mouse-tracking-recognition
鼠标轨迹识别 for 2017年“中国高校计算机大赛—大数据挑战赛”（Big Data Challenge）

比赛详情：http://bdc.saikr.com/vse/bdc/2017

### Task
利用人机验证过程中的鼠标轨迹数据、目标点坐标，将验证行为分为机器行为/人类行为的二分类问题。

### Data Set
- 训练数据集
  - 3000条
  - 数据格式：编号id 鼠标轨迹(x,y,t) 目标点坐标(x,y) label:1 for 正常轨迹，0 for 机器轨迹
- 测试数据集
  - 10w条for初赛
  - 200w条for复赛
  - 数据格式：编号id 鼠标轨迹(x,y,t) 目标点坐标(x,y)
  
### Score
F = (5PR/(2P+3R)) * 100

=> beta^2= 2/3, 查准率更重要。

### 数据探索
- 可视化（由于主办方的开源协议，略去不表）
- 观察数据，提取特征
 - 轨迹是否有回退
 - 位置特征
 - 速度特征
 - 加速度特征
 - 角度特征

### Model Selection
- GBDT for 初赛
 - 61 features
 - 框架参数：
   - n_estimators: 50
   - subsample: 0.7
   - learning_rate: 0.01
   - loss: deviance
 - 基学习器参数：
   - max_depth: 9
   - min_sample_split: 50
   - max_features: 15
   - min_sample_leaf:4

- RF for 复赛
 - 42 features
 - 参数：
   - 决策回归树颗数：100
   - max_features：30
   - threshold： 0.87（理由：模型召回率太低，提高门限来提高召回率）

### 最终结果
- 初赛：
   - Score：85.33
   - 排名：105/1222
- 复赛：
   - Score：60.78
   - 排名：85/104
  
### 赛后思考
- 应该尝试stacking，进行model ensemble的==
- 应该找个队友一起玩的==
- 对模型的理解还不深入，应该考虑清楚的==

