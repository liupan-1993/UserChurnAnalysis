# spark报告
## 项目背景说明
这是由udacity提供的关于某音乐平台用户使用情况的数据集。数据集包含了用户基本信息，及用户在各个时间点访问平台的具体信息。平台现在面临用户流失（取消服务）的问题，此次分析的目的是利现有的数据进行建模，以预测用户是否可能流失。我将利用逻辑回归、决策树、随机森林等机器学习算法进行建模，并用各模型的F1_score来选择模型。

## 分析
* 由于spark在处理小数据集时速度远比不上pandas，所以先用pandas对数据进行探索，读取数据。 

      df = pd.read_json('mini_sparkify_event_data.json',lines=True)

* 查看数据结构

      df.dtypes
|列名|数据类型|
|---|---|
| auth:| string|
| firstName:|| string|
| gender:| string|
| itemInSession:| long|
 | lastName:| string|
 | length:| double|
 | level:| string|
 | location:| string|
 | method:| string|
 | page:| string |
 | registration:| long|
 | sessionId:| long|
 | song:| string|
 | status:| long|
 | ts:| long|
 | userAgent:| string|
 | userId:| string|
 
* 查看各列的空值情况

      pd.isnull(df).sum()     
 |列名|空值数量|
 |---|---|
| ts           |        0|
|userId         |      0|
|sessionId     |       0|
|page          |       0|
|auth          |       0|
|method        |       0|
|status        |       0|
|level         |       0|
|itemInSession |       0|
|location      |    8346|
|userAgent     |    8346|
|lastName      |    8346|
|firstName     |    8346|
|registration  |    8346|
|gender        |    8346|
|artist        |   58392|
|song          |   58392|
|length        |   58392|

location/gender/name等页面同样数量的空值，查看数据可知空值是由于非注册用户访问导致的，这些数据与此次的分析无关，直接删除
artist/song/length也有相同数量的空值，查看数据可知是由于用户访问了非播放页面（登入登出点赞等）导致的，这些数据很有可能与用户是否删除服务相关，不能直接删除

* 查看length的分布情况

       plt.plot(df.length)
       
全部的听歌时长都小于3000s，符合常识，没有异常值需要处理

## 数据预处理
* 首先是ts列，该列记录了时间戳。先将时间戳转化为datetime格式，利用udf自建函数进行转换

      get_time = udf(lambda x : datetime.datetime.fromtimestamp(x/1000))
  
      df_spark = df_spark.withColumn('dateTime', get_time(df_spark.ts))

* 转换完成后发现该数据集统计的是10月1日到12月3日之间的数字，为了便于后续的统计，我将日期全部转换为距离10月1日的天数，并存为'days'列。定义get_day函数利用udf进行转换。


      def get_day(x):

         a = x.month

         if a == 12:

            return 61+x.day

         return 31 * (x.month - 10) + x.day
      getDay = udf(get_day)
      df_spark = df_spark.withColumn('days', getDay(df_spark.dateTime))

* 处理level,gender列,用0 1替换原有值。udf函数进行转换

      change_type_level = udf(lambda x : int(x == 'paid'),IntegerType())
      df_spark = df_spark.withColumn('isPaid', change_type_level(df_spark.level))
      df_spark = df_spark.drop('level')

      change_type_gender = udf(lambda x : int(x == 'M'),IntegerType())
      df_spark = df_spark.withColumn('isMale', change_type_gender(df_spark.gender))
      df_spark = df_spark.drop('gender')
* 处理userAgent列，useragent显示了用户使用的设备，生活经验告诉我不同平台的用户消费习惯是不同的，故可将设备分为苹果设备和非苹果设备。转换后的新列为isApple列

      is_apple = udf(lambda x : 1 if x >0 else 0, IntegerType())
      df_spark = df_spark.withColumn('isApple', is_apple(instr(df_spark.userAgent, 'Mac')))
* 处理page列，page详细的记录了每个用户访问的页面，此列是最为关键的列，可以挖掘出非常多有用的用户信息。

page列的取值有NextSong|Home|Thumbs Up|Add to Playlist|Add Friend|Roll Advert|Login|Logout|Thumbs Down|Downgrade|Help|Settings|About       |Upgrade|Save Settings|Error|Submit Upgrade|Submit Downgrade|Cancellation Confirmation|Cancel|Register|Submit Registration 

首先我关心用户是否取消了服务，用户是否访问和cancel、upgrade相关的页面就至关重要。用户的听歌情况如是否点赞、添加到歌单、下一曲、点踩等是需要留意的。还有免费用户的广告出现次数，用户有多少次登入和登出行为，是否出现了错误界面都是我关心的。
下面列出我关心的值'Cancel','Thumbs Up',’'Add Friend','Add to Playlist','Logout','Login','Roll Advert','Downgrade','Error','Upgrade','Submit Upgrade','Submit Downgrade'，以各值为名称新建列，新列中1表示出现了该值0表示未出现该值。定义转换函数然后for循环遍历要转换的列表。

      def udf_is_equal(target_col, result_col, df): 
          """处理page中的值"""
          udf_ = udf(lambda x : int(x == target_col),IntegerType())
          return df.withColumn(result_col, udf_(df.page))
      #page列需要转换的页面名
      transform_list = ['Cancel','Thumbs Up','Add Friend','Add to Playlist'\
                        ,'Logout','Login','Roll Advert','Downgrade','Error'\
                        ,'Upgrade','Submit Upgrade','Submit Downgrade']
      #转换为独热列并添加到df_spark
      for i in transform_list:
          df_spark = udf_is_equal(i, i.replace(' ',''), df_spark)
          
**现在对原始数据的初步处理完成了，我们新建了许多的新列。接下来就是按照用户对各列进行统计了。**

* 对于'isMale','isApple','Cancel','downgrade','error','submitUpgrade','upgrade'等列，我将只统计用户是否是男、是否取消了服务、是否使用的是苹果设备等，而不进行计数统计。因为对于这些列来说要么计数统计是错误的（ismale、isapple、cancel），要么计数统计对结果帮助不大。
我将创建一个df_target来保存统计后的数据。定义转换函数然后for循环遍历要转换的列表，将每个遍历的值分别利用groupBy进行统计，将转换结果join到df_target。

      def agg_max_by_id(column_name,df_spark):
          return df_spark\
              .groupBy('userId')\
              .agg({column_name:'max'})\
              .withColumnRenamed('max('+column_name+')',column_name)
      df_target = agg_max_by_id('isPaid',df_spark)
      column_one_hot = ['isMale','isApple','Cancel','downgrade','error','submitUpgrade','upgrade']
      for i in column_one_hot:
          right_df = agg_max_by_id(i, df_spark)
          df_target = df_target\
              .join(right_df, df_target.userId == right_df.userId)\
              .drop(right_df.userId)
* 对于'thumbsUp','addFriend','length','logIn','RollAdvert','logOut'等列，就需要计数了，比如一个用户近期添加了多少朋友、听歌时长、电站次数、登入次数、看广告的次数等直接或间接的表明了用户近期的活跃程度、体验如何。

      column_count = ['thumbsUp','addFriend','length','logIn','RollAdvert','logOut']
      for i in column_count:
          right_df = df_spark\
              .groupBy('userId')\
              .agg({i:'sum'})\
              .withColumnRenamed('sum('+i+')', i)
          df_target = df_target\
              .join(right_df, df_target.userId == right_df.userId)\
              .drop(right_df.userId)
* 我原计划统计用户每日的听歌情况，或者利用window窗口函数对用户不同周期的听歌状况做统计。但是由于没有云部署条件，统计完成后特征列过多，在本地执行时会报内存不足的错误，故放弃。使用用户平均听歌时长来替代。

      right_df = df_target\
          .select('userId', df_target['length']/df_target['count(song)'])\
          .withColumnRenamed('(length / count(song))', 'lengthBySong')
      df_target = df_target\
          .join(right_df, df_target.userId == right_df.userId)\
          .drop(right_df.userId)
* 接下来对atrist、song列，统计用户具体听了多少歌，这些歌来出自几位艺术家之手。当然如果条件允许还可以对这两列作进一步的细分处理，例如艺术家的类型、歌曲类型、是否是热门歌手、歌手年代、歌曲年代等。通过这些信息可以更加完整的建立用户画像。

      right_df = df_spark\
          .groupBy('userId')\
          .agg({'artist':'count'})
      df_target = df_target\
          .join(right_df, df_target.userId == right_df.userId)\
          .drop(right_df.userId)
      right_df = df_spark\
          .groupBy('userId')\
          .agg({'song':'count'})
      df_target = df_target\
          .join(right_df, df_target.userId == right_df.userId)\
          .drop(right_df.userId)
* 用户每次在访问home页面后的平均听歌时长也是用户活跃度的一个体现。接下来对此进行统计。先标记每次登陆home的页面（访问home记1），利用window窗口函数进行前向的求和，然后用groupBy函数统计每次登陆听歌次数

      function = udf(lambda ishome : int(ishome == 'Home'), IntegerType())

      user_window = Window \
          .partitionBy('userID') \
          .orderBy(desc('ts')) \
          .rangeBetween(Window.unboundedPreceding, 0)

      cusum = df_spark.filter((df_spark.page == 'NextSong') | (df_spark.page == 'Home')) \
          .select('userID', 'page', 'ts') \
          .withColumn('homevisit', function(col('page'))) \
          .withColumn('period', sum('homevisit').over(user_window))
      right_df = cusum.filter((cusum.page == 'NextSong')) \
          .groupBy('userID', 'period') \
          .agg({'period':'count'})\
          .groupBy('userId')\
          .agg({'count(period)':'avg'})\
          .withColumnRenamed('avg(count(period))','avgSongVistHome')\
          .drop('period')
      df_target = df_target.join(right_df, df_target.userId == right_df.userId)\
          .drop(right_df.userId)
 以上就是我处理数据的完整过程。
统计完成后的df_target数据结构如下
|列名|类型|
|---|---|
| isPaid| integer |
 | isMale |integer |
 | isApple |integer |
 | Cancel |integer |
 | logOut| integer |
 | downgrade |integer |
 | error |integer |
 | submitUpgrade |integer |
 | thumbsUp |long |
 | addFriend |long |
 | length| double |
 | logIn |long |
 | RollAdvert| long |
 | upgrade| long |
 | count(artist) |long |
 | count(song)| long |
 | userId |string |
 | lengthBySong| double |
 | avgSongVistHome |double |

## 建模
建模之前需先进行向量化

      df_target = VectorAssembler(inputCols = target_column, outputCol = 'VecFeatures').transform(df_target)
接下来归一化数据

      mmscaler = MinMaxScaler(inputCol='VecFeatures',outputCol='features')
      model = mmscaler.fit(df_feature_label)

将226名用户按照0.7/0.15/0.15的比例划分为训练集/测试集/验证集

      trainingData, testData, valData = df_feature_label.randomSplit([0.6,0.2,0.2],42)

接下来用逻辑回归训练，网格搜索正则化率[0.1, 0.2, 0.4, 0.8]

      lr = LogisticRegression(maxIter = 20, regParam = 0.0, elasticNetParam=0)

      grid_lr = ParamGridBuilder()\
          .addGrid(lr.regParam, [0.1, 0.2, 0.4, 0.8])\
          .build()
      crossval_lr = CrossValidator(estimator=lr
                    ,estimatorParamMaps = grid_lr
                    ,evaluator= MulticlassClassificationEvaluator())

      crossval_lr_fit = crossval_lr.fit(trainingData)

      result_training_lr = crossval_lr_fit.transform(trainingData)
      result_test_lr = crossval_lr_fit.transform(testData)
      result_val_lr = crossval_lr_fit.transform(valData)
训练完成后得到的模型f1_score分别为训练集0.68,测试集0.73，验证集0.58。测试集的得分高于训练集，这是由于我们的数据量太少了导致的，测试集样本数只有30个而且是比较稀疏的数据。所以结果受采样的影响就非常大。还需要部署到更大的数据样本中进行训练才能得到比较可靠的结果。而受限于计算能力，我并没有将所有可能调整的参数代入网格进行搜索，比如elasticNetParam（正则化方式）、maxIter（最大迭代次数）、tol（收敛极限），只带入了regParam（正则化率），也就是说现在得到的模型还没有达到最优状态

使用决策树训练

      dt = DecisionTreeClassifier()
      model_dt = dt.fit(trainingData)
      result_training_dt = model_dt.transform(trainingData)
      result_test_dt = model_dt.transform(testData)
      resulr_val_dt = model_dt.transform(valData)
我使用决策树进行训练的目的是为了和接下来的随机森林作参照，看看使用随机森林之后相较于决策树提升到底有多大。训练后的模型f1_score分别为训练集0.85,测试集0.69，验证集0.70

使用随机森林训练

      rf = RandomForestClassifier(maxDepth = 5, minInstancesPerNode = 2)
      grid_rf = ParamGridBuilder()\
          .addGrid(rf.maxDepth, [4, 6, 8])\
          .addGrid(rf.minInstancesPerNode, [1,2])\
          .build()
      crossval_rf = CrossValidator(estimator = rf\
                                  ,estimatorParamMaps = grid_rf\
                                  ,evaluator = MulticlassClassificationEvaluator())
      crossval_rf_fit = crossval_rf.fit(trainingData)

      result_training_rf = crossval_rf_fit.transform(trainingData)
      result_test_rf = crossval_rf_fit.transform(testData)
      result_val_rf = crossval_rf_fit.transform(valData)

我利用网格搜索搜索了随机森林的maxdepth（最大深度）、minInstancesPerNode（划分后最小叶子数量）超参数。同样的原因我没有代入过多的超参数，计算出的f1_score分别为训练集0.92,测试集0.77，验证集0.65。随机森林相较于决策树提升还是非常大的，至于测试集和验证集的得分作为参考的意义更大一点，由于样本数量的限制不能作为最终的判定依据，只能作为参考。

## 总结

**模型的选择**
在本地环境下样本和计算能力都非常有限的情况下相对较好的模型是随机森林模型。随机森林训练集的得分还是远高于测试集和验证集，存在过拟合的现象，这个随机森林只有3棵树也许是导致这一现象的主要原因，还有测试样本还是太小了，导致训练结果收到分组的影响非常大。模型泛化能力上还是不足，在调参时模型还是从测试集中学习了过多的知识，当然，这些结论还有待在更大的数据集上进行验证。以及很明显的一点是逻辑回归模型的迭代次数还是太小了，因为训练集的得分也不理想。

**难点**
特征工程无疑是最费时的一项工作，在做特征工程的过程中需要考虑到的监督学习模型所需要的数据集，比如贝叶斯决策树等模型的理想数据和回归模型的理想数据是完全不同的，还需要考虑到如何提取数据更加能反映用户的真实使用情况和体验，数据是否是需要的，是否需要转换等等。特征工程还需要根据模型的表现及时的调整思路，以期得到更好的模型。如何选择模型的评价标准也是难点之一。

**反思及可能的改进方向**
我的特征工程中有些做法还有待商榷，比如将某些特征进行求和而另外的特征判断是或否，还有数据集中还有一些潜在的特征列我还没有挖掘出来。在算法的选择上为何选用这种算法的原因还不是非常明确，是否有其他更优的算法没有进行尝试，在算法调参上还有待完善，所以只能得出一些阶段性的成果。pyspark在本地运行时的速度问题也是一大制约因素，在处理中小型数据集时spark的优势得不到体现。
