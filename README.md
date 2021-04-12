# AI_homework2
---------------------預測手法----------------------

我是用前十四天來預測下一天股票，並且依照預測出來的值判斷是否漲跌

假設原本我有一支股票，然後我預測出明天會漲，那麼下一天就輸出-1

如果預測明天會跌，那就輸出0，直到股票漲則賣出

反之賣空股票則相反

另外若我手上沒有持股的話，則會random(買,賣空)來決定初始狀態，再重複上面的循環

algo概念如下      

  if (沒有持股):

    staus=random(買,賣空)

  esle: //有持股

    if staus==買:

      predict(使用前14天資料)

      if(predict==漲)

        賣出

      else:

        hold

    else: //staus==賣

      predict(使用前14天資料)

      if(predict==跌)

        買入

      else:

        賣

其中比較細節的地方是，做漲跌判斷的時候，是拿昨天預測出來的資料，以及今天預測出來的資料進行判斷

而不是拿昨天""真實""的資料跟今天的預測做判斷

原因如下

網路並不能完美的預測出真實的值，在val上誤差平均在1左右

所以預測出來的解空間，比較像是把真實的解空間加上一點平移以及一點noise，

所以拿昨天""真實""的資料跟今天的預測做判斷，是不合理的，因為兩者的解空間是不一樣的

但是預測出來的解空間還是能帶有真實解空間的Time series，所以拿昨天""預測""出來的資料跟今天預測出來的資料進行判斷，比較能更好的模擬真實的解空間的漲跌



---------------------程式架構----------------------

我所使用的是pytorch來完成的，並且使用的類神經網路為LSTM，並且輸入資料包含Open,High,Low,Close


以下我針對我的code做一些講解

資料再輸入進網路前，我都有分別對Open,High,Low,Close做normalized(以下為所有train的mean及std)

![image](https://github.com/qw61116111/AI_homework2/blob/main/image/nor.jpg)

訓練時我有先拆9:1的train及val用MSE來驗證網路的收斂效果以及是否過擬和

直到可以正確收斂以及解決嚴重過擬和問題後，再把val的資料丟回去trainset裡面一起訓練，以保證testing上的Time series的連續性


接下來我網路的參數以及架構

![image](https://github.com/qw61116111/AI_homework2/blob/main/image/para.jpg)

![image](https://github.com/qw61116111/AI_homework2/blob/main/image/lstm.jpg)


預測test的第一天是用train的最後14天的資料預測

預測test的第二天是用train的最後13天的資料預測+Test第一天資料預測

預測test的第三天是用train的最後12天的資料預測+Test第一天資料預測+Test第二天資料預測

以此類推，以下為程式碼

![image](https://github.com/qw61116111/AI_homework2/blob/main/image/test.jpg)
