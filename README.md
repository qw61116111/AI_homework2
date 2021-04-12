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
    
    if(predict漲)
    
      賣出
      
    else:
    
      hold
      
  else: //staus==賣
  
    predict(使用前14天資料)
    
    if(predict跌)
    
      買入
      
    else:
    
      賣


---------------------程式架構----------------------

我所使用的是pytorch來完成的，並且使用的類神經網路為LSTM，並且輸入資料包含Open,High,Low,Close

並且我是用前十四天預測來預測下一天

以下我針對我的code做一些講解

每張圖在進行訓練的時候，我都有先做normalized的處理，並且maen跟std是之前就先用numpy算好並寫在那邊的

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/mean.jpg)

接下來是我的dataset的處裡
遞一部分我把圖都讀進來，並且用for迴圈對每張圖進行減平均除標準差的normalized

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/dataset.jpg)


接下來我網路的參數以及架構

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/para.jpg)

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/lstm.jpg)


最後開始預測3/23到3/29的方式是，先用3/16到3/22預測3/23，
再把資料往明天做平移，變成用3/17到3/23去預測3/24，其中3/23的資料是用的是我預測的值
以下為平移的code

![image](https://github.com/qw61116111/AI_homework1/blob/main/image/out.jpg)
