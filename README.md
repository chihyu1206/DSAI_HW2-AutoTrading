# DSAI_HW2-AutoTrading
NCKU DSAI homework 2

[usage]
1. sudo apt install gcc g++ python3 python3-pip git (if necessary)
2. python3 -m pip install --upgrade pip (Update pip)
3. git clone https://github.com/chihyu1206/DSAI_HW2-AutoTrading.git (clone the repo)
4. cd DSAI_HW2-AutoTrading (change dir)
5. chmod u+x install.sh (After that, we can execute the shell script file)
6. python3 -m pip install -r requirements.txt (Python=3.6.9 Install the remaining module)
7. sudo ./install.sh (Install the ta-lib and Xgboost which can't be installed directly by pip in Ubuntu)
8. python3 app.py --training training_file --testing testing_file output output_file (Train and predict)

## Description
使用助教所提供IBM之前約1500天的股價數據，搭配XGBoost做multi-class分類，以預測未來20天的買賣點。

### Data Analysis
由於IBM是大型藍籌股 直覺認為股價波動幅度較小 難以直接單純使用提供的開高低收價來制定策略 因此引入常見的Python量化交易套件Ta-Lib 來擴增訓練資料

### Feature Engineering
因為作業說明有說之後會引入走勢相近的不同股票data來訓練，股價分布可能不同，而且Xgboost不像一般單純的決策樹只關心數據分布與機率，所以應該要做Normalization
```
# Do MinMax normalization
maxValue = train_df.to_numpy().max()
minValue = train_df.to_numpy().min()
diff = maxValue - minValue
train = train_df.transform(lambda x: (x - minValue) / diff)
```
接著使用Ta-Lib的函數 找了常見的幾個技術指標加入訓練資料中
```
# Use technical analysis to expand the data 
train["upperband"], train["middleband"], train["lowerband"] = BBANDS(train.close.to_numpy())
train["sar"] = SAR(train.high.to_numpy(), train.low.to_numpy())
train["rsi"] = RSI(train.close.to_numpy(), timeperiod=5)
train["slowk"], train["slowd"] = STOCH(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy())
train["ema"] = EMA(train.close.to_numpy(), timeperiod=5)
train["willr"] = WILLR(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy(), timeperiod=9)

#檢查數據了解到有些技術指標要過頭9天才能有輸出 後面都是完整的不影響Time Series順序 所以直接dropna 把前幾天NA的都移除)
train_data = train.dropna()
train_data = train_data.reset_index(drop=True)
```
### Trading Strategy
這裡我就直接以技術指標來當作交易的依據
總共有6個指標判斷 若是多頭訊號>4 看多(2)，<2則看空(0)，其餘(=2, 3, 4)則持平(1)
```
y = list()
for i in range(len(train_data)):
    isBull = (train_data["open"][i] > train_data["sar"][i], 
              train_data["open"][i] >= train_data["middleband"][i],
              train_data["rsi"][i] > 50,
              train_data["slowk"][i] >= train_data["slowd"][i],
              train_data["open"][i] >= train_data["ema"][i],
              train_data["willr"][i] > -50)
    if np.count_nonzero(isBull) > 4:
        y.append(2)
    elif np.count_nonzero(isBull) < 2:
        y.append(0)
    else:
        y.append(1)
y = np.array(y, dtype=np.int)
```

## Training
最後訓練的部分，是以前20天的價格技術分析等數據 預測20天後當天的買賣策略(e.g., 輸入1st~20nd 輸出40th的買賣策略)
有另外使用GridSearchCV函數，對XGBoost裡三個影響模型效果較大的參數找最佳的組合
```
parameters = {
    'max_depth': list(range(1, 10)),
    'min_child_weight': list(range(1, 10)),
    "n_estimators": list(range(100, 1001, 100))
}
gsearch = GridSearchCV(xgb, param_grid=parameters, scoring="f1", cv=2)
gsearch.fit(X_train, y_train,  eval_set=[(X_val, y_val)], eval_metric="auc", verbose=True)
best_parameters = gsearch.best_estimator_.get_params()
```
得到結果如下 
![Imgur](https://imgur.com/K9kuddT.jpg)
將其代入得到model

```
X_train, X_val, y_train, y_val = train_test_split(new_X, y, test_size=0.3, shuffle=False)
# Use XGBClassifier and mclogloss to do multi-class classification
xgb = XGBClassifier(learning_rate=0.1, 
                objective='multi:softmax',
                num_class=3,
                n_estimators=30, max_depth=3, min_child_weight=10, use_label_encoder=False)
```

得到預測結果(看空or持平or看多)後 搭配庫存需介於-1~1之間的條件 輸出買賣回測到output.csv
```
for i in range(1, len(preds)):
    # bullish
    if preds[i] == 2:
        if unit == 1:
            val = 0
        else:
            val = 1
            unit += 1
    # Do nothing
    elif preds[i] == 1:
        val = 0
    # bearish
    else:
        if unit == -1:
            val = 0
        else:
            val = -1
            unit -= 1
        
    ans.append(val)

# Write the result into output
with open(args.output, "w") as fp:
    for i in range(len(ans)):
        print(ans[i], file=fp)
```