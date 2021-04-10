import pandas as pd
import numpy as np
from talib import BBANDS, SAR, RSI, STOCH, EMA, WILLR
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
# You can write code above the if-main block.

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # Read the training data
    train_df = pd.read_csv(args.training, names=("open", "high", "low", "close"))
    test_df = pd.read_csv(args.testing, names=("open", "high", "low", "close"))
    # Do MinMax normalization
    maxValue = train_df.to_numpy().max()
    minValue = train_df.to_numpy().min()
    diff = maxValue - minValue
    train = train_df.transform(lambda x: (x - minValue) / diff)
    # test = test_df.transform(lambda x: (x - minValue) / diff)
    
    # Use technical analysis to expand the data 
    train["upperband"], train["middleband"], train["lowerband"] = BBANDS(train.close.to_numpy())
    train["sar"] = SAR(train.high.to_numpy(), train.low.to_numpy())
    train["rsi"] = RSI(train.close.to_numpy(), timeperiod=5)
    train["slowk"], train["slowd"] = STOCH(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy())
    train["ema"] = EMA(train.close.to_numpy(), timeperiod=5)
    train["willr"] = WILLR(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy(), timeperiod=5)

    train_data = train.dropna()
    train_data = train_data.reset_index(drop=True)
    
    """
    test["upperband"], test["middleband"], test["lowerband"] = BBANDS(test.close.to_numpy())
    test["sar"] = SAR(test.high.to_numpy(), test.low.to_numpy())
    test["rsi"] = RSI(test.close.to_numpy(), timeperiod=5)
    test["slowk"], test["slowd"] = STOCH(test.high.to_numpy(), test.low.to_numpy(), test.close.to_numpy())
    """
    # 2->bullish, 0->bearish, 1->do nothing
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
    test_size = len(test_df)
    X = list()
    for i in range(20, len(train_data) + test_size - 20):
        X.append(train_data.loc[i-20:i-1, :].values)
    X = np.array(X)
    
    y = y[40:]
   
    test = X[-test_size:]
    new_X = X[:-test_size]
    new_X = new_X.reshape((new_X.shape[0], -1))
    X_train, X_val, y_train, y_val = train_test_split(new_X, y, test_size=0.3, shuffle=False)
    # Use XGBClassifier and mclogloss to do multi-class classification
    xgb = XGBClassifier(learning_rate=0.1, 
                    objective='multi:softmax',
                    num_class=3,
                    n_estimators=30, max_depth=3, min_child_weight=10, use_label_encoder=False)

    
    model = xgb.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mlogloss",
                verbose=True)
    
    # Predict the testing data
    preds = model.predict(test.reshape(test_size, -1))
    ans = []
    unit = 0
    val = 0

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