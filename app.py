import pandas as pd
import numpy as np
from talib import BBANDS, SAR, RSI, STOCH
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
    
    # Read the training and testing data
    train_df = pd.read_csv(args.training, names=("open", "high", "low", "close"))
    test_df = pd.read_csv(args.testing, names=("open", "high", "low", "close"))

    # Do MinMax normalization
    maxValue = train_df.to_numpy().max()
    minValue = train_df.to_numpy().min()
    diff = maxValue - minValue
    train = train_df.transform(lambda x: (x - minValue) / diff)
    test = test_df.transform(lambda x: (x - minValue) / diff)
    
    # Use technical analysis to expand the data 
    train["upperband"], train["middleband"], train["lowerband"] = BBANDS(train.close.to_numpy())
    train["sar"] = SAR(train.high.to_numpy(), train.low.to_numpy())
    train["rsi"] = RSI(train.close.to_numpy(), timeperiod=5)
    train["slowk"], train["slowd"] = STOCH(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy())
    train_data = train.dropna()
    
    test["upperband"], test["middleband"], test["lowerband"] = BBANDS(test.close.to_numpy())
    test["sar"] = SAR(test.high.to_numpy(), test.low.to_numpy())
    test["rsi"] = RSI(test.close.to_numpy(), timeperiod=5)
    test["slowk"], test["slowd"] = STOCH(test.high.to_numpy(), test.low.to_numpy(), test.close.to_numpy())

    # 1/0 on behalf of the open price is higher or lower after 3 days
    train_data["threeDays"] = np.where(train_data.open.shift(-3) > train_data.open, 1, 0)
    
    # We can't judge the correction of last three days when training 
    train = train_data.drop(train_data.tail(3).index, inplace=False)
    y = train.threeDays.to_numpy()
    X = train.drop("threeDays", axis=1).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Use XGBClassifier and AUC to do binary classification
    xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=9, use_label_encoder=False)
    
    model = xgb.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                verbose=False)
    
    # Predict the testing data
    preds = model.predict(test.values)
    ans = []
    unit = 0
    val = 0
    # The first 8 days doesn't have Stochastic Oscillator，KD
    # So we can just predict the price based on our model
    for i in range(1, 8):
        _sum = sum(preds[i-1:i+1])
        # bullish
        if _sum == 2:
            if unit == 1:
                val = 0
            else:
                val = 1
                unit += 1
        # Do nothing
        elif _sum == 1:
            val = 0
        # bearish
        else:
            if unit == -1:
                val = 0
            else:
                val = -1
                unit -= 1
        
        ans.append(val)
    # Draw technical signs into our prediction
    for i in range(8, len(preds)):
        isBull = (test["open"][i] > test["sar"][i], 
                  test["open"][i] >= test["middleband"][i],
                  test["rsi"][i] > 50,
                  test["slowk"][i] >= test["slowd"][i])
        # We are bullish that the stock price will be higher after 3 days
        # and the Technical Signs at once back the prediction 
        if preds[i] == 1 and np.sum(isBull != 0) >= 2:
            if unit == 1:
                val = 0
            else:
                val = 1
                unit += 1
        # We are bearish that the stock price will be higher after 3 days
        # and the Technical Signs at once back the prediction 
        elif preds[i] == 0 and np.sum(isBull != 0) <= 2:
            if unit == -1:
                val = 0
            else:
                val = -1
                unit -= 1
        # Do nothing
        else:
            val = 0
        ans.append(val)

    # Write the result into output
    with open(args.output, "w") as fp:
        for i in range(len(ans)):
            print(ans[i], file=fp)