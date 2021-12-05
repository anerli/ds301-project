# def calculate_returns(model, df):
#     pass
import numpy as np


'''
y_pred: Predicted price
y_actual: Actual price

Calculates returns given predictions by using the prediction for the next day to make a decision whether to 
buy or sell on each day.
'''
def calculate_return(y_pred: list, y_actual: list, sell_short=True):
    # y_pred = y_pred.reshape((-1,)).tolist()
    # y_actual = y_actual.reshape((-1,)).tolist()

    assert len(y_pred) == len(y_actual)

    cumulative_return = 1.0

    for i in range(1, len(y_pred)):
        last_price = y_actual[i-1]
        pred_price = y_pred[i]
        actual_price = y_actual[i]

        price_change = (actual_price / last_price) - 1

        if pred_price > last_price:
            # "buy"
            #ret = (price_change + actual_price) / actual_price
            cumulative_return *= (1 + price_change)
        else:
            # "sell" (or sell short)
            if sell_short:
                cumulative_return *= (1 - price_change)
                
    return cumulative_return


