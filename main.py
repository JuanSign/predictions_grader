import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

# global variables
test_data        : pd.DataFrame                   = None
predictions      : List[Tuple[str, pd.DataFrame]] = []
time             : str                            = str(datetime.now()).split('.')[0]

# check if test_data is provided (and singular)
for items in os.walk("test_data"):
    if(len(items[2]) != 1):
        raise FileNotFoundError("Please provide a SINGLE test data (.csv)!")
    test_data = pd.read_csv(os.path.join(items[0], items[2][0]))
    if 'id' not in test_data.columns:
        raise AttributeError("WRONG TEST DATA FORMAT: missing column 'id'.")
    if 'label' not in test_data.columns:
        raise AttributeError("WRONG TEST DATA FORMAT: missing column 'label'.")
    if test_data['label'].dtype != 'int64' or test_data['label'].nunique() != 2:
        raise ValueError("TEST DATA LABEL IS NOT BOOLEAN")

# load all predictions
for items in os.walk("predictions"):
    if(len(items[2]) == 0):
        raise FileNotFoundError("Please provide AT LEAST ONE prediction!")
    for files in items[2]:
        file_name = files.split('.')[0]
        current_prediction = pd.read_csv(os.path.join(items[0], files))
        if 'id' not in current_prediction.columns:
            print(f"WARN: SKIPPING [{file_name}] PREDICTION")
            print(f"missing column 'id'.")
            continue
        if 'label' not in current_prediction.columns:
            print(f"WARN: SKIPPING [{file_name}] PREDICTION")
            print(f"missing column 'label'.") 
            continue
        if current_prediction['label'].dtype != 'int64' or current_prediction['label'].nunique() != 2:
            print(f"WARN: SKIPPING [{file_name}] PREDICTION")
            print(f"'label' is not boolean.") 
            continue
        predictions.append((file_name, current_prediction))

# grade prediction(s)
for prediction in predictions:
    # comparing dimension
    if len(test_data) != len(prediction[1]):
            print(f"WARN: NOT GRADING [{prediction[0]}]")
            print("Different Number of Rows.")
            print(f"TEST DATA ROWS: {len(test_data)}\n")
            print(f"PREDICTION ROWS: {len(prediction[1])}\n")
            continue

    # create directory
    os.makedirs(f'results/{prediction[0]}_RESULT', exist_ok=True)
    # prediction data
    correct        : List[int] = []
    missmatch      : List[int] = []
    false_negative : List[int] = []
    false_positive : List[int] = []
    # grading
    for index in range(len(test_data)):
        pred_id = prediction[1].iloc[index]['id']
        test_value = test_data.iloc[index]['label']
        pred_value = prediction[1].iloc[index]['label']
        if(test_value == pred_value): correct.append(pred_id)
        else:
            missmatch.append(pred_id)
            if(test_value == 1 and pred_value == 0):
                false_negative.append(pred_id)
            else:
                false_positive.append(pred_id)

    # summary
    with open(f"results/{prediction[0]}_RESULT/Summary.txt", "w") as summary:
        summary.write(f"SUMMARY: [{prediction[0]}]\n")
        summary.write(f"TIMESTAMP: {time}\n\n")
        summary.write(f"CORRECT: {len(correct)/len(test_data)*100:.2f}%\n")
        summary.write(f"FALSE NEGATIVE: {len(false_negative)/len(test_data)*100:.2f}%\n")
        summary.write(f"FALSE POSITIVE: {len(false_positive)/len(test_data)*100:.2f}%\n")

    with open(f"results/{prediction[0]}_RESULT/Correct_ID.csv", "w") as c:
        c.write("correct\n")
        for i in correct:
            c.write(f"{i}\n")
    with open(f"results/{prediction[0]}_RESULT/FN_ID.csv", "w") as fn:
        fn.write("false_negative\n")
        for i in false_negative:
            fn.write(f"{i}\n")
    with open(f"results/{prediction[0]}_RESULT/FP_ID.csv", "w") as fp:
        fp.write("false_positive\n")
        for i in false_positive:
            fp.write(f"{i}\n")