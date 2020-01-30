### TEST MAXE ### Expected 11
y_true = [0,1,2,3,4,5,6,7,8,9,10]
y_pred = [11,12,13,14,15,0,1,4,3,2,1]
max_error(y_true, y_pred)

### TEST MEAN square ERROR ### -> Expected 76.45
y_true = [0,1,2,3,4,5,6,7,8,9,10]
y_pred = [11,12,13,14,15,0,1,4,3,2,1]
mean_squared_error(y_true, y_pred)

### TEST MEDIAN ABSOLUTE ERROR ### -> Expected 9
y_true = [0,1,2,3,4,5,6,7,8,9,10]
y_pred = [11,12,13,14,15,0,1,4,3,2,1]
median_absolute_error(y_true, y_pred)

### TEST MEAN ABSOLUTE ERROR ### -> Expected 8.09
y_true = [0,1,2,3,4,5,6,7,8,9,10]
y_pred = [11,12,13,14,15,0,1,4,3,2,1]
mean_absolute_error(y_true, y_pred)

### TEST R2 ERROR ### -> Expected
y_true = [0,1,2,3,4,5,6,7,8,9,10]
y_pred = [11,12,13,14,15,0,1,4,3,2,1]
r2_score(y_true, y_pred)