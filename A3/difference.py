import csv

data_old = []
data_new = []

with open('submission.csv') as f1:
    data = csv.reader(f1, delimiter=",")
    next(data)
    for row in data:
        data_old.append(row)
f1.close()

with open('submission5.csv') as f2:
    data = csv.reader(f2, delimiter=",")
    for row in data:
        data_new.append(row)
f2.close()

print(data_old[:3])
print(data_new[:3])

# x = 0
# for i in range(len(data_old)):
#     if data_new[i] == data_old[i]:
#         x += 1

# print(x)


