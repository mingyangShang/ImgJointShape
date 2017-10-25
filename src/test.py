import csv

all_row = []

model_map = {}
with open("all.csv", 'r') as f:
    r = csv.reader(f)
    for row in r:
        model_map[str(row[0]).zfill(6)] = row
result = []
with open("v2_all.csv", 'r') as f:
    r = csv.reader(f)
    for row in r:
        if str(row[0]).zfill(6) in model_map:
            result.append(model_map[str(row[0]).zfill(6)])

with open("cleaned_all.csv", 'w') as f:
    w = csv.writer(f)
    w.writerows(result)


# for data_file in ["test.csv", "train.csv", "val.csv"]:
#     with open(data_file, 'r') as f:
#         first = True
#         for row in csv.reader(f):
#             if first:
#                 header = row
#                 first = False
#             else:
#                 all_row.append(row)
#
# with open('v2_all.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerows([header] + all_row)