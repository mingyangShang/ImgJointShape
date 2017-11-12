import csv

all_row = []

model_map = {}
with open("all.csv", 'r') as f:
    r = csv.reader(f)
    for row in r:
        model_map[row[3]] = row

with open('shape_list_03001627_exact_matches.txt', 'r') as f:
    all_chair = f.readlines()
    for row in all_chair:
        class_id, model_id = row.split()
        if model_id not in model_map:
            print("%s not in v2"%model_id)
            continue
        if model_map[model_id][1] != '03001627':
            print("model %s do not match class %s"%(model_id, model_map[model_id][1]))
# result = []
# with open("v2_all.csv", 'r') as f:
#     r = csv.reader(f)
#     for row in r:
#         if str(row[0]).zfill(6) in model_map:
#             result.append(model_map[str(row[0]).zfill(6)])
#
# with open("cleaned_all.csv", 'w') as f:
#     w = csv.writer(f)
#     w.writerows(result)


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