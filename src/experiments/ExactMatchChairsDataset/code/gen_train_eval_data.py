import numpy as np
import csv

def gen_eval_shape_index():
    v1_models = [x.rstrip() for x in open('../filelist_chair_6777.txt', 'r')]
    v2_models = {}
    index = 0
    with open("../cleaned_all.csv", 'r') as f:
        f.readline()
        r = csv.reader(f)
        for row in r:
            v2_models[row[3]] = index
            index += 1

    v1_index = [v2_models[v1_m]  for v1_m in v1_models]
    np.savetxt("../v1_in_v2_index.txt", np.array(v1_index))

def gen_train_data():
    airplane_id, chair_id, car_id = "02691156", "03001627", "02958343"
    shape_index = {airplane_id:[], chair_id:[], car_id:[]}
    chair_rows = []
    with open("../cleaned_all.csv", 'r') as f:
        f.readline()
        r = csv.reader(f)
        i = 0
        for row in r:
            if row[1] in shape_index.keys():
                shape_index[row[1]].append(i)
            if row[1] == '03001627':
                chair_rows.append(row[3]+'\n')
            i += 1

    with open('../filelist_chair_6777.txt', 'r') as f:
        rows = set(f.readlines())
    print "chair_rows",chair_rows
    print rows
    result = list(rows.difference(set(chair_rows)))
    with open("chair_diff.txt", 'w') as f:
        f.writelines(result)


    print len(shape_index[airplane_id]), len(shape_index[chair_id]), len(shape_index[car_id])
    # all_shape_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    # all_img_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy")
    # index = []
    # for v in shape_index.values():
    #     index += v
    # np.savetxt("/home1/shangmingyang/data/ImgJoint3D/feature/train_index.txt", np.array(index))
    # np.savetxt("/home1/shangmingyang/data/ImgJoint3D/feature/train_shape_feature.txt", all_shape_features[np.array(index), :])
    # np.save("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature", all_img_features[np.array(index), :, :])


def gen_eval_shape_feature():
    gen_eval_shape_index()
    eval_index = np.loadtxt("../v1_in_v2_index.txt").astype(int)
    # all_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/shapenet55_nocolor.npy")
    all_features = np.load("/home1/shangmingyang/data/ImgJoint3D/feature/train_img_feature_all.npy")
    eval_features = all_features[eval_index, :]
    np.save("/home1/shangmingyang/data/ImgJoint3D/feature/eval_shape_img_feature.npy", eval_features)

if __name__ == '__main__':
    gen_eval_shape_feature()
    # gen_train_data()