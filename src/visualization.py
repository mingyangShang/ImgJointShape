import matplotlib.pyplot as plt

class_ids = [4074963,2871439,3207941,2747177,2801938,3991062,3513137,3797390,\
                2773838,3624134,4004475,2691156,2818832,2876657,3593526,2946921,\
                2880940,2933112,3337140,3710193,4379243,2942699,2958343,3001627,\
                4256520,2828884,3691459,3046257,3085013,3211117,3948459,4090263,\
                3636649,3642806,3761084,2992529,4401088,3928116,4530566,4460130,\
                4099429,3790512,4225987,3467517,3261776,3759954,4330267,3325088,\
                4554684,4468005,2954340,2808440,3938244,2924116,2843684]

def generate_classes():
    id2class = {}
    with open("/home/shangmingyang/Desktop/ImgJointShape/shapenet55_class.txt", 'r') as f:
        for row in f.readlines():
            print row
            if row[0].isdigit():
                id, name = row.split('\t')
                full_name = name[:name.index('(')]
                if full_name.find(',') != -1:
                    first_name = full_name[:full_name.index(',')]
                else:
                    first_name = full_name
                id2class[id] = first_name

    classes = [id2class[str(id)] for id in class_ids]
    print classes
    # print id2class

def feature_reduction_vis(features, labels, tag="pca", fig_index=1, draw_label=True):
    print features.shape
    fig = plt.figure(fig_index)
    ax = fig.add_subplot(111)
    ax.set_title(tag)
    plt.xlabel('X')
    plt.ylabel('Y')
    # features = features.astype(int)
    # ax.scatter(features[:, 0], features[:, 1], c=labels, marker='.')
    ax.scatter(features[:, 0], features[:, 1], marker='.')
    if draw_label:
        for i in xrange(features.shape[0]):
            plt.text(features[i][0], features[i][1], str(labels[i]), horizontalalignment='center', verticalalignment='center')
    plt.legend("shapenet-shape")
    # plt.show()

def text_annotation_vis():

    plt.show()


if __name__ == '__main__':
    generate_classes()
    # text_annotation_vis()