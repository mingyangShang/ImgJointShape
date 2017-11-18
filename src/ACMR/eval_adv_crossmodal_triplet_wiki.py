import tensorflow as tf
# from adv_crossmodal_triplet_wiki import AdvCrossModalSimple, ModelParams
from adv_crossmodal_similiar_wiki import AdvCrossModalSimple, ModelParams

def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()

    with graph.as_default():
        model = AdvCrossModalSimple(model_params)
    with tf.Session(graph=graph) as sess:
        model.eval(sess)

if __name__ == '__main__':
    tf.app.run()