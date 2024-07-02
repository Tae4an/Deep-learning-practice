# scripts/extract_vggish_features.py
import numpy as np
import tensorflow as tf
import sys
import vggish_input
import vggish_params
import vggish_slim
import vggish_postprocess

def extract_vggish_features(audio_path):
    # 오디오 파일을 VGGish 입력 형식으로 변환
    examples_batch = vggish_input.wavfile_to_examples(audio_path)

    # VGGish 모델 로드
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, '../models/vggish/vggish_model.ckpt')

        # VGGish 모델을 통해 특징 추출
        features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
        embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

    return embedding_batch
