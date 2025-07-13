"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from absl import flags
import go
import features as features_lib
import symmetries
import functools
import numpy as np
import random
import tensorflow as tf
import os

from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib.tpu.python.tpu import tpu_estimator as contrib_tpu_python_tpu_tpu_estimator

flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU')
flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')
flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')
flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')
flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')
flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')
flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')
flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')
flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')
flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')
flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')
flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')
flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')
flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')
flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')
flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')
flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')
flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')
flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')
flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')
flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')
flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')
flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')
flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))
flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')
flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')
flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')

FLAGS = flags.FLAGS

class DualNetwork():
    def __init__(self, save_file):
        self.save_file = save_file
        self.inference_input = None
        self.inference_output = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if FLAGS.gpu_device_list is not None:
            config.gpu_options.visible_device_list = FLAGS.gpu_device_list
        self.sess = tf.Session(graph=tf.Graph(), config=config)
        self.initialize_graph()

    def initialize_graph(self):
        with self.sess.graph.as_default():
            features, labels = get_inference_input()
            params = FLAGS.flag_values_dict()
            estimator_spec = model_fn(features, labels,
                                      tf.estimator.ModeKeys.PREDICT,
                                      params=params)
            self.inference_input = features
            self.inference_output = estimator_spec.predictions
            if self.save_file is not None:
                self.initialize_weights(self.save_file)
            else:
                self.sess.run(tf.global_variables_initializer())
    
    def initialize_weights(self, save_file):
        tf.train.Saver().restore(self.sess, save_file)
    
    def run(self, position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    def run_many(self, positions):
        f = get_features()
        processed = [features_lib.extract_features(p, f) for p in positions]
        if FLAGS.use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        outputs = self.sess.run(self.inference_output,
                                feed_dict={self.inference_input: processed})
        probabilities, value = outputs['policy_output'], outputs['value_output']
        if FLAGS.use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)
        return probabilities, value

def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)

def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)

def get_inference_input():
    feature_type = tf.bool if FLAGS.bool_features else tf.float32
    if FLAGS.input_layout == 'nhwc':
        feature_shape = [None, go.N, go.N, get_features_planes()]
    elif FLAGS.input_layout == 'nchw':
        feature_shape = [None, get_features_planes(), go.N, go.N]
    else:
        raise ValueError('invalid input_layout "%s"' % FLAGS.input_layout)
    return (tf.placeholder(feature_type, feature_shape, name='pos_tensor'),
            {'pi_tensor': tf.placeholder(tf.float32, [None, go.N * go.N + 1]),
             'value_tensor': tf.placeholder(tf.float32, [None])})

def model_fn(features, labels, mode, params):
    policy_output, value_output, logits = model_inference_fn(
        features, mode == tf.estimator.ModeKeys.TRAIN, params)

    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tf.stop_gradient(labels['pi_tensor'])))

    value_cost = params['value_cost_weight'] * tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))

    reg_vars = [v for v in tf.trainable_variables()
                if 'bias' not in v.name and 'beta' not in v.name]
    l2_cost = params['l2_strength'] * \
        tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

    combined_cost = policy_cost + value_cost + l2_cost

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.piecewise_constant(
        global_step, params['lr_boundaries'], params['lr_rates'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if params['quantize']:
        if mode == tf.estimator.ModeKeys.TRAIN:
            contrib_quantize.create_training_graph(
                quant_delay=params['quant_delay'])
        else:
            contrib_quantize.create_eval_graph()

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, params['sgd_momentum'])

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(combined_cost, global_step=global_step)

    # Computations to be executed on CPU, outside of the main TPU queues.
    def eval_metrics_host_call_fn(policy_output, value_output, pi_tensor,
                                  value_tensor, policy_cost, value_cost,
                                  l2_cost, combined_cost, step,
                                  est_mode=tf.estimator.ModeKeys.TRAIN):
        policy_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_output * tf.log(policy_output), axis=1))
        
        policy_target_top_1 = tf.argmax(pi_tensor, axis=1)

        policy_output_in_top1 = tf.to_float(
            tf.nn.in_top_k(policy_output, policy_target_top_1, k=1))
        policy_output_in_top3 = tf.to_float(
            tf.nn.in_top_k(policy_output, policy_target_top_1, k=3))

        policy_top_1_confidence = tf.reduce_max(policy_output, axis=1)
        policy_target_top_1_confidence = tf.boolean_mask(
            policy_output,
            tf.one_hot(policy_target_top_1, tf.shape(policy_output)[1]))

        value_cost_normalized = value_cost / params['value_cost_weight']
        avg_value_observed = tf.reduce_mean(value_tensor)

        with tf.variable_scope('metrics'):
            metric_ops = {
                'policy_cost': tf.metrics.mean(policy_cost),
                'value_cost': tf.metrics.mean(value_cost),
                'value_cost_normalized': tf.metrics.mean(value_cost_normalized),
                'l2_cost': tf.metrics.mean(l2_cost),
                'policy_entropy': tf.metrics.mean(policy_entropy),
                'combined_cost': tf.metrics.mean(combined_cost),
                'avg_value_observed': tf.metrics.mean(avg_value_observed),
                'policy_accuracy_top_1': tf.metrics.mean(policy_output_in_top1),
                'policy_accuracy_top_3': tf.metrics.mean(policy_output_in_top3),
                'policy_top_1_confidence': tf.metrics.mean(policy_top_1_confidence),
                'policy_target_top_1_confidence': tf.metrics.mean(
                    policy_target_top_1_confidence),
                'value_confidence': tf.metrics.mean(tf.abs(value_output)),
            }

        if est_mode == tf.estimator.ModeKeys.EVAL:
            return metric_ops

        eval_step = tf.reduce_min(step)

        summary_writer = contrib_summary.create_file_writer(FLAGS.work_dir)

        with summary_writer.as_default(), contrib_summary.record_summaries_every_n_global_steps(params['summary_steps'], eval_step):
            for metric_name, metric_op in metric_ops.items():
                contrib_summary.scalar(metric_name, metric_op[1], step=eval_step)

        reset_op = tf.variables_initializer(tf.local_variables('metrics'))
        cond_reset_op = tf.cond(
            tf.equal(eval_step % params['summary_steps'], tf.to_int64(1)),
            lambda: reset_op,
            lambda: tf.no_op())

        return contrib_summary.all_summary_ops() + [cond_reset_op]

    metric_args = [
        policy_output,
        value_output,
        labels['pi_tensor'],
        labels['value_tensor'],
        tf.reshape(policy_cost, [1]),
        tf.reshape(value_cost, [1]),
        tf.reshape(l2_cost, [1]),
        tf.reshape(combined_cost, [1]),
        tf.reshape(global_step, [1]),
    ]

    predictions = {
        'policy_output': policy_output,
        'value_output': value_output,
    }

    eval_metrics_only_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.EVAL)
    host_call_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.TRAIN)

    tpu_estimator_spec = contrib_tpu_python_tpu_tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=combined_cost,
        train_op=train_op,
        eval_metrics=(eval_metrics_only_fn, metric_args),
        host_call=(host_call_fn, metric_args)
    )
    return tpu_estimator_spec.as_estimator_spec()

def model_inference_fn(features, training, params):
    if FLAGS.bool_features:
        features = tf.dtypes.cast(features, dtype=tf.float32)

    if FLAGS.input_layout == 'nhwc':
        bn_axis = -1
        data_format = 'channels_last'
    else:
        bn_axis = 1
        data_format = 'channels_first'

    mg_batchn = functools.partial(
        tf.layers.batch_normalization,
        axis=bn_axis,
        momentum=.95,
        epsilon=1e-5,
        center=True,
        scale=True,
        fused=True,
        training=training)

    mg_conv2d = functools.partial(
        tf.layers.conv2d,
        filters=params['conv_width'],
        kernel_size=3,
        padding='same',
        use_bias=False,
        data_format=data_format)

    mg_global_avgpool2d = functools.partial(
        tf.layers.average_pooling2d,
        pool_size=go.N,
        strides=1,
        padding='valid',
        data_format=data_format)

    def mg_activation(inputs):
        if FLAGS.use_swish:
            return tf.nn.swish(inputs)
        return tf.nn.relu(inputs)

    def residual_inner(inputs):
        conv_layer1 = mg_batchn(mg_conv2d(inputs))
        initial_output = mg_activation(conv_layer1)
        conv_layer2 = mg_batchn(mg_conv2d(initial_output))
        return conv_layer2

    def mg_res_layer(inputs):
        residual = residual_inner(inputs)
        output = mg_activation(inputs + residual)
        return output

    def mg_squeeze_excitation_layer(inputs):
        channels = params['conv_width']
        ratio = FLAGS.SE_ratio
        assert channels % ratio == 0

        residual = residual_inner(inputs)
        pool = mg_global_avgpool2d(residual)
        fc1 = tf.layers.dense(pool, units=channels // ratio)
        squeeze = mg_activation(fc1)

        if FLAGS.use_SE_bias:
            fc2 = tf.layers.dense(squeeze, units=2*channels)
            gamma, bias = tf.split(fc2, 2, axis=3)
        else:
            gamma = tf.layers.dense(squeeze, units=channels)
            bias = 0

        sig = tf.nn.sigmoid(gamma)
        scale = tf.reshape(sig, [-1, 1, 1, channels])

        excitation = tf.multiply(scale, residual) + bias
        return mg_activation(inputs + excitation)

    initial_block = mg_activation(mg_batchn(mg_conv2d(features)))

    shared_output = initial_block
    for _ in range(params['trunk_layers']):
        if FLAGS.use_SE or FLAGS.use_SE_bias:
            shared_output = mg_squeeze_excitation_layer(shared_output)
        else:
            shared_output = mg_res_layer(shared_output)

    policy_conv = mg_conv2d(
        shared_output, filters=params['policy_conv_width'], kernel_size=1)
    policy_conv = mg_activation(
        mg_batchn(policy_conv, center=False, scale=False))

    logits = tf.layers.dense(
        tf.reshape(
            policy_conv, [-1, params['policy_conv_width'] * go.N * go.N]),
        go.N * go.N + 1)

    policy_output = tf.nn.softmax(logits, name='policy_output')

    value_conv = mg_conv2d(
        shared_output, filters=params['value_conv_width'], kernel_size=1)
    value_conv = mg_activation(
        mg_batchn(value_conv, center=False, scale=False))

    value_fc_hidden = mg_activation(tf.layers.dense(
        tf.reshape(value_conv, [-1, params['value_conv_width'] * go.N * go.N]),
        params['fc_width']))

    value_output = tf.nn.tanh(
        tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]),
        name='value_output')

    return policy_output, value_output, logits

def maybe_set_seed():
    if FLAGS.training_seed != 0:
        random.seed(FLAGS.training_seed)
        tf.set_random_seed(FLAGS.training_seed)
        np.random.seed(FLAGS.training_seed)

def get_estimator():
    return _get_nontpu_estimator()

def _get_nontpu_estimator():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_summary_steps=FLAGS.summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        session_config=session_config)
    return tf.estimator.Estimator(
        model_fn,
        model_dir=FLAGS.work_dir,
        config=run_config,
        params=FLAGS.flag_values_dict())

def bootstrap():
    maybe_set_seed()
    initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(FLAGS.work_dir, initial_checkpoint_name)
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        features, labels = get_inference_input()
        model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                 params=FLAGS.flag_values_dict())
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, save_file)

def export_model(model_path):
    estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.work_dir,
                                       params=FLAGS.flag_values_dict())
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        print('Copying {} to {}'.format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)

def freeze_graph(model_path, use_trt=False, trt_max_batch_size=8,
                 trt_precision='fp32'):
    output_names = ['policy_output', 'value_output']

    n = DualNetwork(model_path)
    out_graph = tf.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), output_names)

    if use_trt:
        import tensorflow.contrib.tensorrt as trt
        out_graph = trt.create_inference_graph(
            input_graph_def=out_graph,
            outputs=output_names,
            max_batch_size=trt_max_batch_size,
            max_workspace_size_bytes=1 << 29,
            precision_mode=trt_precision)

    metadata = make_model_metadata({
        'engine': 'tf',
        'use_trt': bool(use_trt),
    })

    minigo_model.write_graph_def(out_graph, metadata, model_path + '.minigo')

def make_model_metadata(metadata):
    for f in ['conv_width', 'fc_width', 'trunk_layers', 'use_SE', 'use_SE_bias',
              'use_swish', 'input_features', 'input_layout']:
        metadata[f] = getattr(FLAGS, f)
    metadata['input_type'] = 'bool' if FLAGS.bool_features else 'float'
    metadata['board_size'] = go.N
    return metadata