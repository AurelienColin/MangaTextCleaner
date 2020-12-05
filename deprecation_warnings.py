import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def filter_warnings():
    import warnings
    # yeah, it's ugly, but the deprecation warnings are a massive pain
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation
    deprecation._PER_MODULE_WARNING_LIMIT = 0

    import tensorflow as tf
    tf.get_logger().setLevel(3)
