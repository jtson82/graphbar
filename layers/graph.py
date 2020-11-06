import tensorflow as tf

class GraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            name="graph_conv"):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                    name='w',
                    shape=(self.input_dim, self.output_dim),
                    initializer=tf.glorot_uniform_initializer())
            
            if self.use_bias:
                self.b = tf.get_variable(
                        name='b',
                        initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x):
        x = tf.reshape(x, [-1, self.input_dim])
        x = tf.matmul(x, self.w)

        x = tf.reshape(x, [-1, 200, self.output_dim])
        x = tf.matmul(adj_norm, x)

        if self.use_bias:
            x = tf.reshape(x, [-1, self.output_dim])
            x = tf.add(x, self.b)
            x = tf.reshape(x, [-1, 200, self.output_dim])

        if self.activation is not None:
            x = self.activation(x)

        return x
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GraphGather:
    def __init__(self, **kwargs):
        super(GraphGather, self).__init__(**kwargs)

    def call(self, x):
        x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.squeeze(x, axis=1)
        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class GlobalMaxPooling:
    def __init__(self, **kwargs):
        super(GlobalMaxPooling, self).__init__(**kwargs)

    def call(self, x):
        x = tf.math.reduce_max(x, axis=1, keepdims=True)
        x = tf.squeeze(x, axis=1)
        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
