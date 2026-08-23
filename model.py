class CDC(layers.Layer):
    def __init__(self, out_ch, kernel_size=3, strides=1, theta=THETA, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        self.vani = layers.Conv2D(
            out_ch, kernel_size=kernel_size, strides=strides, padding='same',
            use_bias=False, kernel_initializer=CONV_INIT,
        )

    def call(self, x):
        out_vanilla = self.vani(x)
        if self.theta == 0.0:
            return out_vanilla
        kernel_diff = tf.reduce_sum(self.vani.kernel, axis=(0, 1), keepdims=True)
        out_cd = tf.nn.conv2d(x, kernel_diff, strides=1, padding='SAME')
        return out_vanilla - self.theta * out_cd
    
test_cdc = CDC(8, theta=0.0)
test_input = tf.random.normal((1, 16, 16, 8))
out_cdc = test_cdc(test_input)
out_vani = test_cdc.vani(test_input)
tf.debugging.assert_near(out_cdc, out_vani, atol=1e-6)
print(f'theta=0 identity holds. Shape: {tuple(out_cdc.shape)}')

test_cdc_full = CDC(8, theta=1.0)
const_input = tf.ones((1, 16, 16, 8))
out_const = test_cdc_full(const_input)
interior = out_const[:, 1:-1, 1:-1, :]
max_interior = tf.reduce_max(tf.abs(interior)).numpy()
max_full = tf.reduce_max(tf.abs(out_const)).numpy()
print(f'theta=1, constant input — interior max|output| = {max_interior:.2e}  (expected ~0)')
print(f'                         full max|output|     = {max_full:.2e}  (boundary effects)')

class CDCBnRelu(layers.Layer):
    def __init__(self, out_ch, **kwargs):
        super().__init__(**kwargs)
        self.cdc = CDC(out_ch)
        self.bn = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON)
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.cdc(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class CDC2r(layers.Layer):
    def __init__(self, out_ch, r, **kwargs):
        super().__init__(**kwargs)
        mid = round(out_ch * r)
        self.block1 = CDCBnRelu(mid)
        self.block2 = CDCBnRelu(out_ch)

    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        return x

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            1, kernel_size=kernel_size, padding='same',
            use_bias=False, kernel_initializer=CONV_INIT,
        )

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        attn = tf.concat([avg, mx], axis=-1)
        attn = tf.sigmoid(self.conv(attn))
        return x * attn

def resize_to(x, target_hw):
    h = x.shape[1]
    if h == target_hw:
        return x
    pool = h // target_hw
    return tf.nn.avg_pool2d(x, pool, pool, 'VALID')

class MAFM(layers.Layer):
    def __init__(self, target_hw=32, k_low=7, k_mid=5, k_high=3, **kwargs):
        super().__init__(**kwargs)
        self.target_hw = target_hw
        self.attn_low = SpatialAttention(k_low)
        self.attn_mid = SpatialAttention(k_mid)
        self.attn_high = SpatialAttention(k_high)

    def call(self, low, mid, high):
        low_r = resize_to(self.attn_low(low), self.target_hw)
        mid_r = resize_to(self.attn_mid(mid), self.target_hw)
        high_r = resize_to(self.attn_high(high), self.target_hw)
        return tf.concat([low_r, mid_r, high_r], axis=-1)
class CDCN(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stem = CDCBnRelu(64)

        self.low = keras.Sequential([
            CDCBnRelu(128),
            CDCBnRelu(196),
            CDCBnRelu(128),
        ])
        self.mid = keras.Sequential([
            CDCBnRelu(128),
            CDCBnRelu(196),
            CDCBnRelu(128),
        ])
        self.high = keras.Sequential([
            CDCBnRelu(128),
            CDCBnRelu(196),
            CDCBnRelu(128),
        ])
        self.pool = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.head_a = CDCBnRelu(128)
        self.head_b = CDCBnRelu(64)
        self.head_out = CDC(1)

    def call(self, x, training=False, **kwargs):
        x = self.stem(x, training=training)
        low = self.pool(self.low(x, training=training))
        mid = self.pool(self.mid(low, training=training))
        high = self.pool(self.high(mid, training=training))

        fused = tf.concat([
            resize_to(low, 32),
            resize_to(mid, 32),
            high,
        ], axis=-1)
        x = self.head_a(fused, training=training)
        x = self.head_b(x, training=training)
        return self.head_out(x)
    
class CDCNpp(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.stem = keras.Sequential([
            CDCBnRelu(64),
            CDCBnRelu(128),
        ])

        self.low_cell = keras.Sequential([
            CDCBnRelu(128),
            CDCBnRelu(128),
        ])

        self.mid_cell = keras.Sequential([
            CDC2r(128, r=1.6),
            CDC2r(128, r=1.2),
            CDC2r(128, r=1.4),
        ])

        self.high_cell = keras.Sequential([
            CDCBnRelu(128),
            CDC2r(128, r=1.2),
        ])
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.mafm = MAFM(target_hw=32, k_low=7, k_mid=5, k_high=3)
        self.head_a = CDCBnRelu(128)
        self.head_out = CDC(1)

        # Auxiliary classification head
        self.cls_pool = layers.GlobalAveragePooling2D()
        self.cls_dense = layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.stem(x, training=training)
        low = self.pool(self.low_cell(x, training=training))
        mid = self.pool(self.mid_cell(low, training=training))
        high = self.pool(self.high_cell(mid, training=training))

        fused = self.mafm(low, mid, high)

        feat = self.head_a(fused, training=training)
        depth_out = self.head_out(feat)
        cls_out = self.cls_dense(self.cls_pool(feat))
        return {'depth': depth_out, 'cls': cls_out}
