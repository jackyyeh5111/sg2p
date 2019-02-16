import tensorflow as tf

indices = tf.constant([[0], [0]])
updates = tf.constant([[[5, 5, 5, 5], 
						[6, 6, 6, 6],
                        [7, 7, 7, 7]],
                       [[5, 5, 5, 5], 
                       	[6, 6, 6, 6],
                        [7, 7, 7, 7]]])
shape = tf.constant([4, 3, 4])

print indices # (B, 1)
print updates # (B, 4, 4)
print shape # (3, )

scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
  print(sess.run(scatter))