import tensorflow as tf

# data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

for i in range(100):
    # learning_rate initialize
    learning_rate = 0.01

    # Gradient decent
    with tf.GradientTape() as tape:
        # hypothesis
        hypothesis = W * x_data + b
        # cost
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    # update W, b
    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(
            i, W.numpy(), b.numpy(), cost))
