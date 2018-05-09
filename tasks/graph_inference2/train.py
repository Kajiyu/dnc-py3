import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import time
import sys
import os
import json

from dnc.dnc import DNC
from recurrent_controller import RecurrentController
from generate_data import generate_data


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def binary_cross_entropy(predictions, targets):
    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')
    mem_logs_dir = os.path.join(dirname, 'mem_logs/')
    
    batch_size = 1
    input_size = 90
    output_size = 90
    sequence_max_length = 420
    words_count = 500
    word_size = 50
    read_heads = 4

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = 200000

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    graph = tf.Graph()

    with open("./json/metro_training_data.json", "r") as f:
        data_dict = json.load(f)
    edges = data_dict["edge"]
    metro_graph = data_dict["graph"]

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

            ncomputer = DNC(
                RecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            output, memory_views = ncomputer.get_outputs()
            loss = None
            for _k in range(9):
                tmp_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=output[:,:,_k*10:(_k+1)*10],
                        labels=ncomputer.target_output[:,:,_k*10:(_k+1)*10],
                        name="categorical_loss_"+str(_k+1)
                    )
                )
                if loss is None:
                    loss = tmp_loss
                else:
                    loss = loss + tmp_loss
            loss = loss / 9.0
            # print(loss)

            summeries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    summeries.append(tf.summary.histogram(var.name + '/grad', grad))
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.summary.scalar("Loss", loss))

            summerize_op = tf.summary.merge(summeries)
            no_summerize = tf.no_op()

            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            for i in range(iterations + 1):
                llprint("\rIteration %d/%d" % (i, iterations))
                input_data, target_output = generate_data(batch_size, np.array(edges), metro_graph)

                summerize = (i % 5 == 0)
                mem_summarize = (i % 1000 == 0)
                take_checkpoint = (i != 0) and (i % 1000 == 0)

                loss_value, _, mem_views_values, summary = session.run([
                    loss,
                    apply_gradients,
                    memory_views,
                    summerize_op if summerize else no_summerize
                ], feed_dict={
                    ncomputer.input_data: np.array(input_data),
                    ncomputer.target_output: np.array(target_output),
                    ncomputer.sequence_length: np.array(input_data).shape[1]
                })

                last_100_losses.append(loss_value)
                if summary:
                    summerizer.add_summary(summary, i)
                for key in mem_views_values:
                    mem_views_values[key] = mem_views_values[key].tolist()

                if summerize:
                    llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
                    last_100_losses = []
                

                if mem_summarize:
                    with open(mem_logs_dir+"/"+str(i)+".json", "w") as f:
                        json.dump(mem_views_values, f, ensure_ascii=False, separators=(',', ':'))

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'task1-3-step-%d' % (i))
                    llprint("Done!\n")