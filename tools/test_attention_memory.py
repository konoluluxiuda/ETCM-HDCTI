#!/usr/bin/env python3
"""Probe GPU memory needed by HDCTI-style full self-attention.

This script intentionally does not import the project model. It isolates the
attention pattern that caused the ETCM2.0_core OOM:

    logits = matmul(Q, K^T)       # [nodes, nodes]
    weights = softmax(logits)
    out = matmul(weights, V)

Each tested node count runs in a separate subprocess so an OOM at one size does
not poison the TensorFlow session for the next size.
"""

import argparse
import multiprocessing as mp
import os
import sys
import traceback


def _run_case(nodes, dim, mode, device, heads, layers, backward, queue):
    try:
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if mode == "softmax":
            x = tf.random.normal([nodes, nodes], dtype=tf.float32)
            y = tf.nn.softmax(x, axis=-1)
            fetch = tf.reduce_sum(y)
            shape = [nodes, nodes]
        elif mode == "attention":
            emb_dim = dim * heads
            embeddings = tf.Variable(
                tf.random.normal([nodes, emb_dim], dtype=tf.float32),
                name="embeddings",
            )

            current = embeddings
            for layer in range(layers):
                outputs = []
                for head in range(heads):
                    q_weight = tf.Variable(
                        tf.random.normal([emb_dim, dim], stddev=0.02),
                        name="q_%d_%d" % (layer, head),
                    )
                    k_weight = tf.Variable(
                        tf.random.normal([emb_dim, dim], stddev=0.02),
                        name="k_%d_%d" % (layer, head),
                    )
                    v_weight = tf.Variable(
                        tf.random.normal([emb_dim, dim], stddev=0.02),
                        name="v_%d_%d" % (layer, head),
                    )
                    q = tf.matmul(current, q_weight)
                    k = tf.matmul(current, k_weight)
                    v = tf.matmul(current, v_weight)
                    logits = tf.matmul(q, k, transpose_b=True)
                    weights = tf.nn.softmax(logits / tf.sqrt(float(dim)), axis=-1)
                    outputs.append(tf.matmul(weights, v))
                current = tf.concat(outputs, axis=-1)

            loss = tf.reduce_mean(tf.square(current))
            if backward:
                train = tf.train.AdamOptimizer(0.001).minimize(loss)
                fetch = [train, loss]
            else:
                fetch = loss
            shape = [nodes, nodes]
        else:
            raise ValueError("unsupported mode: %s" % mode)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            value = sess.run(fetch)
            if backward:
                value = value[1]

        queue.put({
            "ok": True,
            "nodes": nodes,
            "shape": shape,
            "value": float(value),
        })
    except Exception as exc:
        queue.put({
            "ok": False,
            "nodes": nodes,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _estimate_matrix_gib(nodes):
    return nodes * nodes * 4 / (1024.0 ** 3)


def main():
    parser = argparse.ArgumentParser(
        description="Test memory pressure from HDCTI-style full self-attention."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[19242],
        help="Node counts to test. Default: 19242, the ETCM2.0_core compound count.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Per-head attention dimension. Default: 32 for 64 dims / 2 heads.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=1,
        help="Number of attention heads to build. Use 2 to mimic HDCTI.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of stacked attention layers. Use 2 to mimic HDCTI.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run one Adam training step, including gradients and optimizer state.",
    )
    parser.add_argument(
        "--mode",
        choices=["attention", "softmax"],
        default="attention",
        help="attention runs QK^T -> softmax -> AV; softmax only allocates [N,N].",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to test. Default: gpu.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional per-case timeout in seconds. 0 means no timeout.",
    )
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    failures = 0

    print("HDCTI attention memory probe")
    print("mode=%s device=%s dim=%d heads=%d layers=%d backward=%s" %
          (args.mode, args.device, args.dim, args.heads, args.layers, args.backward))
    print("One [nodes,nodes] float32 matrix estimates only the lower bound.")
    print()

    for nodes in args.nodes:
        lower_bound = _estimate_matrix_gib(nodes)
        print("===== nodes=%d lower_bound_matrix=%.2f GiB =====" % (nodes, lower_bound))

        queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_case,
            args=(nodes, args.dim, args.mode, args.device, args.heads, args.layers,
                  args.backward, queue),
        )
        proc.start()
        proc.join(args.timeout if args.timeout > 0 else None)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            failures += 1
            print("TIMEOUT nodes=%d after %ds" % (nodes, args.timeout))
            continue

        if queue.empty():
            failures += 1
            print("FAILED nodes=%d process_exit_code=%s no result returned" %
                  (nodes, proc.exitcode))
            continue

        result = queue.get()
        if result["ok"]:
            print("OK nodes=%d shape=%s result=%s" %
                  (result["nodes"], result["shape"], result["value"]))
        else:
            failures += 1
            print("OOM/FAILED nodes=%d error_type=%s" %
                  (result["nodes"], result["error_type"]))
            print(result["error"])
            if "ResourceExhausted" not in result["error_type"] and "OOM" not in result["error"]:
                print(result["traceback"])

        print()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
