import paddle
import time
import numpy as np
import paddle.nn as nn
from paddlenlp.utils.log import logger
from collections import defaultdict
@paddle.no_grad()
def evaluate(model, metric, data_loader, memories0):
    model.eval()
    losses = []
    # copy the memory
    memories = list(memories0)
    tic_train = time.time()
    eval_logging_step = 500

    probs_dict = defaultdict(list)
    label_dict = dict()
    global_steps = 0
    for step, batch in enumerate(data_loader, start=1):
        input_ids, position_ids, token_type_ids, attn_mask, labels, qids, \
            gather_idxs, need_cal_loss = batch
        logits, memories = model(input_ids, memories, token_type_ids,
                                 position_ids, attn_mask)
        logits, labels, qids = list(
            map(lambda x: paddle.gather(x, gather_idxs),
                [logits, labels, qids]))
        # Need to collect probs for each qid, so use softmax_with_cross_entropy
        loss, probs = nn.functional.softmax_with_cross_entropy(
            logits, labels, return_softmax=True)
        losses.append(loss.mean().numpy())
        # Shape: [B, NUM_LABELS]
        np_probs = probs.numpy()
        # Shape: [B, 1]
        np_qids = qids.numpy()
        np_labels = labels.numpy().flatten()
        for i, qid in enumerate(np_qids.flatten()):
            probs_dict[qid].append(np_probs[i])
            label_dict[qid] = np_labels[i]  # Same qid share same label.

        if step % eval_logging_step == 0:
            logger.info("Step %d: loss:  %.5f, speed: %.5f steps/s" %
                        (step, np.mean(losses),
                         eval_logging_step / (time.time() - tic_train)))
            tic_train = time.time()

    # Collect predicted labels
    preds = []
    labels = []
    for qid, probs in probs_dict.items():
        mean_prob = np.mean(np.array(probs), axis=0)
        preds.append(mean_prob)
        labels.append(label_dict[qid])

    preds = paddle.to_tensor(np.array(preds, dtype='float32'))
    labels = paddle.to_tensor(np.array(labels, dtype='int64'))

    metric.update(metric.compute(preds, labels))
    acc_or_f1 = metric.accumulate()
    logger.info("Eval loss: %.5f, %s: %.5f" %
                (np.mean(losses), metric.__class__.__name__, acc_or_f1))
    metric.reset()
    model.train()
    return acc_or_f1