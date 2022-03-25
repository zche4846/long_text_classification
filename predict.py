import os
import paddle
from paddlenlp.utils.env import PPNLP_HOME
from paddlenlp.utils.log import logger
from paddlenlp.taskflow.utils import dygraph_mode_guard, static_mode_guard
from model.modeling import ErnieDocForSequenceClassification
from paddlenlp.transformers import ErnieDocTokenizer
from paddlenlp.datasets import load_dataset
from data import ClassifierIterator, to_json_file
import paddle.nn as nn

def predict(model,
            test_dataloader,
            file_path,
            memories,
            label_list,
            static_mode,
            input_handles=None,
            output_handles=None):

    if static_mode and ((input_handles is None) or (output_handles is None)):
        raise ValueError("Static model inference requires specified input_handles and output_handles")

    label_dict = dict()
    if not static_mode:
        model.eval()
        for _, batch in enumerate(test_dataloader, start=1):
            input_ids, position_ids, token_type_ids, attn_mask, _, qids, \
            gather_idxs, need_cal_loss = batch
            logits, memories = model(input_ids, memories, token_type_ids,
                                     position_ids, attn_mask)
            logits, qids = list(
                map(lambda x: paddle.gather(x, gather_idxs),
                    [logits, qids]))
            probs = nn.functional.softmax(logits, axis=1)
            idx = paddle.argmax(probs, axis=1).numpy()
            idx = idx.tolist()
            labels = [label_list[i] for i in idx]
            for i, qid in enumerate(qids.numpy().flatten()):
                label_dict[str(qid)] = labels[i]
    else:
        for _, batch in enumerate(test_dataloader, start=1):
            input_ids, position_ids, token_type_ids, attn_mask, _, qids, \
            gather_idxs, need_cal_loss = batch
            input_handles[0].copy_from_cpu(input_ids.numpy())
            input_handles[1].copy_from_cpu(paddle.to_tensor(memories).numpy())
            input_handles[2].copy_from_cpu(token_type_ids.numpy())
            input_handles[3].copy_from_cpu(position_ids.numpy())
            input_handles[4].copy_from_cpu(attn_mask.numpy())
            model.run()
            logits = paddle.to_tensor(output_handles[0].copy_to_cpu())
            memories = paddle.to_tensor(output_handles[1].copy_to_cpu())
            logits, qids = list(
                map(lambda x: paddle.gather(x, gather_idxs),
                    [logits, qids]))
            probs = nn.functional.softmax(logits, axis=1)
            idx = paddle.argmax(probs, axis=1).numpy()
            idx = idx.tolist()
            labels = [label_list[i] for i in idx]
            for i, qid in enumerate(qids.numpy().flatten()):
                label_dict[str(qid)] = labels[i]
    to_json_file("iflytek", label_dict, file_path)

class LongDocClassifier:
    def __init__(self, model_name_or_path,
                 trainer_num,
                 rank, batch_size=16,
                 max_seq_length=512,
                 memory_len=128,
                 static_mode=False,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.trainer_num = trainer_num
        self.rank = rank
        self.max_seq_length = max_seq_length
        self.memory_len = memory_len
        self.static_mode = static_mode
        self.kwargs = kwargs
        self.static_path = self.kwargs[
            'static_path'] if 'static_path' in self.kwargs else PPNLP_HOME

        self._construct_tokenizer()
        self._input_preparation()
        if static_mode:
            self._load_static_model()
        else:
            self._construct_model()
        pass

    def _input_preparation(self, dataset="iflytek", preprocess_text_fn=None):
        test_ds = load_dataset("clue", name=dataset, splits=["test"])
        self.label_list = test_ds.label_list
        self.num_classes = len(test_ds.label_list)
        self.test_ds_iter = ClassifierIterator(
                        test_ds,
                        self.batch_size,
                        self._tokenizer,
                        self.trainer_num,
                        trainer_id=self.rank,
                        memory_len=self.memory_len,
                        max_seq_length=self.max_seq_length,
                        mode="eval",
                        preprocess_text_fn=preprocess_text_fn)
        self.test_dataloader = paddle.io.DataLoader.from_generator(
            capacity=70, return_list=True)
        self.test_dataloader.set_batch_generator(self.test_ds_iter, paddle.get_device())

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        :return:
        """
        tokenizer_instance = ErnieDocTokenizer.from_pretrained(self.model_name_or_path)
        self._tokenizer = tokenizer_instance

    def _construct_model(self):
        """
        Construct the inference model for the predictor
        :param model_name_or_path: str
        :return: model instance
        """
        model_instance = ErnieDocForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                           num_classes=self.num_classes,
                                                                           static_mode=self.static_mode)
        self.model_config = model_instance.ernie_doc.config
        self._model = model_instance

    def _load_static_model(self, params_path=None):
        """Load static model"""
        inference_model_path = os.path.join(self.static_path, "static",
                                            "inference")
        if not os.path.exists(inference_model_path + ".pdiparams"):
            with dygraph_mode_guard():
                self._construct_model()
                if params_path:
                    state_dict = paddle.load(params_path)
                    self._model.set_dict(state_dict)
                self._construct_input_spec()
                self._convert_dygraph_to_static()
        model_file = inference_model_path + ".pdmodel"
        params_file = inference_model_path + ".pdiparams"
        self._config = paddle.inference.Config(model_file, params_file)
        self._prepare_static_mode()

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode.
        """
        place = paddle.get_device()
        if place == 'cpu':
            self._config.disable_gpu()
        else:
            self._config.enable_use_gpu(100)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        B, T, H, M, N = self.batch_size, 512, 768, 128, 12
        self._input_spec = [
            paddle.static.InputSpec(shape=[B, T, 1],
                                    dtype="int64",
                                    name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[N, B, M, H],
                                    dtype="float32",
                                    name="memories"),  # memories
            paddle.static.InputSpec(shape=[B, T, 1],
                                    dtype="int64",
                                    name="token_type_ids"),  # token_type_ids
            paddle.static.InputSpec(shape=[B, 2 * T + M, 1],
                                    dtype="int64",
                                    name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[B, T, 1],
                                    dtype="float32",
                                    name="attn_mask"),  # attn_mask
        ]

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert self._model is not None, 'The dygraph model must be created before converting the dygraph model to static model.'
        assert self._input_spec is not None, 'The input spec must be created before converting the dygraph model to static model.'
        logger.info("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(
            self._model, input_spec=self._input_spec)
        save_path = os.path.join(self.static_path, "static", "inference")
        paddle.jit.save(static_model, save_path)
        logger.info("The inference model save in the path:{}".format(save_path))

    def run_model(self, saved_path):
        memories = paddle.zeros(
            [12, 16, 128, 768], dtype="float32")
        file_path = saved_path
        if not self.static_mode:
            predict(self._model, self.test_dataloader, file_path, memories, self.label_list, self.static_mode)
        else:
            predict(self.predictor, self.test_dataloader, file_path, memories, self.label_list, self.static_mode, self.input_handles, self.output_handle)


if __name__ == "__main__":
    # Initialize model
    # paddle.set_device(args.device)
    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    model_name_or_path = "ernie-doc-base-zh"
    if rank == 0:
        if os.path.exists(model_name_or_path):
            logger.info("init checkpoint from %s" % model_name_or_path)
    predictor = LongDocClassifier(model_name_or_path=model_name_or_path,
                                  trainer_num=trainer_num,
                                  rank=trainer_num,
                                  static_mode=True)
    # predictor.run_model()