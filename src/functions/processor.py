import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import transformers
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.data.processors.utils import DataProcessor

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)



def klue_convert_example_to_features(example, max_seq_length, is_training, prem_max_sentence_length, hypo_max_sentence_length, language):

    # 데이터의 유효성 검사를 위한 부분
    # ========================================================
    label = None
    if is_training:
        # Get label
        label = example.label

        # label_dictionary에 주어진 label이 존재하지 않으면 None을 feature로 출력
        # If the label cannot be found in the text, then skip this example.
        ## kind_of_label: label의 종류
        kind_of_label = ["entailment", "contradiction", "neutral"]
        actual_text = kind_of_label[label] if label<=len(kind_of_label) else label
        if actual_text not in kind_of_label:
            logger.warning("Could not find label: '%s' \n not in entailment, contradiction, and neutral", actual_text)
            return None
    # ========================================================

    # 단어와 토큰 간의 위치 정보 확인
    tok_to_orig_index = {"premise": [], "hypothesis": []}  # token 개수만큼 # token에 대한 word의 위치
    orig_to_tok_index = {"premise": [], "hypothesis": []}  # origin 개수만큼 # word를 토큰화하여 나온 첫번째 token의 위치
    all_doc_tokens = {"premise": [], "hypothesis": []}  # origin text를 tokenization
    token_to_orig_map = {"premise": {}, "hypothesis": {}}

    for case in example.merge.keys():
        new_merge = []
        new_word = []
        idx = 0
        for merge_idx in example.merge[case]:
            for m_idx in merge_idx:
                new_word.append(example.doc_tokens[case][m_idx])
            new_word.append("<WORD>")
            merge_idx = [m_idx+idx for m_idx in range(0,len(merge_idx))]
            new_merge.append(merge_idx)
            idx = max(merge_idx)+1
            new_merge.append([idx])
            idx+=1
        example.merge[case] = new_merge
        example.doc_tokens[case] = new_word

    for case in example.merge.keys():
        for merge_idx in example.merge[case]:
            for word_idx in merge_idx:
                # word를 토큰화하여 나온 첫번째 token의 위치
                orig_to_tok_index[case].append(len(tok_to_orig_index[case]))
                if (example.doc_tokens[case][word_idx] == "<WORD>"):
                    sub_tokens = ["<WORD>"]
                else: sub_tokens = tokenizer.tokenize(example.doc_tokens[case][word_idx])
                for sub_token in sub_tokens:
                    # token 저장
                    all_doc_tokens[case].append(sub_token)
                    # token에 대한 word의 위치
                    tok_to_orig_index[case].append(word_idx)
                    # token_to_orig_map: {token:word}
                    token_to_orig_map[case][len(tok_to_orig_index[case]) - 1] = len(orig_to_tok_index[case]) - 1

    # print("tok_to_orig_index\n"+str(tok_to_orig_index))
    # print("orig_to_tok_index\n"+str(orig_to_tok_index))
    # print("all_doc_tokens\n"+str(all_doc_tokens))
    # print("token_to_orig_map\n\tindex of token : index of word\n\t"+str(token_to_orig_map))

    # =========================================================

    if int(transformers.__version__[0]) <= 3:
        # sequence_added_tokens: [CLS], [SEP]가 추가된 토큰이므로 2
        ## roberta or camembert: 3
        ## <s> sen1 </s> </s> sen2 </s>
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if "roberta" in language or "camembert" in language
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        #print("sequence_added_tokens(# using special token): "+str(sequence_added_tokens))

        # special token을 제외한 최대 들어갈 수 있는 실제 premise와 hypothesis의 token길이
        ## BERT같은 경우 입력으로 '[CLS] P [SEP] H [SEP]'이므로
        ### sequence_pair_added_tokens는 special token의 개수인 3
        ### tokenizer.max_len = 512 & tokenizer.max_len_sentences_pair = 509
        ## RoBERTa는 입력으로 <s> P </s> </s> H </s>로 구성되므로
        ### sequence_pair_added_tokens는 special token의 개수인 4
        ### tokenizer.max_len = 512 & tokenizer.max_len_sentences_pair = 508
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
        #print("sequence_pair_added_tokens(# of special token in text): "+str(sequence_pair_added_tokens))

        # 최대 길이 넘는지 확인
        assert len(all_doc_tokens["premise"]) + len(all_doc_tokens["hypothesis"]) + sequence_pair_added_tokens <= tokenizer.max_len

    else:
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if "roberta" in language or "camembert" in language
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        #print("sequence_added_tokens(# using special token): "+str(sequence_added_tokens))

        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
        #print("sequence_pair_added_tokens(# of special token in text): "+str(sequence_pair_added_tokens))

        # 최대 길이 넘는지 확인
        assert len(all_doc_tokens["premise"]) + len(
            all_doc_tokens["hypothesis"]) + sequence_pair_added_tokens <= tokenizer.model_max_length

    input_ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids(token) for token in all_doc_tokens["premise"]]
    prem_word_idxs = [0] + list(filter(lambda x: input_ids[x] == tokenizer.convert_tokens_to_ids("<WORD>"),range(len(input_ids))))

    input_ids += [tokenizer.sep_token_id]
    if "roberta" in language or "camembert" in language: input_ids += [tokenizer.sep_token_id]
    token_type_ids = [0] * len(input_ids)

    input_ids += [tokenizer.convert_tokens_to_ids(token) for token in all_doc_tokens["hypothesis"]] + [tokenizer.sep_token_id]
    hypo_word_idxs = list(filter(lambda x: [tokenizer.convert_tokens_to_ids(token) for token in all_doc_tokens["hypothesis"]][x] == tokenizer.convert_tokens_to_ids("<WORD>"),range(len([tokenizer.convert_tokens_to_ids(token) for token in all_doc_tokens["hypothesis"]]))))

    hypo_word_idxs = [len(token_type_ids)-1] + [x+len(token_type_ids) for x in  hypo_word_idxs]

    token_type_ids = token_type_ids + [1] * (len(input_ids) - len(token_type_ids))
    position_ids = list(range(0, len(input_ids)))

    # non_padded_ids: padding을 제외한 토큰의 index 번호
    non_padded_ids = [i for i in input_ids]

    # tokens: padding을 제외한 토큰
    non_padded_tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

    attention_mask = [1]*len(input_ids)

    paddings = [tokenizer.pad_token_id]*(max_seq_length - len(input_ids))

    if tokenizer.padding_side == "right":
        input_ids += paddings
        attention_mask += [0]*len(paddings)
        token_type_ids += paddings
        position_ids += paddings
    else:
        input_ids  = paddings + input_ids
        attention_mask = [0]*len(paddings) + attention_mask
        token_type_ids = paddings + token_type_ids
        position_ids = paddings + position_ids

        prem_word_idxs = [x+len(paddings) for x in prem_word_idxs]
        hypo_word_idxs = [x + len(paddings) for x in hypo_word_idxs]

    # """
    # mean pooling
    prem_not_word_list = []
    for k, p_idx in enumerate(prem_word_idxs[1:]):
        prem_not_word_idxs = [0] * len(input_ids);
        for j in range(prem_word_idxs[k] + 1, p_idx):
            prem_not_word_idxs[j] = 1 / (p_idx - prem_word_idxs[k] - 1)
        prem_not_word_list.append(prem_not_word_idxs)
    prem_not_word_list = prem_not_word_list + [[0] * len(input_ids)] * (
                prem_max_sentence_length - len(prem_not_word_list))


    hypo_not_word_list = [];
    for k, h_idx in enumerate(hypo_word_idxs[1:]):
        hypo_not_word_idxs = [0] * len(input_ids);
        for j in range(hypo_word_idxs[k] + 1, h_idx):
            hypo_not_word_idxs[j] = 1 / (h_idx - hypo_word_idxs[k] - 1)
        hypo_not_word_list.append(hypo_not_word_idxs)
    hypo_not_word_list = hypo_not_word_list + [[0] * len(input_ids)] * (
                hypo_max_sentence_length - len(hypo_not_word_list))
    """
    # (a,b, |a-b|, a*b)
    prem_not_word_list = [[], []]
    for k, p_idx in enumerate(prem_word_idxs[1:]):
        prem_not_word_list[0].append(prem_word_idxs[k] + 1)
        prem_not_word_list[1].append(p_idx - 1)
    prem_not_word_list[0] = prem_not_word_list[0] + [int(hypo_word_idxs[-1]+i+2) for i in range(0, (prem_max_sentence_length - len(prem_not_word_list)))]
    prem_not_word_list[1] = prem_not_word_list[1] + [int(hypo_word_idxs[-1] + i + 2) for i in range(0, (prem_max_sentence_length - len(prem_not_word_list)))]

    hypo_not_word_list = [[],[]];
    for k, h_idx in enumerate(hypo_word_idxs[1:]):
        hypo_not_word_list[0].append(hypo_word_idxs[k]+1)
        hypo_not_word_list[1].append(h_idx+1)
    hypo_not_word_list[0] = hypo_not_word_list[0] + [int(hypo_word_idxs[-1]+i+2) for i in range(0,(hypo_max_sentence_length - len(hypo_not_word_list)))]
    hypo_not_word_list[1] = hypo_not_word_list[1] + [int(hypo_word_idxs[-1] + i + 2) for i in range(0, (hypo_max_sentence_length - len(hypo_not_word_list)))]
    """

    # p_mask: mask with 0 for token which belong premise and hypothesis including CLS TOKEN
    #           and with 1 otherwise.
    # Original TF implem also keep the classification token (set to 0)
    p_mask = np.ones_like(token_type_ids)
    if tokenizer.padding_side == "right":
        # [CLS] P [SEP] H [SEP] PADDING
        p_mask[:len(all_doc_tokens["premise"]) + len(all_doc_tokens["hypothesis"]) + 1] = 0
    else:
        p_mask[-(len(all_doc_tokens["premise"]) + len(all_doc_tokens["hypothesis"]) + 1): ] = 0

    # pad_token_indices: input_ids에서 padding된 위치
    pad_token_indices = np.array(range(len(non_padded_ids), len(input_ids)))
    # special_token_indices: special token의 위치
    special_token_indices = np.asarray(
        tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    ).nonzero()

    p_mask[pad_token_indices] = 1
    p_mask[special_token_indices] = 1

    # Set the cls index to 0: the CLS index can be used for impossible answers
    # Identify the position of the CLS token
    cls_index = input_ids.index(tokenizer.cls_token_id)

    p_mask[cls_index] = 0

    # prem_dependency  = [[premise_tail, premise_head, dependency], [], ...]
    # hypo_dependency = [[hypothesis_tail, hypothesis_head, dependency], [], ...]]
    if example.dependency["premise"] == [[]]:
        example.dependency["premise"] = [[prem_max_sentence_length-1,prem_max_sentence_length-1,0] for _ in range(0,prem_max_sentence_length)]
    else:
        example.dependency["premise"] = example.dependency["premise"] + [[prem_max_sentence_length-1,prem_max_sentence_length-1,0] for i in range(0, abs(prem_max_sentence_length-len(example.dependency["premise"])))]
    if example.dependency["hypothesis"] == [[]]:
        example.dependency["hypothesis"] = [[hypo_max_sentence_length-1,hypo_max_sentence_length-1,0] for _ in range(0,hypo_max_sentence_length)]
    else:
        example.dependency["hypothesis"] = example.dependency["hypothesis"]+ [[hypo_max_sentence_length-1,hypo_max_sentence_length-1,0] for i in range(0, abs(hypo_max_sentence_length-len(example.dependency["hypothesis"])))]

    prem_dependency = example.dependency["premise"]
    hypo_dependency = example.dependency["hypothesis"]

    return  KLUE_NLIFeatures(
            input_ids,
            attention_mask,
            token_type_ids,
            #position_ids,
            cls_index,
            p_mask.tolist(),
            example_index=0,
            tokens=non_padded_tokens,
            token_to_orig_map=token_to_orig_map,
            label = label,
            guid = example.guid,
            language = language,
            prem_dependency = prem_dependency,
            hypo_dependency=hypo_dependency,
            prem_not_word_list = prem_not_word_list,
            hypo_not_word_list = hypo_not_word_list,
        )



def klue_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def klue_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        is_training,
        return_dataset=False,
        threads=1,
        prem_max_sentence_length = 0,
        hypo_max_sentence_length = 0,
        tqdm_enabled=True,
        language = None,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=klue_convert_example_to_features_init, initargs=(tokenizer,)) as p:

        # annotate_ = 하나의 example에 대한 여러 feature를 리스트로 모은 것
        # annotate_ = list(feature1, feature2, ...)
        annotate_ = partial(
            klue_convert_example_to_features,
            max_seq_length=max_seq_length,
            prem_max_sentence_length=prem_max_sentence_length,
            hypo_max_sentence_length=hypo_max_sentence_length,
            is_training=is_training,
            language = language,
        )

        # examples에 대한 annotate_
        # features = list( feature1, feature2, feature3, ... )
        ## len(features) == len(examples)
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert klue nli examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    example_index = 0  # example의 id  ## len(features) == len(examples)
    for example_feature in tqdm(
            features, total=len(features), desc="add example index", disable=not tqdm_enabled
    ):
        if not example_feature:
            continue

        example_feature.example_index = example_index
        new_features.append(example_feature)
        example_index += 1

    features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        ## RoBERTa doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment.
        if language == "electra":
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        # all_position_ids = torch.tensor([f.#position_ids for f in features], dtype=torch.long)

        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # 전체 feature의 개별 index

        # all_dependency  = [[[premise_tail, premise_head, dependency], [], ...],[[hypothesis_tail, hypothesis_head, dependency], [], ...]], [[],[]], ... ]
        all_prem_dependency = torch.tensor([f.prem_dependency for f in features], dtype=torch.long)
        all_hypo_dependency = torch.tensor([f.hypo_dependency for f in features], dtype=torch.long)

        all_prem_not_word_list = torch.tensor([f.prem_not_word_list for f in features], dtype=torch.float)
        all_hypo_not_word_list = torch.tensor([f.hypo_not_word_list for f in features], dtype=torch.float)

        if not is_training:

            if language == "electra":
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks, all_token_type_ids, #all_position_ids,
                    all_cls_index, all_p_mask, all_feature_index,
                    all_prem_dependency, all_hypo_dependency,
                    all_prem_not_word_list, all_hypo_not_word_list
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks, # all_token_type_ids, all_position_ids,
                    all_cls_index, all_p_mask, all_feature_index,
                    all_prem_dependency, all_hypo_dependency,
                    all_prem_not_word_list, all_hypo_not_word_list
                )
        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            # label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
            # all_labels = torch.tensor([label_dict[f.label] for f in features], dtype=torch.long)

            if language == "electra":
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    #all_position_ids,
                    all_labels,
                    all_cls_index,
                    all_p_mask,
                    all_example_indices,
                    all_feature_index,
                    all_prem_dependency, all_hypo_dependency,
                    all_prem_not_word_list, all_hypo_not_word_list
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    # all_token_type_ids,
                    # all_position_ids,
                    all_labels,
                    all_cls_index,
                    all_p_mask,
                    all_example_indices,
                    all_feature_index,
                    all_prem_dependency, all_hypo_dependency,
                    all_prem_not_word_list, all_hypo_not_word_list
                )


        return features, dataset
    else:
        return features

class KLUE_NLIProcessor(DataProcessor):
    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            gold_label = None
            label = tensor_dict["gold_label"].numpy().decode("utf-8")
        else:
            gold_label = tensor_dict["gold_label"].numpy().decode("utf-8")
            label = None

        return KLUE_NLIExample(
            guid=tensor_dict["guid"].numpy().decode("utf-8"),
            genre=tensor_dict["genre"].numpy().decode("utf-8"),
            premise=tensor_dict["premise"]["origin"].numpy().decode("utf-8"),
            premise_koala=tensor_dict["premise"]["merge"]["koala"].numpy().decode("utf-8"),
            premise_merge=tensor_dict["premise"]["merge"]["origin"].numpy().decode("utf-8"),
            hypothesis=tensor_dict["hypothesis"]["origin"].numpy().decode("utf-8"),
            hypothesis_koala=tensor_dict["hypothesis"]["merge"]["koala"].numpy().decode("utf-8"),
            hypothesis_merge=tensor_dict["hypothesis"]["merge"]["origin"].numpy().decode("utf-8"),
            gold_label=gold_label,
            label=label,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.KLUE_NLIExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of KLUE_NLIExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("KLUE_NLIProcessor should be instantiated via KLUE_NLIV1Processor.")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            data_file = self.train_file if filename is None else filename
            if data_file.split(".")[-1] == "json":
                input_data = json.load(reader)
            elif data_file.split(".")[-1] == "jsonl":
                input_data = []
                for line in reader:
                    data = json.loads(line)
                    input_data.append(data)
        return self._create_examples(input_data, 'train', self.train_file if filename is None else filename)

    def get_dev_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("KLUE_NLIProcessor should be instantiated via KLUE_NLIV1Processor.")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            data_file = self.dev_file if filename is None else filename
            if data_file.split(".")[-1] == "json":
                input_data = json.load(reader)
            elif data_file.split(".")[-1] == "jsonl":
                input_data = []
                for line in reader:
                    data = json.loads(line)
                    input_data.append(data)
        return self._create_examples(input_data, "dev", self.dev_file if filename is None else filename)

    def get_example_from_input(self, input_dictionary):
        # guid, genre, premise, hypothesis
        guid=input_dictionary["guid"]
        premise=input_dictionary["premise"]
        hypothesis=input_dictionary["hypothesis"]
        gold_label=None
        label = None

        examples = [KLUE_NLIExample(
            guid=guid,
            genre="",
            premise=premise,
            hypothesis=hypothesis,
            gold_label=gold_label,
            label=label,
        )]
        return examples

    def _create_examples(self, input_data, set_type, data_file):
        is_training = set_type == "train"
        num = 0
        examples = []
        parsing = data_file.split("/")[-1].split("_")[0]
        if parsing == "merge": parsing = "koala"
        for entry in tqdm(input_data):
            guid = entry["guid"]
            if "genre" in entry: genre = entry["genre"]
            elif "source" in entry: genre = entry["source"]
            else: genre = entry["guid"].split("_")[0]
            premise = entry["premise"]["origin"]
            premise_merge = entry["premise"]["merge"]["origin"]
            premise_koala = entry["premise"]["merge"][parsing]

            hypothesis = entry["hypothesis"]["origin"]
            hypothesis_merge = entry["hypothesis"]["merge"]["origin"]
            hypothesis_koala = entry["hypothesis"]["merge"][parsing]

            gold_label = None
            label = None

            if is_training:
                label = entry["gold_label"]
            else:
                gold_label = entry["gold_label"]

            example = KLUE_NLIExample(
                guid=guid,
                genre=genre,
                premise=premise,
                premise_koala=premise_koala,
                premise_merge=premise_merge,
                hypothesis=hypothesis,
                hypothesis_koala=hypothesis_koala,
                hypothesis_merge=hypothesis_merge,
                gold_label=gold_label,
                label=label,
            )
            examples.append(example)
        # len(examples) == len(input_data)
        return examples


class KLUE_NLIV1Processor(KLUE_NLIProcessor):
    train_file = "klue-nil-v1_train.json"
    dev_file = "klue-nli-v1_dev.json"


class KLUE_NLIExample(object):
    def __init__(
        self,
        guid,
        genre,
        premise,
        premise_koala,
        premise_merge,
        hypothesis,
        hypothesis_koala,
        hypothesis_merge,
        gold_label=None,
        label=None,
    ):
        self.guid = guid
        self.genre = genre
        self.premise = premise
        self.premise_merge = premise_merge
        self.premise_koala = premise_koala
        self.hypothesis = hypothesis
        self.hypothesis_koala = hypothesis_koala
        self.hypothesis_merge = hypothesis_merge

        label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
        if gold_label in label_dict.keys():
            gold_label = label_dict[gold_label]
        self.gold_label = gold_label

        if label in label_dict.keys():
            label = label_dict[label]
        self.label = label

        # doct_tokens : 띄어쓰기 기준으로 나누어진 어절(word)로 만들어진 리스트
        ##      sentence1                   sentence2
        self.doc_tokens = {"premise":self.premise.strip().split(), "hypothesis":self.hypothesis.strip().split()}

        # merge: 말뭉치의 시작위치를 어절 기준으로 만든 리스트
        merge_index = [[],[]]
        merge_word = [[],[]]
        for merge in self.premise_merge:
            if merge[1] != []: merge_index[0].append(merge[1])
        for merge in self.hypothesis_merge:
            if merge[1] != []: merge_index[1].append(merge[1])

        # 구문구조 종류
        depend2idx = {"None":0}; idx2depend ={0:"None"}
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'None', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2

        if ([words for words in self.premise_koala if words[2][0] != words[2][1]] == []): merge_word[0].append([])
        else:
            for words in self.premise_koala:
                if words[2][0] != words[2][1]:
                    if not [merge_index[0].index(words[2][0]), merge_index[0].index(words[2][1])] in [merge_word[:-1] for merge_word in merge_word[0]]:
                        merge_word[0].append([merge_index[0].index(words[2][0]), merge_index[0].index(words[2][1]),depend2idx[words[1][0]+"-"+words[1][1]]])
                    else:
                        idx = [merge_w[:-1] for merge_w in merge_word[0]].index([merge_index[0].index(words[2][0]), merge_index[0].index(words[2][1])])
                        tag = idx2depend[[merge_w[-1] for merge_w in merge_word[0]][idx]].split("-")
                        if (words[1][1] in ['SBJ', 'CNJ', 'OBJ']) and(tag[1] in ['CMP', 'MOD', 'AJT', 'None', "UNDEF"]):
                            merge_word[0][idx][2] = depend2idx[tag[0]+"-"+words[1][1]]


        if ([words for words in self.hypothesis_koala if words[2][0] != words[2][1]] == []): merge_word[1].append([])
        else:
            for words in self.hypothesis_koala:
                if words[2][0] != words[2][1]:
                    if not [merge_index[1].index(words[2][0]), merge_index[1].index(words[2][1])] in [merge_word[:-1] for merge_word in merge_word[1]]:
                        merge_word[1].append([merge_index[1].index(words[2][0]), merge_index[1].index(words[2][1]),depend2idx[words[1][0]+"-"+words[1][1]]])
                    else:
                        idx = [merge_w[:-1] for merge_w in merge_word[1]].index([merge_index[1].index(words[2][0]), merge_index[1].index(words[2][1])])
                        tag = idx2depend[[merge_w[-1] for merge_w in merge_word[1]][idx]].split("-")
                        if (words[1][1] in ['SBJ', 'CNJ', 'OBJ']) and(tag[1] in ['CMP', 'MOD', 'AJT', 'None', "UNDEF"]):
                            merge_word[1][idx][2] = depend2idx[tag[0]+"-"+words[1][1]]

        self.merge = {"premise":merge_index[0], "hypothesis":merge_index[1]}
        self.dependency = {"premise":merge_word[0], "hypothesis":merge_word[1]}


class KLUE_NLIFeatures(object):
    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            #position_ids,
            cls_index,
            p_mask,
            example_index,
            token_to_orig_map,
            guid,
            tokens,
            label,
            language,
            prem_dependency,
            hypo_dependency,
            prem_not_word_list,
            hypo_not_word_list,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        if language == "electra": self.token_type_ids = token_type_ids
        #self.position_ids = #position_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.token_to_orig_map = token_to_orig_map
        self.guid = guid
        self.tokens = tokens

        self.label = label

        self.prem_dependency = prem_dependency
        self.hypo_dependency = hypo_dependency

        self.prem_not_word_list = prem_not_word_list,
        self.hypo_not_word_list = hypo_not_word_list,


class KLUEResult(object):
    def __init__(self, example_index, label_logits, gold_label=None, cls_logits=None):
        self.label_logits = label_logits
        self.example_index = example_index

        if gold_label:
            self.gold_label = gold_label
            self.cls_logits = cls_logits