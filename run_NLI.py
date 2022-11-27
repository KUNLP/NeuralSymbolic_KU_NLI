import argparse
import os
import logging
from attrdict import AttrDict

# roberta
from transformers import AutoTokenizer
from transformers import RobertaConfig

#from src.model.model_graph import RobertaForSequenceClassification
from src.model.model5 import RobertaForSequenceClassification
#from src.model.model_baseline import RobertaForSequenceClassification

from src.model.main_functions5 import train, evaluate, predict
#from src.model.main_functions_baseline import train, evaluate, predict

from src.functions.utils import init_logger, set_seed

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def create_model(args):

    if args.model_name_or_path.split("/")[-2] == "roberta":

        # 모델 파라미터 Load
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
        )

        config.num_labels = args.num_labels
        # roberta attention 추출하기
        config.output_attentions=True

        # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
        # BertTokenizerFast로 되어있음
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,

        )
        print(tokenizer)

        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
            config=config,
            prem_max_sentence_length=args.prem_max_sentence_length,
            hypo_max_sentence_length=args.hypo_max_sentence_length,
            # from_tf=True if args.from_init_weight else False
        )

        args.model_name_or_path = args.cache_dir
        # print(tokenizer.convert_tokens_to_ids("<WORD>"))

    # vocab 추가
    # 중요 단어의 UNK 방지 및 tokenize를 방지해야하는 경우(HTML 태그 등)에 활용
    # "세종대왕"이 OOV인 경우 ['세종대왕'] --tokenize-->  ['UNK'] (X)
    # html tag인 [td]는 tokenize가 되지 않아야 함. (완전한 tag의 형태를 갖췄을 때, 의미를 갖기 때문)
    #                             ['[td]'] --tokenize-->  ['[', 't', 'd', ']'] (X)

    if args.from_init_weight and args.add_vocab:
        if args.from_init_weight:
            add_token = {
                "additional_special_tokens": ["[td]", "추가 단어 1", "추가 단어 2"]}
            # 추가된 단어는 tokenize 되지 않음
            # ex
            # '[td]' vocab 추가 전 -> ['[', 't', 'd', ']']
            # '[td]' vocab 추가 후 -> ['[td]']
            tokenizer.add_special_tokens(add_token)
            model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    print("<WORD> idx")
    print(tokenizer.convert_tokens_to_ids("<WORD>"))
    return model, tokenizer

def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    model, tokenizer = create_model(args)

    # Running mode에 따른 실행
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        #for i in range(1, 13): evaluate(args, model, tokenizer, logger, epoch_idx =i)
        evaluate(args, model, tokenizer, logger, epoch_idx =args.checkpoint)
        
    elif args.do_predict:
        predict(args, model, tokenizer)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # Directory

    # ------------------------------------------------------------------------------------------------
    # english
    cli_parser.add_argument("--data_dir", type=str, default="./data/snli/parsing/dependency")

    cli_parser.add_argument("--train_file", type=str, default='snli_1.0_train_2.jsonl')
    cli_parser.add_argument("--eval_file", type=str, default='snli_1.0_dev_2.jsonl')
    cli_parser.add_argument("--predict_file", type=str, default='snli_1.0_test_2.jsonl')

    cli_parser.add_argument("--num_labels", type=int, default=3)

    # roberta
    cli_parser.add_argument("--model_name_or_path", type=str, default="./roberta/init_weight")
    cli_parser.add_argument("--cache_dir", type=str, default="./roberta/init_weight")

    #------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--data_dir", type=str, default="./data")
    # cli_parser.add_argument("--train_file", type=str, default="klue-nli-v1_train.json")
    # cli_parser.add_argument("--eval_file", type=str, default="klue-nli-v1_dev.json")
    # cli_parser.add_argument("--predict_file", type=str, default="klue-nli-v1_dev.json") #"klue-nli-v1_dev_sample_10.json")

    #cli_parser.add_argument("--num_labels", type=int, default=3)

    # roberta
    #cli_parser.add_argument("--model_name_or_path", type=str, default="./roberta/init_weight_ver1")
    #cli_parser.add_argument("--cache_dir", type=str, default="./roberta/init_weight_ver1")
    ## (baseline)checkout-1074: acc = 85.13  # checkout-4 89.91

    # ------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--data_dir", type=str, default="./data/merge")

    # cli_parser.add_argument("--num_labels", type=int, default=3)

    ## merge는 koala DP 사용
    # cli_parser.add_argument("--train_file", type=str, default='merge_1re_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='merge_1re_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='merge_1re_klue_nli_dev.json')

    ## parsing은 우리연구실 DP 사용
    # cli_parser.add_argument("--train_file", type=str, default='parsing_1_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='parsing_1_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='parsing_1_klue_nli_dev.json')

    # ------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--train_file", type=str, default='merge_re3_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='merge_re3_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='merge_re3_klue_nli_dev.json')

    # cli_parser.add_argument("--train_file", type=str, default='parsing_3_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='parsing_3_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='parsing_3_klue_nli_dev.json')

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--train_file", type=str, default='merge_re4_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='merge_re4_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='merge_re4_klue_nli_dev.json')

    # cli_parser.add_argument("--train_file", type=str, default='parsing_4_klue_nli_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='parsing_4_klue_nli_dev.json')
    # cli_parser.add_argument("--predict_file", type=str, default='parsing_4_klue_nli_dev.json')

    # ------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------------

    # merge_1re_klue_nli_train.json
    ## SIC1
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_5")   # checkout-3 90.31 checkout-4 90.22

    # ver1_4: ver1_3에서 NP-MOD와 같은 tag정보를 안주고 서로 연결되었다는 정보만 주면??
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_4")   # checkout-5 90.44
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_3")   # checkout-5 90.31

    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_2")   # checkout-5 90.29

    # ver1_woDP: ver1_wDP에서 NP-MOD와 같은 tag정보를 안주고 서로 연결되었다는 정보만 주기
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_woDP")# checkout-4  90.53
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1_wDP") # checkout-3 90.60
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver1")     # checkout-5 90.11   # checkout-27 91.13

    ## SIC2
    # cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver2_wDP") # checkout-4 90.24
    # merge_3_klue_nli_train.json
    # cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver3_wDP") # checkout-5 90.31
    # merge_4_klue_nli_train.json
    # cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/ver4_wDP") # checkout-4 90.49

    ##################################################################################################################################

    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver1_wDP") # checkout-5 90.60    90.56 ± 0.04
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver3_wDP") # checkout-3 90.51    90.45 ± 0.06
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_wDP") # checkout-4 90.82    90.78 ± 0.04

    # ./roberta/my_model/parsing/ver4_wDP

    ## bilstm + bilinear
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_2")       # checkout-5 90.53     90.45 ± 0.08

    ## tag정보 + bilstm + bilinear
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_3")       # checkout-5 90.67     90.63 ± 0.04

    ## tag연결 + bilstm + bilinear
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_4")       # checkout-5 90.76     90.65 ± 0.11

    ## tag연결 + biaffine attention + bilstm + bilinear
    # cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_woDP")   # checkout-5 90.47     90.36 ± 0.11

    ## SIC2
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/my_model/parsing/ver4_wDP_SIC2") # checkout-5 90.53    90.50 ± 0.03

    #GAT
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/gat_model/parsing/mean_pooling")           # checkout-5 90.53     # 90.42 ± 0.09
    #cli_parser.add_argument("--output_dir", type=str,default="./roberta/gat_model/parsing/mean_pooling/connecting")  # checkout-3 90.80     90.65 ± 0.15
    ## 이거 두개는 너무 성능 안나옴 패스
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/gat_model/parsing/mean_pooling/parser_both")  # checkout-
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/gat_model/parsing/mean_pooling/connecting_both")  # checkout-

    # snli datasets
    cli_parser.add_argument("--output_dir", type=str, default="./roberta/snli/baseline/parsing/mean_pooling")

    # ------------------------------------------------------------------------------------------------------------
    # snli
    cli_parser.add_argument("--prem_max_sentence_length", type=int, default=100)
    cli_parser.add_argument("--hypo_max_sentence_length", type=int, default=100)
    ## klue # ver1 = 18     ver3,4 = 27
    #cli_parser.add_argument("--prem_max_sentence_length", type=int, default=27) # data["premise"]["merge"]["origin"]의 최대 개수 + 1(뒤에는 padding으로 보기)
    #cli_parser.add_argument("--hypo_max_sentence_length", type=int, default=27)

    # https://github.com/KLUE-benchmark/KLUE-baseline/blob/main/run_all.sh
    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=256) #512)
    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=1e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=8)
    cli_parser.add_argument("--eval_batch_size", type=int, default=16)
    cli_parser.add_argument("--num_train_epochs", type=int, default=10)

    #cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--logging_steps", type=int, default=100)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default= True) #False)#True)
    cli_parser.add_argument("--checkpoint", type=str, default="8") # (~150000: checkpoint-8: acc-84.06)
    cli_parser.add_argument("--add_vocab", type=bool, default=False)
    cli_parser.add_argument("--do_train", type=bool, default=True)#False)#True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)#True)
    cli_parser.add_argument("--do_predict", type=bool, default=False)#True)#False)

    cli_args = cli_parser.parse_args()

    main(cli_args)
