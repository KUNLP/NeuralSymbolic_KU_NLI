import os
import numpy as np
import pandas as pd
import torch
import timeit
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.file_utils import is_torch_available

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src.functions.utils import load_examples, set_seed, to_list
from src.functions.metric import get_score, get_sklearn_score

from functools import partial

def train(args, model, tokenizer, logger):
    max_acc =0
    # 학습에 사용하기 위한 dataset Load
    ## dataset: tensor형태의 데이터셋
    ## all_input_ids,
        # all_attention_masks,
        # all_labels,
        # all_cls_index,
        # all_p_mask,
        # all_example_indices,
        # all_feature_index

    train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # t_total: total optimization step
    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",  args.train_batch_size  * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    if not args.from_init_weight: global_step += int(args.checkpoint)

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    epoch_idx=0
    if not args.from_init_weight: epoch_idx += int(args.checkpoint)

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # train 모드로 설정
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # 모델에 입력할 입력 tensor 저장
            inputs_list = ["input_ids", "attention_mask"]
            if args.model_name_or_path.split("/")[-2] == "electra": inputs_list.append("token_type_ids")
            inputs_list.append("labels")
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            inputs_list2 = ['hypo_word_idxs', 'prem_word_idxs', 'hypo_span', 'prem_span']
            for m, input in enumerate(inputs_list2): inputs[input] = batch[-(m+1)]

            # Loss 계산 및 저장
            ## outputs = (total_loss,) + outputs
            outputs = model(**inputs)
            loss = outputs[0]

            # 높은 batch size는 학습이 진행하는 중에 발생하는 noisy gradient가 경감되어 불안정한 학습을 안정적이게 되도록 해줌
            # 높은 batch size 효과를 주기위한 "gradient_accumulation_step"
            ## batch size *= gradient_accumulation_step
            # batch size: 16
            # gradient_accumulation_step: 2 라고 가정
            # 실제 batch size 32의 효과와 동일하진 않지만 비슷한 효과를 보임
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            ## batch_size의 개수만큼의 데이터를 입력으로 받아 만들어진 모델의 loss는
            ## 입력 데이터들에 대한 특징을 보유하고 있다(loss를 어떻게 만드느냐에 따라 달라)
            ### loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction = ?)
            ### reduction = mean : 입력 데이터에 대한 평균
            loss.backward()
            tr_loss += loss.item()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step+1),loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        epoch_idx += 1
        #logger.info("***** Eval results *****")
        #results = evaluate(args, model, tokenizer, logger, epoch_idx = str(epoch_idx), tr_loss = loss.item())

        output_dir = os.path.join(args.output_dir, "model/checkpoint-{}".format(epoch_idx))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 학습된 가중치 및 vocab 저장
        ## pretrained 모델같은 경우 model.save_pretrained(...)로 저장
        ## nn.Module로 만들어진 모델일 경우 model.save(...)로 저장
        ### 두개가 모두 사용되는 모델일 경우 이 두가지 방법으로 저장을 해야한다!!!!
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        # # model save
        # if (args.logging_steps > 0 and max_acc < float(results["accuracy"])):
        #     max_acc = float(results["accuracy"])
        #     # 모델 저장 디렉토리 생성
        #     output_dir = os.path.join(args.output_dir, "model/checkpoint-{}".format(epoch_idx))
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #
        #     # 학습된 가중치 및 vocab 저장
        #     ## pretrained 모델같은 경우 model.save_pretrained(...)로 저장
        #     ## nn.Module로 만들어진 모델일 경우 model.save(...)로 저장
        #     ### 두개가 모두 사용되는 모델일 경우 이 두가지 방법으로 저장을 해야한다!!!!
        #     model.save_pretrained(output_dir)
        #     tokenizer.save_pretrained(output_dir)
        #     # torch.save(args, os.path.join(output_dir, "training_args.bin"))
        #     logger.info("Saving model checkpoint to %s", output_dir)

        mb.write("Epoch {} done".format(epoch + 1))

    return global_step, tr_loss / global_step

# 정답이 사전부착된 데이터로부터 평가하기 위한 함수
def evaluate(args, model, tokenizer, logger, epoch_idx = "", tr_loss = 1):
    # 데이터셋 Load
    ## dataset: tensor형태의 데이터셋
    ## example: json형태의 origin 데이터셋
    ## features: index번호가 추가된 list형태의 examples 데이터셋
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(epoch_idx))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()

    # 예측 라벨
    pred_logits = torch.tensor([], dtype = torch.long).to(args.device)
    for batch in progress_bar(eval_dataloader):
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs_list = ["input_ids", "attention_mask"]
            if args.model_name_or_path.split("/")[-2] == "electra": inputs_list.append("token_type_ids")
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            inputs_list2 = ['hypo_word_idxs', 'prem_word_idxs', 'hypo_span', 'prem_span']
            for m, input in enumerate(inputs_list2): inputs[input] = batch[-(m + 1)]

            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits,outputs[0]], dim = 0)

    # pred_label과 gold_label 비교
    pred_logits= pred_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    ## gold_labels = 0 or 1 or 2
    gold_labels = [example.gold_label for example in examples]

    # print('\n\n=====================outputs=====================')
    # for g,p in zip(gold_labels, pred_labels):
    #     print(str(g)+"\t"+str(p))
    # print('===========================================================')

    # 평가 시간 측정을 위한 time 변수
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # 최종 예측값과 원문이 저장된 example로 부터 성능 평가
    ## results  = {"macro_precision":round(macro_precision, 4), "macro_recall":round(macro_recall, 4), "macro_f1_score":round(macro_f1_score, 4), \
    ##        "accuracy":round(total_accuracy, 4), \
    ##       "micro_precision":round(micro_precision, 4), "micro_recall":round(micro_recall, 4), "micro_f1":round(micro_f1_score, 4)}
    idx2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
    #results = get_score(pred_labels, gold_labels, idx2label)
    results = get_sklearn_score(pred_labels, gold_labels, idx2label)

    output_dir = os.path.join( args.output_dir, 'eval')

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type ='w'

    # 평가 스크립트 기반 성능 저장을 위한 파일 생성
    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else: eval_file_name = "init_weight"
    output_eval_file = os.path.join(output_dir, "eval_result_{}.txt".format(eval_file_name))

    with open(output_eval_file, out_file_type, encoding='utf-8') as f:
        f.write("train loss: {}\n".format(tr_loss))
        f.write("epoch: {}\n".format(epoch_idx))
        for k in results.keys():
            f.write("{} : {}\n".format(k, results[k]))
        f.write("=======================================\n\n")
    return results

def predict(args, model, tokenizer):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True, do_predict=True)

    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    print("***** Running Prediction *****")
    print("  Num examples = %d", len(dataset))

    # 예측 라벨
    pred_logits = torch.tensor([], dtype=torch.long).to(args.device)
    for batch in progress_bar(eval_dataloader):
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs_list = ["input_ids", "attention_mask"]
            if args.model_name_or_path.split("/")[-2] == "electra": inputs_list.append("token_type_ids")
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            inputs_list2 = ['hypo_word_idxs', 'prem_word_idxs', 'hypo_span', 'prem_span']
            for m, input in enumerate(inputs_list2): inputs[input] = batch[-(m + 1)]

            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits, outputs[0]], dim=0)

    # pred_label과 gold_label 비교
    pred_logits = pred_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    ## gold_labels = 0 or 1 or 2
    gold_labels = [example.gold_label for example in examples]

    idx2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
    #results = get_score(pred_labels, gold_labels, idx2label)
    results = get_sklearn_score(pred_labels, gold_labels, idx2label)

    # 검증 스크립트 기반 성능 저장
    output_dir = os.path.join(args.output_dir, 'test')

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type = 'w'

    ## 검증 스크립트 기반 성능 저장을 위한 파일 생성
    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else:
        eval_file_name = "init_weight"
    output_test_file = os.path.join(output_dir, "test_result_{}_incorrect.txt".format(eval_file_name))

    with open(output_test_file, out_file_type, encoding='utf-8') as f:
       print('\n\n=====================outputs=====================')
       for i,(g,p) in enumerate(zip(gold_labels, pred_labels)):
           if g != p:
               f.write("premise: {}\thypothesis: {}\tcorrect: {}\tpredict: {}\n".format(examples[i].premise, examples[i].hypothesis, idx2label[g], idx2label[p]))
       for k in results.keys():
           f.write("{} : {}\n".format(k, results[k]))
       f.write("=======================================\n\n")

    out = {"premise":[], "hypothesis":[], "correct":[], "predict":[]}
    for i,(g,p) in enumerate(zip(gold_labels, pred_labels)):
        #if g != p:
            for k,v in zip(out.keys(),[examples[i].premise, examples[i].hypothesis, idx2label[g], idx2label[p]]):
                out[k].append(v)
    for k, v in zip(out.keys(), [examples[i].premise, examples[i].hypothesis, idx2label[g], idx2label[p]]):
        out[k].append(v)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(output_dir, "test_result_{}.csv".format(eval_file_name)), index=False)

    return results