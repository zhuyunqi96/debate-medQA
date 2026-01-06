# encoding=utf-8
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import argparse
import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import tqdm
import re
import random
import pynvml
import numpy as np


def split_text_into_chunks(paragraph, max_words=1000):
    """
    将英文段落分割成单词数不超过max_words的文本块
    
    参数:
    paragraph (str): 输入的英文段落
    max_words (int): 每个块的最大单词数，默认为1000
    
    返回:
    list: 包含分割后文本块的列表
    """
    # 将段落分割成单词列表
    words = paragraph.split()
    
    # 如果单词总数小于等于max_words，直接返回整个段落
    if len(words) <= max_words:
        return [paragraph]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    # 遍历所有单词
    for word in words:
        # 如果当前块还能容纳这个单词
        if current_word_count + 1 <= max_words:
            current_chunk.append(word)
            current_word_count += 1
        else:
            # 保存当前块并开始新块
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def get_device_memory_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(4)
    
    # 获取设备总内存和已使用内存
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # total_memory = info.total / (1024 ** 2)  # GB
    used_memory = info.used / (1024 ** 2)    # GB
    
    # print(f"设备总显存: {total_memory:.2f} GB")
    # print(f"设备已用显存: {used_memory:.2f} GB")
    
    pynvml.nvmlShutdown()
    return used_memory

def remove_consecutive_repeated_sentences(text):
    """
    删除文本中连续重复的句子（保留首次出现的句子），非连续重复的句子保留
    
    参数:
        text: 输入文本（支持中英文）
        
    返回:
        处理后的文本，已删除连续重复的句子
    """
    # 智能分割句子（兼容中英文标点）
    sentence_pattern = r'(?<![A-Za-z]\.)(?<=[\n.!?。！？])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # 处理连续重复
    result = []
    prev_sentence = None
    
    for sentence in sentences:
        # 标准化处理：移除首尾空格和标点，保留核心内容
        cleaned = re.sub(r'^[：“”‘’【】\s]+|[：“”‘’【】\s]+$', '', sentence).strip()
        
        # 空句子跳过
        if not cleaned:
            continue
            
        # 检查是否与上一句核心内容相同
        if cleaned != prev_sentence:
            result.append(sentence)  # 保留原始句子格式
            prev_sentence = cleaned
    
    # 重组文本（保留原有分隔符结构）
    return ''.join(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run baseline answer generation.")
    parser.add_argument("--model", "-model", default=None, type=int)
    parser.add_argument("--debateType", "-debateType", default=None, type=str)
    parser.add_argument("--alter", "-alter", default=None, type=str)
    parser.add_argument("--useCoT", "-useCoT", default=1, type=int)
    parser.add_argument("--judgeType", "-judgeType", default=0, type=int)
    parser.add_argument("--datasetID", "-datasetID", default=-1, type=int)
    parser.add_argument("--testMemAlloc", "-testMemAlloc", default=0, type=int)
    parser.add_argument("--testMemAlloc_samples", "-testMemAlloc_samples", default=100, type=int)
    parser.add_argument("--CUDA_VISIBLE_DEVICES", "-CUDA_VISIBLE_DEVICES", default="4", type=str)
    parser_args = parser.parse_args()

    CUDA_VISIBLE_DEVICES = parser_args.CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES']= CUDA_VISIBLE_DEVICES

    torch.cuda.reset_peak_memory_stats()
    max_GPU_mem = []

    judgeType = int(parser_args.judgeType)
    judgeType_suffix = f"-type{judgeType}" if judgeType in [0, 1] and judgeType != 0 else ""

    datasetID = int(parser_args.datasetID)
    assert datasetID in [-1, 0, 1], "datasetID should be in [-1, 0, 1]"


    print(f"judgeType {judgeType}")
    print(f"datasetID {datasetID}")

    USE_MODEL_number = parser_args.model

    debateType = parser_args.debateType
    assert debateType is not None

    alterType = parser_args.alter
    alterType = int(alterType) if alterType is not None else None

    assert alterType in [None, 1, 2, 3], f"alterType {alterType} not in [None, 1, 2, 3]"
    assert parser_args.useCoT in [0, 1], f"useCoT {parser_args.useCoT} not in [0, 1]"

    USE_CHAIN_OF_THOUGHT = True if parser_args.useCoT == 1 else False

    METHOD_TYPE = "baseline-CoT" if USE_CHAIN_OF_THOUGHT else "baseline-NO-CoT"
    METHOD_TYPE = METHOD_TYPE.replace("baseline", "debate-RAG")
    
    extendData_suffix = ""
    if datasetID in [0, 1]:
        METHOD_TYPE = f"{METHOD_TYPE}-extendData{datasetID}"
        extendData_suffix = f"-extendData{datasetID}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 大模型生成答案
    TARGET_MODEL_PATH = "DeepSeek-R1-Distill-Qwen-1.5B"
    MODEL_NAME_surfix = ""
    if USE_MODEL_number == 2:
        TARGET_MODEL_PATH = "Llama-3.2-1B-Instruct"
        MODEL_NAME_surfix = "-LLAMA-3.2-1B"


    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH)
    template = tokenizer.chat_template
    template = template.replace("<think>\\n", "")  # 去掉末尾的 <think>\\n
    tokenizer.chat_template = template  # Set the new template

    MODEL_AGENT_1 = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    MODEL_AGENT_1.half()  # 使用半精度
    MODEL_AGENT_1.to(device)
    MODEL_AGENT_1.eval()
    print(f"=== MODEL_AGENT_1 loaded {TARGET_MODEL_PATH} ===")


    # 嵌入模型与向量库
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        # cache_folder=""
    )
    train_jsonl_path = "agent-debates/train-agent-debate-judge-all.jsonl"

    if debateType == "judge" and judgeType == 1:
        train_jsonl_path = "agent-debates/train-agent-debate-judge-type1-all.jsonl"


    print(f"debateType {debateType}, alterType {alterType}")
    assert debateType in ["first", "challenge", "judge"]

    def clean_json_string(input):
        output = input.split("</think>")[-1].split("<｜end▁of▁sentence｜>")[0]
        output = output.replace("```json", "").replace("```", "").replace("   ", "").replace("  ", "").strip()
        return output

    vectorstore = []
    with open(train_jsonl_path, "r", encoding="utf-8") as r_f:

        print(f"load from {train_jsonl_path}")

        for i, line in enumerate(r_f):
            data = json.loads(line.strip())
            dataset, type, question, answer = data["dataset"], data["type"], data["question"], data["answer"]
            agent_1_debate, agent_2_challenge, agent_3_judge_sum = data.get("agent_1_debate"), data.get("agent_2_challenge"), data.get("agent_3_judge_sum")

            
            agent_1_debate = remove_consecutive_repeated_sentences(clean_json_string(agent_1_debate))
            agent_2_challenge = remove_consecutive_repeated_sentences(clean_json_string(agent_2_challenge))
            agent_3_judge_sum = remove_consecutive_repeated_sentences(clean_json_string(agent_3_judge_sum))
            
            
            prefix = f"Question: {question}\nAnswer: {answer}\n"
            if alterType == 1:
                prefix = ""


            if alterType in [2, 3]:

                def clean_symbol(text): # 清理中英文的符号
                    clean_text = re.sub(r'^[：“”‘’【】\s]+|[：“”‘’【】\s]+$', '', text).strip()
                    clean_text = clean_text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "")
                    clean_text = clean_text.replace("\n", " ").replace("\t", " ").replace("\"", "").replace("'", "").replace("`", "").replace(" , ", " ")
                    if len(clean_text) > 0:
                        if clean_text[0] == ":":
                            clean_text = clean_text[1:]

                    return clean_text.strip()
                
                if "direct_knowledge" in agent_1_debate:
                    agent_1_debate = agent_1_debate.split("direct_knowledge")[-1]
                if "direct_knowledge" in agent_2_challenge:
                    agent_2_challenge = agent_2_challenge.split("direct_knowledge")[-1]
                if "summary" in agent_3_judge_sum:
                    agent_3_judge_sum = agent_3_judge_sum.split("summary")[-1].split("\",")[0]

                agent_1_debate = clean_symbol(agent_1_debate)
                agent_2_challenge = clean_symbol(agent_2_challenge)
                agent_3_judge_sum = clean_symbol(agent_3_judge_sum)

                if alterType == 3: # 如果是 alter 3, 则不加 prefix
                    prefix = ""


            if debateType == "judge" and judgeType == 1:

                if "1" in agent_3_judge_sum: # 如果包含 "1"，则选择 agent_1_debate
                    agent_3_judge_sum = agent_1_debate
                elif "2" in agent_3_judge_sum: # 如果包含 "2"，则选择 agent_2_challenge
                    agent_3_judge_sum = agent_2_challenge
                else: # 否则默认选择 agent_1_debate
                    agent_3_judge_sum = agent_1_debate

            # print(agent_1_debate)
            # print("#####################################################")
            # print(agent_2_challenge)
            # print("#####################################################")
            # print(agent_3_judge_sum)
            # exit()

            if debateType == "first":
                vectorstore.append(
                    f"{prefix}{agent_1_debate}"
                )

            elif debateType == "challenge":
                vectorstore.append(
                    f"{prefix}{agent_2_challenge}"
                )

            elif debateType == "judge":
                vectorstore.append(
                    f"{prefix}{agent_3_judge_sum}"
                )


    if datasetID == 0:
        with open("RAG-compare-dataset/mimic-iv-discharge_split.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
            for key in ["train", "eval", "test"]:
                temp_list = dataset[key]
                temp_list = [item["source"] for item in temp_list]
                temp_list = [item for item in temp_list if len(item) > 20]  # 只添加长度大于20的文本

                for paragraph in temp_list:
                    paragraph_chunks = split_text_into_chunks(paragraph, max_words=1000)
                    for chunk in paragraph_chunks:
                        vectorstore.append(chunk)

    elif datasetID == 1:
        for text_file in os.listdir("RAG-compare-dataset/textbooks-18-en"):
            if text_file.endswith(".txt"):
                with open(os.path.join("RAG-compare-dataset/textbooks-18-en", text_file), "r", encoding="utf-8") as f:
                    content = f.read()

                    for paragraph in content.split("\n\n"):
                        paragraph = paragraph.strip()
                        if len(paragraph) > 20:  # 只添加长度大于20的段落
                            
                            paragraph_chunks = split_text_into_chunks(paragraph, max_words=1000)
                            for chunk in paragraph_chunks:
                                vectorstore.append(chunk)

    persist_directory_path = f"./chroma-vectorstore/{debateType}{judgeType_suffix}{extendData_suffix}"
    if alterType is not None:
        persist_directory_path = f"{persist_directory_path}-alter-{alterType}"



    if not os.path.exists(persist_directory_path):
        os.mkdir(persist_directory_path)
        vectorstore = Chroma.from_texts(
            vectorstore, embeddings, persist_directory=persist_directory_path
        )
    else:
        vectorstore = Chroma(
            persist_directory=persist_directory_path, embedding_function=embeddings
        )
    
    # 检索与生成整合
    RETRIEVE_TOP_K = 10
    print(f"RETRIEVE_TOP_K {RETRIEVE_TOP_K}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    print(f"=== retriever prepared ===")

    RETRIEVE_REORDER_TOP_K = 3
    print(f"RETRIEVE_REORDER_TOP_K {RETRIEVE_REORDER_TOP_K}")


    ANSWER_FORMAT = """\
{"answer": [answer of the question], "explanation": [explanation of the answer]}\
"""

    test_jsonl_path = "prepared-datasets/test.jsonl"
    with open(test_jsonl_path, "r", encoding="utf-8") as r_f:
        total_size = sum(1 for _ in r_f) # 统计多少条

    METHOD_TYPE = f"{METHOD_TYPE}-{debateType}"
    print(f"=== METHOD_TYPE: {METHOD_TYPE}, USE_CHAIN_OF_THOUGHT: {USE_CHAIN_OF_THOUGHT} ===")

    #############
    debate_json_path = f"agent-debates-4/test-answer-{METHOD_TYPE}{MODEL_NAME_surfix}{judgeType_suffix}"
    if alterType is not None:
        debate_json_path = f"{debate_json_path}-alter-{alterType}"

    debate_json_path += ".jsonl"
    #############

    print(f"write json path = {debate_json_path}")
    
    ################
    testMemAlloc = parser_args.testMemAlloc == 1
    testMemAlloc_samples = parser_args.testMemAlloc_samples
    if testMemAlloc:
        random.seed(42)
        # 从 total_size 中随机选择 testMemAlloc_samples 个索引，不重复筛选
        testMemAlloc_indices = sorted(list(set(random.sample(range(total_size), testMemAlloc_samples))))
        total_size = testMemAlloc_samples
        print(f"=== testMemAlloc {testMemAlloc}, testMemAlloc_samples {testMemAlloc_samples} ===")

    if os.path.exists(debate_json_path) and not testMemAlloc:
        os.remove(debate_json_path)

    batch_size = 1

    with open(test_jsonl_path, "r", encoding="utf-8") as r_f:
        tbar = tqdm.trange(total_size // batch_size)

        INPUT_BATCH, DICT_BATCH = [], []

        for i, line in enumerate(r_f):

            data = json.loads(line.strip())
            dataset, type, question, answer = data["dataset"], data["type"], data["question"], data["answer"]


            docs = retriever.invoke(question)
            reordering = LongContextReorder()
            reordered_docs = reordering.transform_documents(docs)
            retrieved_context = "\n\n".join([doc.page_content for doc in reordered_docs[:RETRIEVE_REORDER_TOP_K]])


            PROMPTED_QUERY = f"Role: You are a Medical Expert Bot.\n\n\
                Given the similar Q&A context: {retrieved_context}\n\n\
                Analyze the following Question: \nQuestion: {question}\n\n\
                Your task is to analyse the question, and then provide your answer and your explanation of your answer.\n\
                Provide your response strictly following this JSON format: \n\n{ANSWER_FORMAT}."
            
            if USE_CHAIN_OF_THOUGHT:
                 PROMPTED_QUERY += "<think>"

            
            INPUT_BATCH.append(PROMPTED_QUERY)
            DICT_BATCH.append(data)

            if len(INPUT_BATCH) == batch_size:
                inputs = tokenizer.batch_encode_plus(
                    INPUT_BATCH,
                    return_tensors="pt",
                    # padding=True,
                    # pad_to_multiple_of=8,
                ).to(device)

                outputs_MODEL_1 = MODEL_AGENT_1.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=50,
                    max_new_tokens=2048,
                    repetition_penalty=1.2,
                    num_beams=4,
                )
                # 解码输出
                DECODED_ANSWERS = tokenizer.batch_decode(outputs_MODEL_1, clean_up_tokenization_spaces=True)

                for decoded_ans, data_dict in zip(DECODED_ANSWERS, DICT_BATCH):

                    if testMemAlloc and i in testMemAlloc_indices:
                        continue

                    with open(debate_json_path, "a", encoding="utf-8") as f:
                        temp_dict = dict(data_dict)
                        temp_dict[f"agent_answer_{METHOD_TYPE}"] = decoded_ans.strip()
                        # 将每个字典转换为JSON字符串并写入文件，每行一个
                        json_line = json.dumps(temp_dict, ensure_ascii=False)
                        f.write(json_line + "\n")

                INPUT_BATCH, DICT_BATCH = [], []

                tbar.update()

                max_GPU_mem.append(get_device_memory_usage())
                ########################
                if testMemAlloc and tbar.n == testMemAlloc_samples:
                    current_method = debate_json_path.split("/")[-1].replace(".jsonl", "")
                    # 打开jsonl文件，写入显存占用信息
                    with open(f"GPU-mem-alloc-v2.jsonl", "a", encoding="utf-8") as mem_f:
                        json_line = json.dumps({
                            "method": current_method,
                            "memory_MB_median": np.median(max_GPU_mem),
                            "memory_MB_peak": max(max_GPU_mem),
                            "GPU_mem": max_GPU_mem
                        }, ensure_ascii=False)
                        mem_f.write(json_line + "\n")
                    print(f"=== peak memory usage: {max_GPU_mem} MB, {current_method} ===")
                    exit()
                ########################

            
    print(f"=== done ===")