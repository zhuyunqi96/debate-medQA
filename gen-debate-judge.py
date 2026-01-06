# encoding=utf-8
import os

os.environ['CUDA_VISIBLE_DEVICES']= "3"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import argparse
import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import tqdm
import re

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
    parser.add_argument("--start", "-start", default=0, type=int)
    parser.add_argument("--end", "-end", default=2000, type=int)
    parser.add_argument("--judgeType", "-judgeType", default=0, type=int)

    parser_args = parser.parse_args()

    USE_MODEL_number = parser_args.model

    judgeType = int(parser_args.judgeType)
    start_index = int(parser_args.start)
    end_index = int(parser_args.end)

    print(f"judgeType {judgeType}")
    
    print(f"start_index {start_index} end_index {end_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    USE_CHAIN_OF_THOUGHT = True
    print(f"USE_CHAIN_OF_THOUGHT {USE_CHAIN_OF_THOUGHT}")

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
    MODEL_AGENT_1.half()
    MODEL_AGENT_1.to(device)
    MODEL_AGENT_1.eval()
    print(f"=== MODEL_AGENT_1 loaded {TARGET_MODEL_PATH} ===")

    end_index = min(end_index, 10357)


    train_jsonl_path = f"agent-debates/train-agent-debate-challenge-part-{start_index}-{end_index}.jsonl"
    with open(train_jsonl_path, "r", encoding="utf-8") as r_f:
            total_size = sum(1 for _ in r_f) # 统计多少条

    # if end_index > total_size:
        # end_index = total_size

    judgeType_suffix = f"-type{judgeType}-" if judgeType in [0, 1] and judgeType != 0 else ""

    debate_json_path = f"agent-debates/train-agent-debate-judge{judgeType_suffix}{MODEL_NAME_surfix}-part-{start_index}-{end_index}.jsonl"
    print(f"write json path = {debate_json_path}")

    if os.path.exists(debate_json_path):
        os.remove(debate_json_path)

    batch_size = 1

    tbar_size = (end_index - start_index) // batch_size

    def clean_json_string(input):
        output = input.split("</think>")[-1].split("<｜end▁of▁sentence｜>")[0]
        output = output.replace("```json", "").replace("```", "").replace("   ", "").replace("  ", "").strip()
        return output

    ANSWER_FORMAT, PROMPTED_QUERY = None, None
    with open(train_jsonl_path, "r", encoding="utf-8") as r_f:
        tbar = tqdm.trange(tbar_size)

        INPUT_BATCH, DICT_BATCH = [], []

        for i, line in enumerate(r_f):

            # if i < start_index or i >= end_index:
                # continue

            data = json.loads(line.strip())
            dataset, type, question, answer, agent_1_debate, agent_2_challenge = data["dataset"], data["type"], data["question"], data["answer"], data["agent_1_debate"], data["agent_2_challenge"]
            
            agent_1_argument = clean_json_string(agent_1_debate)
            agent_1_argument = remove_consecutive_repeated_sentences(agent_1_argument)


            agent_2_argument = clean_json_string(agent_2_challenge)
            agent_2_argument = remove_consecutive_repeated_sentences(agent_2_argument)

            if judgeType == 0: # default version
                ANSWER_FORMAT = """{"summary": "[summary of valuable information and knowledge in bullet points or structured paragraphs]"}"""

                PROMPTED_QUERY = f"Role: You are a critical thinker, judge and medical knowledge analyst Bot.\n\n\
                Base on the Q&A and Analysis-1 and Analysis-2: \nQuestion: {question}\nAnswer: {answer}\nAnalysis-1: {agent_1_argument}\n\nAnalysis-2: {agent_2_argument}\
                Your task is to decompose medical Q&A pairs, critically evaluate Analysis-1 and Analysis-2, then summarize valuable information and knowledge.\n\
                Provide your response strictly following this JSON format: \n\n{ANSWER_FORMAT}."

            elif judgeType == 1: # select better one version
                ANSWER_FORMAT = """{"better analysis": "[1 or 2]"}"""

                PROMPTED_QUERY = f"Role: You are a critical thinker, judge and medical knowledge analyst Bot.\n\n\
                Base on the Q&A and Analysis-1 and Analysis-2: \nQuestion: {question}\nAnswer: {answer}\nAnalysis-1: {agent_1_argument}\n\nAnalysis-2: {agent_2_argument}\
                Your task is to decompose medical Q&A pairs, critically evaluate Analysis-1 and Analysis-2, then select the better one between Analysis-1 and Analysis-2. \
                Simply provide the number of the better analysis in the response.\n\
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
                # decoded_ans = DECODED_ANSWERS[0].split("</think>")[-1]
                # print(f"{question}\n{answer}=====\n{decoded_ans}")

                # print(f"{decoded_ans}")
                # exit()

                for decoded_ans, data_dict in zip(DECODED_ANSWERS, DICT_BATCH):

                    with open(debate_json_path, "a", encoding="utf-8") as f:
                        temp_dict = dict(data_dict)
                        temp_dict[f"agent_3_judge_sum"] = decoded_ans.strip()
                        # 将每个字典转换为JSON字符串并写入文件，每行一个
                        json_line = json.dumps(temp_dict, ensure_ascii=False)
                        f.write(json_line + "\n")

                INPUT_BATCH, DICT_BATCH = [], []

                tbar.update()

            
    print(f"=== done ===")