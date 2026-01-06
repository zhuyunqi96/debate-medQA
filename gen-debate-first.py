# encoding=utf-8
import os

os.environ['CUDA_VISIBLE_DEVICES']= "6"
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run baseline answer generation.")
    parser.add_argument("--model", "-model", default=None, type=int)
    parser.add_argument("--start", "-start", default=0, type=int)
    parser.add_argument("--end", "-end", default=2000, type=int)
    parser_args = parser.parse_args()

    USE_MODEL_number = parser_args.model
    start_index = parser_args.start
    end_index = parser_args.end
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
    MODEL_AGENT_1.half()  # 使用半精度
    MODEL_AGENT_1.to(device)
    MODEL_AGENT_1.eval()
    print(f"=== MODEL_AGENT_1 loaded {TARGET_MODEL_PATH} ===")

    # # 嵌入模型与向量库
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-base-en-v1.5",
    #     # cache_folder=""
    # )
    # vectorstore = Chroma.from_texts(
    #     [
    #         "Retrieval-augmented generation (RAG) is an innovative approach in the field of natural language processing (NLP) that combines the strengths of retrieval-based and generation-based models to enhance the quality of generated text.",
    #         "Retrieval Augmented Generation (RAG) is a pattern that works with pretrained Large Language Models (LLM) and your own data to generate responses",
    #     ], 
    #     embeddings
    # )
    # # 检索与生成整合
    # retriever = vectorstore.as_retriever()
    # print(f"=== retriever prepared ===")
    # TOP_K_RESULT = 5
    # MAX_TOKEN = 1024

    # question = "what is RAG?"

    # docs = retriever.invoke(question)
    # for doc_i, doc in enumerate(docs):
    #     print(f"retrieved doc {doc_i}: {doc.page_content}")

    # prompt = PromptTemplate.from_template("Based on the context: {context}\nAnswer: {query}")
    # inputs = tokenizer(
    #     prompt.format(context=docs[0].page_content, query="What is RAG?"), 
    #     return_tensors="pt"
    # ).to(device)
    # outputs_MODEL_1 = MODEL_AGENT_1.generate(**inputs)
    # print("=== Answer MODEL_AGENT_1 ===")

    # print(tokenizer.decode(outputs_MODEL_1[0]))  # 输出详细解释...


    ANSWER_FORMAT = """\
{"direct_knowledge": ["List of explicit medical facts"], \
"indirect_knowledge": ["List of implicit reasoning pattern"], \
"knowledge_gaps": ["List of unaddressed/unmentioned critical aspects"]}, \
{"key_learnings": ["3-5 summarized principles from the Q&A"]}\
"""

    train_jsonl_path = "prepared-datasets/train.jsonl"
    with open(train_jsonl_path, "r", encoding="utf-8") as r_f:
            total_size = sum(1 for _ in r_f) # 统计多少条

    if end_index > total_size:
        end_index = total_size

    debate_json_path = f"agent-debates/train-agent-debate-first{MODEL_NAME_surfix}-part-{start_index}-{end_index}.jsonl"
    print(f"write json path = {debate_json_path}")

    if os.path.exists(debate_json_path):
        os.remove(debate_json_path)

    batch_size = 1

    tbar_size = (end_index - start_index) // batch_size

    with open(train_jsonl_path, "r", encoding="utf-8") as r_f:
        tbar = tqdm.trange(tbar_size)

        INPUT_BATCH, DICT_BATCH = [], []

        for i, line in enumerate(r_f):

            if i < start_index or i >= end_index:
                continue

            data = json.loads(line.strip())
            dataset, type, question, answer = data["dataset"], data["type"], data["question"], data["answer"]
            PROMPTED_QUERY = f"Role: You are a Medical Knowledge Analyst Bot.\n\n\
                Analyze the following Q&A: \nQuestion: {question}\nAnswer: {answer}\n\n\
                Your task is to decompose medical Q&A pairs, extract explicit and implicit knowledge, and provide critical insights for the medical Q&A.\n\
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
                # DECODED_ANSWERS = tokenizer.batch_decode(outputs_MODEL_1)
                # print(DECODED_ANSWERS)
                # exit()

                for decoded_ans, data_dict in zip(DECODED_ANSWERS, DICT_BATCH):

                    with open(debate_json_path, "a", encoding="utf-8") as f:
                        temp_dict = dict(data_dict)
                        temp_dict[f"agent_1_debate"] = decoded_ans.strip()
                        # 将每个字典转换为JSON字符串并写入文件，每行一个
                        json_line = json.dumps(temp_dict, ensure_ascii=False)
                        f.write(json_line + "\n")

                INPUT_BATCH, DICT_BATCH = [], []

                tbar.update()

            
    print(f"=== done ===")