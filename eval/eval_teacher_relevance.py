import requests
import linecache2 as linecache
import os
import time

def get_chat_gpt_response(prompt):
    url = "xxx"
    headers = {
        "Authorization": "xxx",
        "Content-Type": "application/json"
    }
    data = {
        "model": "xxx",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

def main():
    input_file = "xxx"
    knowledge_file = "xxx"
    output_file = "relevance.txt"
    
    # 用于记录脚本当前进度的文件
    progress_file = "progress_log.txt"
    
    # 1. 读取进度文件，获取上次成功处理的行号
    start_line = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as pf:
            line_str = pf.read().strip()
            if line_str.isdigit():
                start_line = int(line_str)
    
    with open(input_file, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            # 如果当前行号 < 已记录进度，则跳过
            if i < start_line:
                continue
            
            line = line.strip()
            know_list = linecache.getline(knowledge_file, i + 1).strip()
            
            # 构造 prompt
            prompt_temp = (
                "Now you need to evaluate whether a teacher-student teaching dialogue is relevant to the teaching content of this lesson."
                f"The teaching content of this lesson is as follows: {know_list}\n"
                f"The teacher-student teaching dialogue is as follows: {line}\n"
                "Your output format should be in one line, [Relevance]: [Your score (0 to 5 points, minimum 0, maximum 5)]."
            )
            
            try:
                # 调用API
                response = get_chat_gpt_response(prompt_temp)
                content = response["choices"][0]["message"]["content"]
                
                # 打印并写入文件
                print(f"[行 {i}] {content}")
                with open(output_file, "a+", encoding="utf-8") as outfile:
                    outfile.write(content + "\n")
                
                # 2. 成功处理后更新进度文件
                with open(progress_file, "w", encoding="utf-8") as pf:
                    pf.write(str(i + 1))
            
            except Exception as e:
                # 出错时提示并中断循环；下次启动从进度文件继续
                print(f"[错误] 第 {i} 行处理失败: {e}")
                print("下次运行脚本时将从此处继续。")
                break

if __name__ == "__main__":
    main()