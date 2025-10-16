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
    output_file = 'coverage.txt'
    
    # 用于记录进度的文件路径
    progress_file = 'progress_log.txt'

    # 1. 先读取上一次处理到的行号
    start_line = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as pf:
            line_str = pf.read().strip()
            if line_str.isdigit():
                start_line = int(line_str)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            # 2. 如果当前行号还没到上次进度，则跳过
            if i < start_line:
                continue
            
            line = line.strip()
            know_list = linecache.getline(knowledge_file, i + 1).strip()
            
            prompt_temp = (
                "Now you need to evaluate how much of the teaching content of this lesson is covered by a teacher-student teaching dialogue."
                f"The teaching content of this lesson is as follows: {know_list}\n"
                f"The teacher-student teaching dialogue is as follows: {line}\n"
                "Your output format should be in one line, [Coverage]: [Your assessment (0% to 100%)]."
            )
            
            try:
                response = get_chat_gpt_response(prompt_temp)
                content = response['choices'][0]['message']['content']
                
                print(f"[行 {i}] {content}")
                
                # 写入输出文件
                with open(output_file, 'a+', encoding='utf-8') as outfile:
                    outfile.write(content + '\n')
                
                # 3. 成功处理后更新进度
                with open(progress_file, 'w', encoding='utf-8') as pf:
                    pf.write(str(i + 1))
            
            except Exception as e:
                # 如果出错，可以先打印错误信息，然后 break 或者 raise
                print(f"[错误] 第{i}行处理失败: {e}")
                print("下次运行脚本时将从此处继续。")
                break

if __name__ == "__main__":
    main()