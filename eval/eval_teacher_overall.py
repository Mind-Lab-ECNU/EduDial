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
    output_file = 'overall.txt'
    
    # 进度记录文件路径
    progress_file = 'progress_log.txt'

    # 1. 读取进度文件，获取上次处理到的行号
    start_line = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as pf:
            line_str = pf.read().strip()
            if line_str.isdigit():
                start_line = int(line_str)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            # 2. 如果当前行号小于上次的处理行号，跳过该行
            if i < start_line:
                continue

            line = line.strip()
            know_list = linecache.getline(knowledge_file, i + 1).strip()
            
            prompt_temp = "Now you need to conduct an overall content evaluation of a teacher-student teaching dialogue. There are nine evaluation metrics in total, which are:\
                [1. Insight - Definition: Evaluate whether the teacher can accurately capture and deeply understand students' learning needs, knowledge levels, and problem intentions. Detailed Description: - Need Identification: Does the teacher identify students' specific learning needs and difficulties through questioning, observation, or other methods. - Problem Analysis: Does the teacher analyze the problems raised by students and understand the underlying learning obstacles or misconceptions. - Personalized Understanding: Does the teacher demonstrate an understanding of each student's unique learning style and needs.];\
                [2. Response - Definition: Evaluate whether the teacher can effectively solve students' problems and provide practical and constructive guidance. Detailed Description: - Problem Solving: Does the teacher provide clear and effective solutions to students' problems. - Guidance Specificity: Are the suggestions provided by the teacher specific, actionable, and able to help students make actual improvements. - Resource Provision: Does the teacher recommend relevant resources (such as textbooks, practice problems, reference materials) to support students' further learning.];\
                [3. Feedback - Definition: Evaluate whether the teacher can provide timely and effective feedback to students to help them improve their learning. Detailed Description: - Timeliness: Is feedback provided promptly after students raise questions, avoiding delays in students' learning progress. - Constructiveness: Is the feedback content constructive and able to guide students on how to improve. - Two-way Communication: Does the teacher encourage students to respond to feedback, promoting two-way communication.];\
                [4. Thinking - Definition: Evaluate whether the teacher can stimulate students' critical thinking abilities, including analytical skills, open-mindedness, and self-assessment abilities. Detailed Description: - Question Guidance: Does the teacher guide students to think deeply through open-ended questions. - Analytical Training: Does the teacher cultivate students' analytical abilities, helping them break down complex problems. - Self-Assessment: Does the teacher encourage students to engage in self-reflection and evaluation to enhance autonomous learning abilities.];\
                [5. Interactivity - Definition: Evaluate the frequency and quality of the teacher's interaction with students in teaching dialogue, including questioning techniques and methods of responding to student feedback. Detailed Description: - Questioning Techniques: Does the teacher use effective questioning methods to promote student thinking and participation. - Feedback Response: Does the teacher actively respond to student feedback, promoting continuous interaction. - Interaction Frequency: Is the interaction between teacher and students frequent, avoiding one-way knowledge transmission.];\
                [6. Emotional Support - Definition: Evaluate whether the teacher can provide emotional support during the teaching process, establish a positive learning environment, and help students build confidence and a positive learning attitude. Detailed Description: - Emotional Care: Does the teacher pay attention to students' emotional states and provide necessary care and support. - Motivation and Encouragement: Does the teacher stimulate students' learning motivation and confidence through encouragement and praise. - Building Trust: Does the teacher establish a trusting relationship with students, making them feel safe and respected.];\
                [7. Adaptability - Definition: Evaluate the teacher's ability to adjust teaching methods according to different students' learning styles and needs to ensure that each student receives personalized guidance. Detailed Description: - Teaching Method Adjustment: Does the teacher adjust teaching strategies and methods based on student feedback and learning progress. - Personalized Guidance: Can the teacher provide personalized guidance and support based on different students' characteristics. - Flexible Response: Can the teacher flexibly handle unexpected situations in the classroom to ensure teaching effectiveness.];\
                [8. Fluency - Definition: Evaluate whether the teacher's expression is clear, easy to understand, and has a natural and smooth tone. Detailed Description: - Language Expression: Is the language used by the teacher accurate and concise, avoiding vague or obscure expressions. - Logical Structure: Does the teacher's explanation have a clear logical structure that is easy for students to understand. - Natural Tone: Is the teacher's tone friendly and patient during communication, creating a good communication atmosphere.];\
                [9. Goal - Definition: Evaluate whether the teacher's guidance helps achieve predetermined teaching objectives, such as knowledge mastery and ability cultivation. Detailed Description: - Clear Objectives: Are specific learning objectives clearly set for teaching activities. - Objective Alignment: Are the teacher's teaching behaviors and guidance consistent with the set objectives. - Outcome Assessment: Does the teacher verify whether students have achieved predetermined objectives through assessment methods.];\
                The scoring range for each metric is 0 to 5 points, with a minimum score of 0 and a maximum score of 5. The content of this lesson is " + know_list + "\
                The teacher-student teaching dialogue is as follows: " + line + "\
                Your output format should be in one line, [Insight]: [Your score (0 to 5 points)], [Response]: [Your score (0 to 5 points)]..., [Goal]: [Your score (0 to 5 points)]."

            
            try:
                # 调用API获取评估结果
                response = get_chat_gpt_response(prompt_temp)
                content = response['choices'][0]['message']['content']
                
                print(f"[行 {i}] {content}")
                
                # 将结果写入输出文件
                with open(output_file, 'a+', encoding='utf-8') as outfile:
                    outfile.write(content + '\n')
                
                # 3. 每处理完一行，更新进度文件，记录当前行号
                with open(progress_file, 'w', encoding='utf-8') as pf:
                    pf.write(str(i + 1))
            
            except Exception as e:
                # 出现异常时打印错误并跳过当前行
                print(f"[错误] 第{i}行处理失败: {e}")
                print("下次运行脚本时将从此处继续。")
                break

if __name__ == "__main__":
    main()