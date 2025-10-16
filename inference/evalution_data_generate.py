from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import BitsAndBytesConfig

device = "cuda"  # the device to load the model onto

model_teacher_path = "xxx"
model_s1_path = "xxx"
model_s2_path = "xxx"
model_s3_path = "xxx"

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Load the models and tokenizers
with init_empty_weights():
    model_teacher = AutoModelForCausalLM.from_pretrained(model_teacher_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)
tokenizer_teacher = AutoTokenizer.from_pretrained(model_teacher_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)

model_student1 = AutoModelForCausalLM.from_pretrained(model_s1_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)
tokenizer_student1 = AutoTokenizer.from_pretrained(model_s1_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)

model_student2 = AutoModelForCausalLM.from_pretrained(model_s2_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)
tokenizer_student2 = AutoTokenizer.from_pretrained(model_s2_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)

model_student3 = AutoModelForCausalLM.from_pretrained(model_s3_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)
tokenizer_student3 = AutoTokenizer.from_pretrained(model_s3_path, device_map="auto", 
                              trust_remote_code=True, quantization_config=quantization_config)


with open('xxx', 'r', encoding='utf-8') as file:
    for line in file:
        
        knowlege_point = line.strip()

        # Initialize the messages with the initial prompts
        messages_teacher = [{"role": "system", "content": "You are a high school math teacher (T). Your role is to help your students master mathematical concepts."}]
        messages_teacher.append({"role": "user", "content":"The content of this lesson is" + knowlege_point})
        messages_student1 = [{"role": "system", "content": "You are a high school student (S1) with excellent mathematics grades, and your task is to learn and master mathematical knowledge."}]
        messages_student2 = [{"role": "system", "content": "You are a high school student (S2) with medium mathematics grades, and your task is to learn and master mathematical knowledge."}]
        messages_student3 = [{"role": "system", "content": "You are a high school student (S3) with struggle mathematics grades, and your task is to learn and master mathematical knowledge."}]

        # Function to generate a response from a model
        def generate_response(messages, model, tokenizer):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024  # Reduce the number of new tokens to further reduce memory usage
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response

        # Function to run multi-turn dialogue between teacher and student
        def run_dialogue(turns=12):  # Reduce the number of turns to reduce memory usage
            for turn in range(turns):
                # Generate response from the teacher model
                teacher_response = generate_response(messages_teacher, model_teacher, tokenizer_teacher)
                messages_teacher.append({"role": "assistant", "content": teacher_response})
                
                if turn == 0:
                    # First turn, students only see the teacher's response
                    messages_student1.append({"role": "user", "content": "(T): " + teacher_response})
                    messages_student2.append({"role": "user", "content": "(T): " + teacher_response})
                    messages_student3.append({"role": "user", "content": "(T): " + teacher_response})

                    student_response1 = generate_response(messages_student1, model_student1, tokenizer_student1)
                    student_response2 = generate_response(messages_student2, model_student2, tokenizer_student2)
                    student_response3 = generate_response(messages_student3, model_student3, tokenizer_student3)
                    
                    messages_student1.append({"role": "assistant", "content": student_response1})
                    messages_student2.append({"role": "assistant", "content": student_response2})
                    messages_student3.append({"role": "assistant", "content": student_response2})

                else:
                    # From second turn onwards, students see both teacher's and the other student's responses
                    messages_student1.append({"role": "user", "content": "(T): " + teacher_response})
                    messages_student2.append({"role": "user", "content": "(T): " + teacher_response})
                    messages_student3.append({"role": "user", "content": "(T): " + teacher_response})

                    messages_student1.append({"role": "user", "content": "(S2): " + student_response2})
                    messages_student1.append({"role": "user", "content": "(S3): " + student_response3})

                    messages_student2.append({"role": "user", "content": "(S1): " + student_response1})
                    messages_student2.append({"role": "user", "content": "(S3): " + student_response3})

                    messages_student3.append({"role": "user", "content": "(S1): " + student_response1})
                    messages_student3.append({"role": "user", "content": "(S2): " + student_response2})
                    
                    student_response1 = generate_response(messages_student1, model_student1, tokenizer_student1)
                    student_response2 = generate_response(messages_student2, model_student2, tokenizer_student2)
                    student_response3 = generate_response(messages_student3, model_student3, tokenizer_student3)
                    
                    messages_student1.append({"role": "assistant", "content": student_response1})
                    messages_student2.append({"role": "assistant", "content": student_response2})
                    messages_student3.append({"role": "assistant", "content": student_response3})

                # Teacher responds to both students' messages
                messages_teacher.append({"role": "user", "content": "(S1): " + student_response1})
                messages_teacher.append({"role": "user", "content": "(S2): " + student_response2})
                messages_teacher.append({"role": "user", "content": "(S3): " + student_response3})
                # Print the dialogue
                print(f"(T): {teacher_response}")
                print(f"(S1): {student_response1}")
                print(f"(S2): {student_response2}")
                print(f"(S3): {student_response3}")

        run_dialogue(turns=15)

        with open("xxx",'a+',encoding = 'utf=8') as files:
            files.write(str(messages_teacher))
            files.write('\n')