from datasets import load_dataset

dataset = load_dataset("sem_eval_2014_task_1", split="train")
print(dataset)