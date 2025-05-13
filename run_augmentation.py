import pandas as pd


df = pd.read_csv("./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/train.tsv", sep='\t')


from augment_dataframe import create_openai_client, augment_dataframe
client = create_openai_client( 
    api_key="sk-xxxxx", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

augment_dataframe(
    df=df,
    client=client,
    output_file= "./data/augment_data/augmented_train_dataset.tsv",
    model="qwen-turbo-latest",
    num_augments_per_sample=2,
    num_threads=20
)
