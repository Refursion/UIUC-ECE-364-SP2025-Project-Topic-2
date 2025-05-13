import pandas as pd
import threading
import queue
import openai
import re
import html
from tqdm import tqdm

def clean_tweet(text):
    text = html.unescape(text)
    text = re.sub(r'@USER', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_openai_client(api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client

def augment_tweet(
    client,
    tweet: str,
    model: str = "qwen-turbo-latest",
    num_outputs: int = 3,
    temperature: float = 0.7,
    max_retry: int = 3
):

    system_prompt = (
        f"You are an expert in data augmentation for English tweets.\n"
        f"Your task is to rephrase input tweets while strictly following these rules:\n"
        "- Preserve the original meaning and sentiment\n"
        "- Avoid altering factual information\n"
        "- Use natural English writing style suitable for Twitter\n"
        "- Output only the rephrased versions without any explanations, introductions, or conclusions\n"
        "- You can change the tweet structure instaed of mere synonym replacement while keeping all the information expressed, especially specific numbers or pisitions\n"
        f"For each input tweet, generate {num_outputs} rephrased versions.\n"
        "List each version with a number prefix (e.g., 1., 2., 3.).\n"
    )


    user_prompt = (
        f"Tweet:\n{tweet}"
    )

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=500,
                n=1,
            )
            content = response.choices[0].message.content
            return parse_outputs(content, expected_num=num_outputs)
        except Exception as e:
            print(f"Error augmenting tweet (attempt {attempt + 1}): {e}")
            import time
            time.sleep(2 * (attempt + 1))
    return []


def parse_outputs(content: str, expected_num: int = 3):
    lines = content.split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if line:
            if line[0].isdigit() and (line[1] == '.' or line[1] == '、' or line[1] == ' '):
                line = line[2:].strip()
            results.append(line)
    return results[:expected_num]


def augment_dataframe(
    df: pd.DataFrame,
    client: openai.OpenAI,
    output_file: str,
    model: str = "qwen-turbo-latest",
    num_augments_per_sample: int = 3,
    num_threads: int = 20,
    save_every: int = 500
):
    task_queue = queue.Queue()
    result_list = []
    result_lock = threading.Lock()
    pbar = tqdm(total=len(df), desc="Augmenting Tweets")

    for idx, row in df.iterrows():
        text = row["Text"]
        label = row["Label"]
        cleaned_text = clean_tweet(text)
        task_queue.put((cleaned_text, label))

    def worker():
        while not task_queue.empty():
            try:
                text, label = task_queue.get_nowait()
            except queue.Empty:
                break

            augmented_versions = augment_tweet(
                client, text, model=model, num_outputs=num_augments_per_sample
            )

            local_results = [{"Text": text, "Label": label}]  
            for aug_text in augmented_versions:
                aug_text_cleaned = clean_tweet(aug_text)
                local_results.append({"Text": aug_text_cleaned, "Label": label})

            with result_lock:
                result_list.extend(local_results)
                if len(result_list) % save_every == 0:
                    temp_df = pd.DataFrame(result_list)
                    temp_df.to_csv(output_file, sep='\t', index=False)

            pbar.update(1)
            task_queue.task_done()


    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


    df_result = pd.DataFrame(result_list)
    df_result.to_csv(output_file, sep='\t', index=False)
    print(f"Saved {len(df_result)} samples to {output_file}")

